"""Historical calibration, unchanged gates and preceding-quarter maintenance.

Run only on saved neural logits after training; no model weights are modified.
"""
import argparse,json
import numpy as np
import pandas as pd
from src.utils import ROOT,dump_json
from src.evaluate import evaluate
from src.neural.protocol import load_config,labels_for,disjoint,verify_freeze
from src.neural.calibration import fit_temperature,probabilities,select_gate,adaptive_ece,previous_quarter,clopper_pearson,nll_from_logits
from src.neural.statistics import cluster_ratios

def frame_logits(g,k): return g[[f'z_{i}' for i in range(k)]].to_numpy(dtype=float)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--domain',required=True,choices=['debt_collection','credit_card'])
    parser.add_argument('--model-key',required=True,choices=['laya_ce','laya_rlcd','matched_head','modernbert'])
    parser.add_argument('--seed',required=True,type=int,choices=[17,43,101]);args=parser.parse_args()
    verify_freeze();cfg=load_config('calibration');labels=labels_for(args.domain);k=len(labels)
    folder=ROOT/'outputs/neural'/args.domain/args.model_key/f'seed_{args.seed}'
    out=folder/'posthoc'
    if (out/'complete.json').exists():raise RuntimeError('Post-hoc outputs already complete; no silent overwrite')
    out.mkdir(parents=True,exist_ok=True)
    def read(period):
        return pd.read_csv(folder/f'raw_{period}.csv.gz',dtype={'complaint_id':str,'family':str})
    def truth(g):return g.issue.map({x:i for i,x in enumerate(labels)}).to_numpy()
    q4=read('2024Q4');parts=[]
    for role in ['temperature_fit','gate_selection','gate_verification']:
        parts.append(q4[(q4.role==role)&q4.calibration_role_eligible])
    disjoint(*parts);disjoint(*parts,families=True)
    temp=fit_temperature(frame_logits(parts[0],k),truth(parts[0]),parts[0])
    gate=select_gate(frame_logits(parts[1],k),truth(parts[1]),parts[1],temp['temperature'])
    dump_json(out/'frozen_temperature.json',temp);dump_json(out/'frozen_gate.json',gate)
    verification={'role':'gate_verification','threshold':gate['threshold'],'reselected':False}
    if gate['threshold'] is not None:
        v=parts[2];p=probabilities(frame_logits(v,k),temp['temperature']);mask=p.max(1)>=gate['threshold']
        n=int(mask.sum());correct=int((p.argmax(1)[mask]==truth(v)[mask]).sum())
        interval=clopper_pearson(correct,n,alpha=.05,one_sided=True)
        verification.update({'accepted':n,'coverage':float(mask.mean()),'accepted_accuracy':correct/n if n else None,
            'lower_95':None if interval is None else interval[0],
            'verified':bool(n>=200 and mask.mean()>=.1 and interval is not None and interval[0]>=.9)})
    else:verification['verified']=False
    dump_json(out/'december_verification.json',verification)
    metrics=[];classes=[];selective_rows=[];curves=[];bins=[];temps=[];group_gates=[]
    periods=load_config()['future_quarters']+load_config()['exploratory_quarters']
    rng_seed=cfg['maintenance']['sampling_seed']
    for window in periods:
        g=read(window);y=truth(g);z=frame_logits(g,k)
        prev=read(previous_quarter(window));yp=truth(prev);zp=frame_logits(prev,k)
        disjoint(prev,g)
        full=fit_temperature(zp,yp,prev,fit_role='rolling_full',target_quarter=window)
        variants=[('raw',1.0,None),('frozen',temp['temperature'],None),('rolling_full',full['temperature'],None)]
        temps.append({'period':window,'arm':'rolling_full',**full})
        for draw in range(cfg['maintenance']['draws']):
            # Same nested samples for all model families/seeds; no oracle class strata.
            quarter_number=int(window[:4])*4+int(window[-1])
            order=np.random.default_rng(rng_seed+quarter_number*1000+draw).permutation(len(prev))
            for budget in cfg['maintenance']['budgets']:
                if len(prev)<budget:
                    temps.append({'period':window,'arm':f'rolling_{budget}','draw':draw,'success':False,'reason':'budget unavailable'});continue
                ix=order[:budget];sample=prev.iloc[ix]
                fit=fit_temperature(zp[ix],yp[ix],sample,fit_role=f'rolling_{budget}',target_quarter=window)
                temps.append({'period':window,'arm':f'rolling_{budget}','draw':draw,'labeled_records':budget,
                    'labeled_families':sample.family.nunique(),**fit})
                variants.append((f'rolling_{budget}',fit['temperature'],draw))
        for arm,t,draw in variants:
            p=probabilities(z,t)
            thresholds=cfg['gate']['diagnostic_thresholds']+([gate['threshold']] if gate['threshold'] is not None else [])
            subsets={'operational':np.ones(len(g),dtype=bool),'strict_novel':g.strict_novel_family.to_numpy(),
                'pilot_novel':g.novel_family.to_numpy()&g.family_representative.to_numpy()}
            for subset,mask in subsets.items():
                if not mask.any():continue
                met,pc,cm,rb,sels,curve=evaluate(y[mask],p[mask],labels,thresholds)
                met['nll']=nll_from_logits(z[mask],y[mask],t)
                ae,ab=adaptive_ece(y[mask],p[mask]);met['adaptive_ece']=ae
                keys={'period':window,'arm':arm,'draw':draw,'subset':subset,'temperature':t}
                metrics.append({**keys,**met});classes.extend({**keys,**r} for r in pc)
                selective_rows.extend({**keys,**r} for r in sels)
                # Detailed curves/bins/confusions for main arms; budget draws keep metrics.
                if draw is None:
                    curves.extend({**keys,**r} for r in curve)
                    bins.extend({**keys,'kind':'equal_width',**r} for r in rb)
                    bins.extend({**keys,'kind':'equal_mass_ties',**r} for r in ab)
                    pd.DataFrame(cm,index=labels,columns=labels).to_csv(out/f'confusion_{arm}_{window}_{subset}.csv')
            if draw is None and gate['threshold'] is not None:
                accepted=p.max(1)>=gate['threshold'];correct=p.argmax(1)==y
                indicators=np.column_stack([np.ones(len(y),dtype=bool)]+[y==j for j in range(k)])
                den=indicators*accepted[:,None];num=den*correct[:,None]
                interval,draws=cluster_ratios(g.family,num,den,replicates=load_config('statistics')['gate_bootstrap_replicates'],
                    alpha=.05/5 if window in load_config()['future_quarters'] else .05)
                for j,label in enumerate(['ALL']+labels):
                    support=int(indicators[:,j].sum());n=int(den[:,j].sum());good=int(num[:,j].sum())
                    nfamily=g.loc[den[:,j].astype(bool),'family'].nunique()
                    ci=interval[:,j].tolist() if n else None
                    if j>0 and n:ci=np.nanquantile(draws[:,j],[.025,.975]).tolist()
                    if n and nfamily<30:ci=[0.,1.]
                    # Subgroups are descriptive pointwise, never certified from this band.
                    group_gates.append({'period':window,'arm':arm,'issue':label,'support':support,'accepted':n,
                        'accepted_families':nfamily,'coverage':n/support if support else None,
                        'accepted_accuracy':good/n if n else None,'interval':ci,
                        'precision_flag':'empty' if not n else 'counts_only' if nfamily<30 else 'low_precision' if nfamily<100 else 'adequate_for_descriptive_estimation',
                        'interval_scope':'simultaneous_quarter_aggregate' if label=='ALL' else 'pointwise_95_descriptive_subgroup'})
    for name,rows in [('metrics',metrics),('per_class',classes),('selective',selective_rows),('risk_coverage',curves),
                       ('reliability',bins),('maintenance_fits',temps)]:pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    dump_json(out/'class_conditional_gate.json',group_gates)
    dump_json(out/'complete.json',{'complete':True,'weights_modified':False,'historical_gate':gate['status'],
        'temperature_success':temp['success'],'verification':verification,'labels':labels})

if __name__=='__main__':main()
