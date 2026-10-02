"""Fixed three-seed paired contrasts on cached predictions, CPU only."""
import argparse,json
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from src.utils import ROOT,dump_json
from src.neural.protocol import load_config,labels_for,verify_freeze
from src.neural.statistics import align_prediction_frames,paired_mean_loss_contrast,paired_macro_f1_contrast,holm

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--domain',choices=['debt_collection','credit_card'],required=True)
    args=parser.parse_args();verify_freeze();labels=labels_for(args.domain);cfg=load_config();quarters=cfg['future_quarters']
    keys=['laya_ce','matched_head','modernbert']+(['laya_rlcd'] if args.domain=='debt_collection' else [])
    reference=None;data={}
    for key in keys:
        losses=[];rolling=[];preds=[]
        for seed in cfg['seeds']:
            folder=ROOT/'outputs/neural'/args.domain/key/f'seed_{seed}'
            post=folder/'posthoc'
            if not (post/'complete.json').exists():raise RuntimeError('Complete every prespecified cached analysis first')
            temperature=json.loads((post/'frozen_temperature.json').read_text())['temperature']
            recent=pd.read_csv(post/'maintenance_fits.csv');recent=recent[recent.arm=='rolling_full']
            frame=pd.concat([pd.read_csv(folder/f'raw_{q}.csv.gz',dtype={'complaint_id':str,'family':str}) for q in quarters],ignore_index=True)
            if reference is None:reference=frame
            frame=align_prediction_frames(reference,frame)
            y=frame.issue.map({x:i for i,x in enumerate(labels)}).to_numpy()
            z=frame[[f'z_{j}' for j in range(len(labels))]].to_numpy(dtype=float)
            def loss(scale):
                zz=z/scale
                return logsumexp(zz,axis=1)-zz[np.arange(len(y)),y]
            losses.append(loss(temperature));preds.append(z.argmax(1))
            scale=frame.period.map(recent.set_index('period').temperature).to_numpy()[:,None]
            rolling.append(loss(scale))
        data[key]={'loss':np.array(losses),'rolling_loss':np.array(rolling),'pred':np.array(preds)}
    kwargs={'quarters':quarters,'replicates':load_config('statistics')['bootstrap_replicates']}
    primary=paired_mean_loss_contrast(reference,data['laya_ce']['loss'],data['matched_head']['loss'],**kwargs)
    secondary={
        'laya_minus_matched_macro_f1':paired_macro_f1_contrast(reference,y,data['laya_ce']['pred'],data['matched_head']['pred'],k=len(labels),**kwargs),
        'laya_minus_modernbert_NLL':paired_mean_loss_contrast(reference,data['laya_ce']['loss'],data['modernbert']['loss'],**kwargs),
        'laya_rolling_minus_frozen_NLL':paired_mean_loss_contrast(reference,data['laya_ce']['rolling_loss'],data['laya_ce']['loss'],**kwargs)}
    if 'laya_rlcd' in data:
        secondary['rlcd_minus_ce_NLL']=paired_mean_loss_contrast(reference,data['laya_rlcd']['loss'],data['laya_ce']['loss'],**kwargs)
    adjusted=holm([x['centered_bootstrap_p'] for x in secondary.values()])
    for result,p in zip(secondary.values(),adjusted):result['holm_p']=p
    out=ROOT/'outputs/neural'/args.domain/'paired_contrasts.json'
    if out.exists():raise RuntimeError('Do not silently overwrite completed comparisons')
    dump_json(out,{'domain':args.domain,'primary_Laya_minus_matched_frozen_NLL':primary,'secondary':secondary,
        'training_seeds':cfg['seeds'],'equal_quarter_weighting':True,'probability_ensemble':False})

if __name__=='__main__':main()
