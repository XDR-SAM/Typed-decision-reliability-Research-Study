"""CPU post-hoc fits with explicit temporal contracts; never retrain weights."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp,softmax
from scipy.stats import beta
from src.neural.protocol import check_role,load_config

def validated(logits,y):
    z=np.asarray(logits,dtype=np.float64);y=np.asarray(y,dtype=int)
    if z.ndim!=2 or len(z)!=len(y) or not np.isfinite(z).all():
        raise ValueError('Finite aligned raw logits required')
    if len(y)==0 or y.min()<0 or y.max()>=z.shape[1]: raise ValueError('Invalid labels')
    return z,y

def previous_quarter(q):
    year=int(q[:4]);number=int(q[-1])
    return f'{year-1}Q4' if number==1 else f'{year}Q{number-1}'

def calendar_quarters(frame):
    return frame.date.astype(str).map(lambda d:d[:4]+'Q'+str((int(d[5:7])-1)//3+1))

def fit_temperature(logits,y,metadata,fit_role='temperature_fit',target_quarter=None):
    z,y=validated(logits,y)
    if len(metadata)!=len(y): raise ValueError('Metadata/logit mismatch')
    if target_quarter is None:
        if fit_role!='temperature_fit':raise ValueError('Initial temperature fit is October-only')
        check_role(metadata,fit_role)
    else:
        allowed=load_config()['future_quarters']+load_config()['exploratory_quarters']
        if target_quarter not in allowed:raise ValueError('Unplanned maintenance target quarter')
        if not calendar_quarters(metadata).eq(previous_quarter(target_quarter)).all():
            raise ValueError('Rolling fit requires immediately preceding quarter only')
    cfg=load_config('calibration')['temperature']
    if len(y)<cfg['minimum_rows'] or len(np.unique(y))<cfg['minimum_classes']:
        return {'temperature':1.0,'success':False,'reason':'insufficient support'}
    def loss(log_t):
        v=z/np.exp(log_t)
        return float(np.mean(logsumexp(v,axis=1)-v[np.arange(len(y)),y]))
    lo,hi=np.log(cfg['bounds'])
    res=minimize_scalar(loss,bounds=(lo,hi),method='bounded',options={'xatol':1e-9})
    if not res.success or not np.isfinite(res.fun):
        return {'temperature':1.0,'success':False,'reason':'optimizer failure'}
    return {'temperature':float(np.exp(res.x)),'success':True,'n':len(y),
        'fit_role':fit_role,'target_quarter':target_quarter,'nll':float(res.fun),
        'at_boundary':bool(min(res.x-lo,hi-res.x)<1e-4)}

def probabilities(logits,temperature=1.0):
    if not np.isfinite(temperature) or temperature<=0: raise ValueError('Positive finite temperature required')
    return softmax(np.asarray(logits,dtype=np.float64)/temperature,axis=1)

def nll_from_logits(logits,y,temperature=1.0):
    z,y=validated(logits,y)
    if temperature<=0 or not np.isfinite(temperature):raise ValueError('Positive finite temperature required')
    z=z/temperature
    return float(np.mean(logsumexp(z,axis=1)-z[np.arange(len(y)),y]))

def adaptive_ece(y,p,bins=15):
    confidence=p.max(1);correct=p.argmax(1)==y
    # Quantile edges approximate equal mass; never split equal-confidence ties.
    edges=np.unique(np.quantile(confidence,np.linspace(0,1,bins+1)))
    group=np.searchsorted(edges[1:-1],confidence,side='right')
    rows=[];ece=0.0
    for b in np.unique(group):
        mask=group==b
        acc=float(correct[mask].mean());c=float(confidence[mask].mean())
        ece+=float(mask.mean())*abs(acc-c)
        rows.append({'bin':int(b),'n':int(mask.sum()),'accuracy':acc,'mean_confidence':c})
    return float(ece),rows

def clopper_pearson(correct,total,alpha=.05,one_sided=False):
    if total==0: return None
    tail=alpha if one_sided else alpha/2
    lower=0.0 if correct==0 else float(beta.ppf(tail,correct,total-correct+1))
    upper=1.0 if correct==total else float(beta.ppf(1-tail,correct+1,total-correct))
    return [lower,upper]

def select_gate(logits,y,metadata,temperature):
    check_role(metadata,'gate_selection')
    if not metadata.family.is_unique: raise ValueError('Gate selection requires one row per family')
    z,y=validated(logits,y);p=probabilities(z,temperature)
    cfg=load_config('calibration')['gate'];grid=np.r_[np.arange(50,100)/100,.995]
    correct=p.argmax(1)==y;conf=p.max(1);rows=[]
    for threshold in grid:
        mask=conf>=threshold;n=int(mask.sum());k=int(correct[mask].sum())
        interval=clopper_pearson(k,n,alpha=cfg['alpha']/len(grid),one_sided=True)
        coverage=float(mask.mean())
        qualifies=(n>=cfg['minimum_accepted_families'] and coverage>=cfg['minimum_coverage']
                   and interval is not None and interval[0]>=cfg['target_accuracy'])
        rows.append({'threshold':float(threshold),'accepted':n,'coverage':coverage,
            'accuracy':k/n if n else None,'simultaneous_lower_bound':None if interval is None else interval[0],
            'qualifies':qualifies})
    eligible=[r for r in rows if r['qualifies']]
    chosen=min(eligible,key=lambda r:(-r['coverage'],r['threshold'])) if eligible else None
    return {'threshold':None if chosen is None else chosen['threshold'],
        'status':'selected' if chosen else 'no_qualifying_gate','selected':chosen,'candidates':rows,
        'target_accuracy':cfg['target_accuracy'],'minimum_coverage':cfg['minimum_coverage'],
        'warning':'Family representatives are a dependence proxy, not proof of independent consumers; December verification is required.'}
