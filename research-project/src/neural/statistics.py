"""Dependence-aware uncertainty and paired NLL contrasts (CPU)."""
import numpy as np
import pandas as pd

def align_prediction_frames(reference,prediction):
    if not reference.complaint_id.is_unique or not prediction.complaint_id.is_unique:
        raise ValueError('Prediction Complaint IDs must be unique')
    if set(reference.complaint_id)!=set(prediction.complaint_id):raise ValueError('Different evaluation examples')
    aligned=prediction.set_index('complaint_id').loc[reference.complaint_id].reset_index()
    for column in ['issue','date','period','family']:
        if not np.array_equal(reference[column].to_numpy(),aligned[column].to_numpy()):
            raise ValueError('Prediction metadata mismatch: '+column)
    return aligned

def paired_mean_loss_contrast(frame,loss_a,loss_b,quarters,replicates=2000):
    # loss matrices are [training_seeds, rows]; average losses, never probabilities.
    a=np.asarray(loss_a);b=np.asarray(loss_b)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=len(frame):raise ValueError('Aligned seed-by-row losses required')
    masks=np.column_stack([frame.period.to_numpy()==q for q in quarters]).astype(float)
    if (masks.sum(0)==0).any():raise ValueError('Missing planned quarter')
    delta=(a-b).mean(0);num=masks*delta[:,None]
    _,draws=cluster_ratios(frame.family,num,masks,replicates=replicates)
    observed=float(np.mean(num.sum(0)/masks.sum(0)));estimates=draws.mean(1)
    p=(1+np.sum(np.abs(estimates-observed)>=abs(observed)))/(len(estimates)+1)
    return {'difference':observed,'interval_95':np.quantile(estimates,[.025,.975]).tolist(),
        'centered_bootstrap_p':float(p),'seed_differences':[float(np.mean([(a[i]-b[i])[masks[:,j].astype(bool)].mean() for j in range(len(quarters))])) for i in range(len(a))]}

def paired_macro_f1_contrast(frame,y,pred_a,pred_b,quarters,k,replicates=2000,seed=20261002):
    a=np.asarray(pred_a);b=np.asarray(pred_b);y=np.asarray(y)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=len(frame):raise ValueError('Aligned seed-by-row predictions required')
    codes,families=pd.factorize(frame.family,sort=True);f=len(families)
    masks=[frame.period.to_numpy()==q for q in quarters]
    def value(weights):
        delta=[]
        for m in masks:
            truth=np.bincount(y[m],weights=weights[m],minlength=k)
            for left,right in zip(a,b):
                scores=[]
                for pred in [left,right]:
                    predicted=np.bincount(pred[m],weights=weights[m],minlength=k)
                    tp=np.bincount(y[m],weights=weights[m]*(pred[m]==y[m]),minlength=k)
                    scores.append(np.divide(2*tp,truth+predicted,out=np.zeros(k),where=truth+predicted>0).mean())
                delta.append(scores[0]-scores[1])
        return float(np.mean(delta))
    observed=value(np.ones(len(frame)));rng=np.random.default_rng(seed);draws=[]
    for _ in range(replicates):draws.append(value(rng.multinomial(f,np.full(f,1/f))[codes]))
    draws=np.array(draws);p=(1+np.sum(np.abs(draws-observed)>=abs(observed)))/(replicates+1)
    return {'difference':observed,'interval_95':np.quantile(draws,[.025,.975]).tolist(),'centered_bootstrap_p':float(p)}

def cluster_ratios(families,numerators,denominators,replicates=2000,seed=20261002,alpha=.05):
    """Resample entire global family IDs, preserving multiplicity of their rows.

    Matrices may have one column per quarter, model, seed or true class.
    Shared bootstrap weights keep comparisons paired and preserve cross-time links.
    """
    codes,unique=pd.factorize(np.asarray(families),sort=True);f=len(unique)
    a=np.asarray(numerators,dtype=float);b=np.asarray(denominators,dtype=float)
    if a.ndim==1:a=a[:,None]
    if b.ndim==1:b=b[:,None]
    if a.shape!=b.shape or a.shape[0]!=len(codes):raise ValueError('Aligned numerator/denominator matrices required')
    aa=np.column_stack([np.bincount(codes,weights=a[:,i],minlength=f) for i in range(a.shape[1])])
    bb=np.column_stack([np.bincount(codes,weights=b[:,i],minlength=f) for i in range(b.shape[1])])
    rng=np.random.default_rng(seed);draws=[]
    for first in range(0,replicates,32):
        weights=rng.multinomial(f,np.full(f,1/f),size=min(32,replicates-first))
        num=weights@aa;den=weights@bb
        draws.append(np.divide(num,den,out=np.full_like(num,np.nan),where=den>0))
    draws=np.vstack(draws)
    with np.errstate(invalid='ignore'):
        interval=np.nanquantile(draws,[alpha/2,1-alpha/2],axis=0)
    return interval,draws

def paired_nll_contrast(frame,logits_a,logits_b,y,quarters,replicates=2000):
    from scipy.special import logsumexp
    a=np.asarray(logits_a,dtype=float);b=np.asarray(logits_b,dtype=float);y=np.asarray(y,dtype=int)
    if a.shape!=b.shape or len(a)!=len(frame):raise ValueError('Models must be aligned on exact same rows')
    loss_a=logsumexp(a,axis=1)-a[np.arange(len(y)),y]
    loss_b=logsumexp(b,axis=1)-b[np.arange(len(y)),y]
    period=frame.period.to_numpy();m=np.column_stack([period==q for q in quarters]).astype(float)
    num=m*(loss_a-loss_b)[:,None]
    _,draws=cluster_ratios(frame.family,num,m,replicates=replicates)
    observed=float(np.mean(num.sum(0)/m.sum(0)))
    estimates=draws.mean(1)
    ci=np.quantile(estimates,[.025,.975]).tolist()
    centered=estimates-observed
    p=(1+np.sum(np.abs(centered)>=abs(observed)))/(len(centered)+1)
    return {'equal_quarter_NLL_difference':observed,'interval_95':ci,'centered_bootstrap_p':float(p)}

def holm(pvalues):
    p=np.asarray(pvalues);order=np.argsort(p);result=np.zeros(len(p));running=0.
    for j,i in enumerate(order):
        running=max(running,min(1.,(len(p)-j)*p[i]));result[i]=running
    return result.tolist()
