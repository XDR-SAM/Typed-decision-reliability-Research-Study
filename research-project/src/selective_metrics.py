import numpy as np
def selective(y,p,thresholds):
    c=p.max(1);correct=p.argmax(1)==y;rows=[]
    for t in thresholds:
        m=c>=t;n=int(m.sum());acc=float(correct[m].mean()) if n else None
        rows.append({'threshold':t,'accepted':n,'coverage':float(m.mean()),'accepted_accuracy':acc,'risk':None if acc is None else 1-acc})
    # Threshold-realizable curve, ties kept together (no oracle tie ordering).
    order=np.argsort(-c,kind='stable');cs=c[order];cum=np.cumsum(~correct[order]);ends=np.r_[np.flatnonzero(cs[:-1]!=cs[1:]),len(cs)-1]
    if len(ends)>500:ends=ends[np.unique(np.linspace(0,len(ends)-1,500).astype(int))]
    curve=[{'coverage':float((j+1)/len(c)),'risk':float(cum[j]/(j+1)),'threshold':float(cs[j])} for j in ends]
    return rows,curve
