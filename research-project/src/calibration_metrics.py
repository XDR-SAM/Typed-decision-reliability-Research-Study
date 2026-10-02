import numpy as np
from sklearn.metrics import log_loss

def probability_metrics(y,p):
    y=np.asarray(y);p=np.asarray(p);n,k=p.shape
    pred=p.argmax(1);conf=p.max(1);correct=(pred==y)
    brier=np.mean(np.sum((p-np.eye(k)[y])**2,axis=1))
    bins=[];ece=0.
    for b in range(15):
        m=(conf>=b/15)&((conf<(b+1)/15) if b<14 else (conf<=1))
        if m.any():
            acc=float(correct[m].mean());c=float(conf[m].mean());ece+=m.mean()*abs(acc-c)
            bins.append({'bin':b,'n':int(m.sum()),'accuracy':acc,'mean_confidence':c})
    return {'nll':float(log_loss(y,p,labels=np.arange(k))),'brier':float(brier),'ece':float(ece),
            'mean_confidence':float(conf.mean()),'confidence_minus_accuracy':float(conf.mean()-correct.mean())},bins
