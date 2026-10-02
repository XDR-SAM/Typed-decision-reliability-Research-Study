import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,precision_recall_fscore_support,confusion_matrix,roc_auc_score
from src.calibration_metrics import probability_metrics
from src.selective_metrics import selective

def evaluate(y,p,classes,thresholds):
    pred=p.argmax(1);correct=pred==y
    metrics={'n':len(y),'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,labels=np.arange(len(classes)),average='macro',zero_division=0)),
             'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
             'correctness_auc':float(roc_auc_score(correct,p.max(1))) if len(np.unique(correct))==2 else None}
    prob,bins=probability_metrics(y,p);metrics.update(prob)
    pr,rec,f,sup=precision_recall_fscore_support(y,pred,labels=np.arange(len(classes)),zero_division=0)
    perclass=[{'issue':c,'precision':float(pr[i]),'recall':float(rec[i]),'f1':float(f[i]),'support':int(sup[i])} for i,c in enumerate(classes)]
    sel,curve=selective(y,p,thresholds)
    return metrics,perclass,confusion_matrix(y,pred,labels=np.arange(len(classes))),bins,sel,curve
