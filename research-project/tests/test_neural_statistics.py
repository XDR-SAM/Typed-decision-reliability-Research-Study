import numpy as np
import pandas as pd
from src.neural.statistics import cluster_ratios,paired_nll_contrast,holm,align_prediction_frames,paired_mean_loss_contrast,paired_macro_f1_contrast
import pytest

def test_clusters_stay_together_and_paired_null_is_zero():
    families=['a','a','b','c']
    ci,draws=cluster_ratios(families,np.ones(4),np.ones(4),replicates=50)
    assert np.allclose(ci,1) and np.allclose(draws,1)
    meta=pd.DataFrame({'family':families,'period':['2025Q1']*4})
    z=np.tile([1.,0.],(4,1))
    result=paired_nll_contrast(meta,z,z,np.zeros(4,dtype=int),['2025Q1'],replicates=50)
    assert result['equal_quarter_NLL_difference']==0 and result['interval_95']==[0.,0.]

def test_holm_adjustment():
    assert np.allclose(holm([.01,.04,.2]),[.03,.08,.2])

def test_id_alignment_rejects_changed_labels_and_f1_null():
    frame=pd.DataFrame({'complaint_id':['a','b'],'issue':['x','y'],'date':['2025-01-01']*2,'period':['2025Q1']*2,'family':['1','2']})
    assert align_prediction_frames(frame,frame.iloc[::-1]).complaint_id.tolist()==['a','b']
    with pytest.raises(ValueError):align_prediction_frames(frame,frame.assign(issue='wrong'))
    pred=np.array([[0,1],[0,1],[0,1]])
    result=paired_macro_f1_contrast(frame,np.array([0,1]),pred,pred,['2025Q1'],2,replicates=10)
    assert result['difference']==0 and result['interval_95']==[0.,0.]
    loss=np.array([[1.,2.],[2.,3.],[3.,4.]])
    result=paired_mean_loss_contrast(frame,loss,loss,['2025Q1'],replicates=10)
    assert result['difference']==0
