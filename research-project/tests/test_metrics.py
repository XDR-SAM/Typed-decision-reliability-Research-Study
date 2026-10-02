import numpy as np
from src.calibration_metrics import probability_metrics
from src.selective_metrics import selective
def test_probability_definitions():
    m,_=probability_metrics(np.array([0,1]),np.array([[1.,0.],[0.,1.]]))
    assert m['brier']==0 and m['ece']==0
    m,_=probability_metrics(np.array([0,1]),np.array([[.5,.5],[.5,.5]]))
    assert abs(m['brier']-.5)<1e-10 and abs(m['nll']-np.log(2))<1e-10
def test_empty_acceptance_is_unavailable_not_perfect():
    rows,curve=selective(np.array([0,1]),np.array([[.5,.5],[.5,.5]]),[.5,.9])
    assert rows[1]['accepted_accuracy'] is None
    assert len(curve)==1 and curve[0]['coverage']==1.
