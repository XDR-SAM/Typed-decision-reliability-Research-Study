import pandas as pd
from src.utils import ROOT,period
def test_boundaries():
    assert period('2024-06-30')=='train'
    assert period('2024-07-01')=='validation'
    assert period('2024-09-30')=='validation'
    assert period('2024-10-01')=='2024Q4'
def test_actual_chronology_and_ids():
    df=pd.read_pickle(ROOT/'data/processed/pilot.pkl')
    assert df.complaint_id.is_unique
    assert df[df.train_keep].date.max()<df[df.period=='validation'].date.min()
    assert df[df.period=='validation'].date.max()<df[df.period=='2024Q4'].date.min()
    assert df.date.min()>='2023-09-01'
