import pandas as pd
from src.tfidf_baseline import text_inputs
from src.utils import ROOT
def test_metadata_cannot_change_features():
    a=pd.DataFrame({'narrative':['I received a letter.'],'issue':['secret target'],'product':['Debt collection'],'company':['ABC']})
    b=a.copy();b['issue']='INJECTED TARGET';b['product']='DIFFERENT';b['company']='XYZ'
    assert text_inputs(a)==text_inputs(b)==['I received a letter.']
def test_label_policy_is_historical_only():
    import yaml
    df=pd.read_pickle(ROOT/'data/processed/pilot.pkl')
    counts=df[df.period=='train'].issue.value_counts()
    expected={k for k,v in counts.items() if k and v>=200}
    conf=yaml.safe_load((ROOT/'configs/debt_collection_issues.yaml').read_text())
    assert {x['original_label'] for x in conf['labels'] if x['included']}==expected
