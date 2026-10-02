import pandas as pd
from data.deduplicate import tokens,shingles
from src.utils import ROOT
def test_template_normalization():
    assert tokens('Paid $123 to XXXX.')==tokens('paid $999 to XXXXX.')
def test_clean_training_and_novel_sensitivity():
    df=pd.read_pickle(ROOT/'data/processed/pilot.pkl')
    assert df[df.train_keep].exact_hash.is_unique
    hist=set(df[df.period.isin(['train','validation'])].family)
    future=df[df.novel_family&df.family_representative]
    assert not hist.intersection(set(future.family))
    assert not future.duplicated(['period','family']).any()
