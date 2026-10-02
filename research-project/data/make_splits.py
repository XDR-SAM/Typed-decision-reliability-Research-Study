import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils import ROOT

def main():
    df=pd.read_pickle(ROOT/'data/processed/debt_deduplicated.pkl')
    df['train_keep']=(df.period=='train')&df.included&~df.duplicated('exact_hash')
    historical=set(df[df.period.isin(['train','validation'])].family)
    df['novel_family']=~df.family.isin(historical)
    df['family_representative']=~df.duplicated(['period','family'])
    df.to_pickle(ROOT/'data/processed/pilot.pkl')
    df[['complaint_id','date','period','issue','included','family','train_keep','novel_family','family_representative']].to_csv(ROOT/'outputs/audits/split_manifest.csv.gz',index=False)
    print(df.groupby('period').size().to_dict())
if __name__=='__main__':main()
