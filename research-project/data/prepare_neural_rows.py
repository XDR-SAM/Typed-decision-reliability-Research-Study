"""Export immutable next-stage rows; no vectorizer, predictive fit or GPU calls."""
import sys, json, hashlib
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils import ROOT,dump_json
from src.neural.protocol import role,labels_for,disjoint

def apply_roles(df):
    df=df.sort_values(['date','complaint_id']).copy()
    df['role']=df.date.map(role)
    df['calibration_role_eligible']=False
    seen=set(df.loc[df.role.isin(['train','validation']),'family'].astype(str))
    for name in ['temperature_fit','gate_selection','gate_verification']:
        g=df[(df.role==name)&df.included]
        mask=~g.family.astype(str).isin(seen) & ~g.duplicated('family')
        df.loc[g.index[mask],'calibration_role_eligible']=True
        # Exclude ALL earlier role families, including discarded repeats.
        seen.update(g.family.astype(str))
    df['strict_novel_family']=False
    seen=set()
    for name,g in df.groupby('period',sort=False):
        mask=~g.family.astype(str).isin(seen) & ~g.duplicated('family')
        df.loc[g.index[mask],'strict_novel_family']=True
        seen.update(g.family.astype(str))
    parts=[df[(df.role==name)&df.calibration_role_eligible]
           for name in ['temperature_fit','gate_selection','gate_verification']]
    disjoint(*parts);disjoint(*parts,families=True)
    return df

def main():
    summaries={}
    for domain in ['debt_collection','credit_card']:
        out=ROOT/'outputs/neural_preparation'/domain;out.mkdir(parents=True,exist_ok=True)
        if domain=='debt_collection':
            # Portable JSONL + immutable pilot split metadata; avoids needing the
            # pilot's Python 3.14 / pandas 3 pickle implementation on Kaggle.
            df=pd.read_json(ROOT/'data/processed/debt_collection.jsonl.gz',lines=True,convert_dates=False)
            meta=pd.read_csv(ROOT/'outputs/audits/split_manifest.csv.gz',dtype={'complaint_id':str})
            df['complaint_id']=df.complaint_id.astype(str)
            derived=['complaint_id','included','family','train_keep','novel_family','family_representative']
            df=df.merge(meta[derived],on='complaint_id',validate='one_to_one')
            df['family']='dc:'+df.family.astype(str)
        else:
            df=pd.read_json(out/'rows.jsonl.gz',lines=True,convert_dates=False)
        df=apply_roles(df)
        df.to_json(out/'rows.jsonl.gz',orient='records',lines=True,compression='gzip',force_ascii=False)
        df[df.train_keep].to_json(out/'train.jsonl.gz',orient='records',lines=True,compression='gzip',force_ascii=False)
        df[(df.role=='validation')&df.included].to_json(out/'validation.jsonl.gz',orient='records',lines=True,compression='gzip',force_ascii=False)
        df.drop(columns='narrative').to_csv(out/'split_manifest.csv.gz',index=False)
        keep=df[df.included]
        support=keep.groupby(['role','issue']).agg(n=('complaint_id','size'),families=('family','nunique'),
             calibration_eligible=('calibration_role_eligible','sum')).reset_index()
        support.to_csv(out/'role_support.csv',index=False)
        summaries[domain]={'labels':labels_for(domain),'rows':len(df),'training_rows':int(df.train_keep.sum()),
            'roles':keep.groupby('role').size().to_dict(),
            'calibration_roles':keep[keep.calibration_role_eligible].groupby('role').size().to_dict(),
            'rows_sha256':hashlib.sha256((out/'rows.jsonl.gz').read_bytes()).hexdigest()}
    dump_json(ROOT/'outputs/neural_preparation/row_manifest.json',summaries)
    print(json.dumps(summaries,indent=2))

if __name__=='__main__': main()
