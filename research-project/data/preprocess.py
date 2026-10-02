"""Historical-only class policy, label shifts, lengths and exact duplicate audit."""
import json,re,sys,unicodedata
from pathlib import Path
import numpy as np,pandas as pd,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils import ROOT,config,dump_json

def main():
    df=pd.read_json(ROOT/'data/processed/debt_collection.jsonl.gz',lines=True,convert_dates=False)
    df=df.sort_values(['date','complaint_id']).reset_index(drop=True)
    training=df[df.period=='train'];counts=training.issue.value_counts()
    included=sorted(i for i,n in counts.items() if i and n>=config()['training_label_minimum'])
    mapping=[]
    for issue in sorted(df.issue.unique()):
        future=df[df.issue==issue].groupby('period').size().to_dict()
        mapping.append({'original_label':issue,'included':issue in included,'training_count':int(counts.get(issue,0)),
                        'exclusion_reason':None if issue in included else 'Absent or <200 training narratives; no future-performance selection',
                        'period_counts':{k:int(v) for k,v in future.items()}})
    (ROOT/'configs/debt_collection_issues.yaml').write_text(yaml.safe_dump({'selection_basis':'training counts only; exact labels, no merges','labels':mapping},sort_keys=False,allow_unicode=True),encoding='utf8')
    df['included']=df.issue.isin(included)
    df['normalized_hash']=df.narrative.map(lambda x:__import__('hashlib').sha256(re.sub(r'\s+',' ',unicodedata.normalize('NFKC',x).lower()).encode()).hexdigest())
    df['exact_first_period']=df.groupby('exact_hash',sort=False).period.transform('first')
    df['exact_prior_period']=df.period!=df.exact_first_period
    df['exact_within_repeat']=df.duplicated(['period','exact_hash'])
    df['exact_historical_seen']=df.exact_hash.isin(set(df[df.period.isin(['train','validation'])].exact_hash))
    df.to_pickle(ROOT/'data/processed/debt_prepared.pkl')
    df.drop(columns='narrative').to_csv(ROOT/'outputs/audits/debt_metadata.csv.gz',index=False)
    table=df.groupby(['period','issue']).size().unstack(fill_value=0)
    table.to_csv(ROOT/'outputs/audits/debt_issue_counts.csv')
    table.div(table.sum(axis=1),axis=0).to_csv(ROOT/'outputs/audits/debt_issue_proportions.csv')
    rows=[]
    for p,g in df.groupby('period',sort=False):
        freq=g.issue.value_counts(normalize=True)
        r={'period':p,'n':len(g),'included':int(g.included.sum()),'unknown_or_excluded_fraction':float(1-g.included.mean()),
           'label_entropy_bits':float(-(freq*np.log2(freq)).sum()),'exact_unique':g.exact_hash.nunique(),
           'exact_within_repeats':int(g.exact_within_repeat.sum()),'exact_prior_period':int(g.exact_prior_period.sum())}
        for v in ['characters','words']:
            for label,value in [('mean',g[v].mean()),('median',g[v].median()),('p05',g[v].quantile(.05)),('p95',g[v].quantile(.95)),('max',g[v].max())]:r[v+'_'+label]=float(value)
        rows.append(r)
    pd.DataFrame(rows).to_csv(ROOT/'outputs/audits/debt_period_summary.csv',index=False)
    groups=df.groupby('exact_hash').agg(n=('issue','size'),labels=('issue','nunique'),periods=('period','nunique'),first_date=('date','min'),last_date=('date','max'))
    groups[groups.n>1].to_csv(ROOT/'outputs/audits/exact_duplicate_groups.csv')
    dump_json(ROOT/'outputs/audits/label_status.json',{'included':included,'new_after_training':sorted(set(df.issue)-set(counts.index)),
        'absent_by_period':{p:[i for i in included if table.loc[p,i]==0] for p in table.index},
        'renaming':'No automatic equivalence inferred; new strings are reported, never silently merged.',
        'exact_groups_with_conflicting_labels':int(((groups.n>1)&(groups.labels>1)).sum())})
    # This diagnostic phrase list is derived from label names alone, before model fitting.
    phrases=['debt not owed','collect debt not owed','false statements','false representation','written notification',
             'communication tactics','threatened to contact','share information improperly','electronic communications',
             'negative or legal action','threatened legal action']
    (ROOT/'configs/keyword_ablation.yaml').write_text(yaml.safe_dump({'basis':'Direct issue-name phrases fixed before fitting; no future feature inspection','phrases':phrases},sort_keys=False),encoding='utf8')
    print('INCLUDED',included,'N',len(df),flush=True)

if __name__=='__main__':main()
