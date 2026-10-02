"""Conservative template families: normalized 5-shingles, MinHash candidates,
exact Jaccard >=.85 edges, transitive components. Not person-level identity.
"""
import re,sys,unicodedata,time
from functools import lru_cache
from pathlib import Path
import numpy as np,pandas as pd
from datasketch import MinHash,MinHashLSH
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils import ROOT,dump_json,sha

def tokens(text):
    text=unicodedata.normalize('NFKC',text).lower()
    text=re.sub(r'\b\d+(?:[.,]\d+)*\b','num',text)
    text=re.sub(r'\bx{2,}\b','redacted',text)
    return re.findall(r'\b\w+\b',text)
def shingles(words):
    return set(' '.join(words[i:i+5]).encode() for i in range(max(0,len(words)-4)))
def main():
    start=time.time();df=pd.read_pickle(ROOT/'data/processed/debt_prepared.pkl');n=len(df)
    parent=list(range(n))
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    lsh=MinHashLSH(threshold=.7,num_perm=128)
    norm_seen={};exact_seen={};short=0;edges=[];candidates=0;canonical_links=0
    @lru_cache(maxsize=4096)
    def get_shingles(i):return shingles(tokens(df.at[i,'narrative']))
    for i,text in enumerate(df.narrative):
        raw=df.at[i,'exact_hash']
        if raw in exact_seen:
            union(i,exact_seen[raw]);continue
        exact_seen[raw]=i
        words=tokens(text);h=sha(' '.join(words))
        if h in norm_seen:
            union(i,norm_seen[h]);canonical_links+=1;continue
        norm_seen[h]=i
        if len(words)<20:short+=1;continue
        ss=shingles(words);mh=MinHash(num_perm=128,seed=20261002);mh.update_batch(list(ss))
        cand=lsh.query(mh);candidates+=len(cand)
        for j in sorted(cand):
            if find(i)==find(j):continue
            other=get_shingles(j)
            if min(len(ss),len(other))/max(len(ss),len(other))<.85:continue
            jac=len(ss&other)/len(ss|other)
            if jac>=.85:
                union(i,j);edges.append({'row_a':j,'row_b':i,'jaccard':jac,'id_a':df.at[j,'complaint_id'],'id_b':df.at[i,'complaint_id']})
        lsh.insert(i,mh)
        if i%2000==0:print(f'DEDUP {i}/{n} verified_links={len(edges)} elapsed={time.time()-start:.1f}s',flush=True)
    df['family']=[find(i) for i in range(n)]
    first=df.groupby('family',sort=False).period.transform('first')
    df['family_first_period']=first;df['family_prior_period']=df.period!=first
    df.to_pickle(ROOT/'data/processed/debt_deduplicated.pkl')
    family=df.groupby('family').agg(n=('issue','size'),labels=('issue','nunique'),periods=('period','nunique'),first_date=('date','min'),last_date=('date','max'))
    family[family.n>1].to_csv(ROOT/'outputs/audits/template_families.csv')
    pd.DataFrame(edges,columns=['row_a','row_b','jaccard','id_a','id_b']).to_csv(ROOT/'outputs/audits/verified_near_duplicate_edges.csv',index=False)
    historical=set(df[df.period.isin(['train','validation'])].family)
    rows=[];cls=[]
    for p,g in df.groupby('period',sort=False):
        novel=g[~g.family.isin(historical)]
        rows.append({'period':p,'n':len(g),'families':g.family.nunique(),'family_prior_period':int(g.family_prior_period.sum()),
                     'seen_historical_fraction':float(g.family.isin(historical).mean()),'novel_records':len(novel),'novel_families':novel.family.nunique(),
                     'largest_family':int(g.family.value_counts().max())})
        for issue,s in g.groupby('issue'):
            cls.append({'period':p,'issue':issue,'n':len(s),'families':s.family.nunique(),'novel_families':s[~s.family.isin(historical)].family.nunique()})
    pd.DataFrame(rows).to_csv(ROOT/'outputs/audits/duplicates_by_period.csv',index=False)
    pd.DataFrame(cls).to_csv(ROOT/'outputs/audits/independent_counts_by_class.csv',index=False)
    dump_json(ROOT/'outputs/audits/dedup_summary.json',{'records':n,'exact_unique':len(exact_seen),'normalized_unique':len(norm_seen),
        'families':int(df.family.nunique()),'repeated_families':int((family.n>1).sum()),'cross_period_families':int((family.periods>1).sum()),
        'rows_in_cross_period_families':int(family.loc[family.periods>1,'n'].sum()),'conflicting_label_families':int((family.labels>1).sum()),
        'canonical_duplicate_links':canonical_links,'verified_near_links':len(edges),'lsh_candidates':candidates,'short_texts_not_lsh':short,
        'elapsed_seconds':time.time()-start,'limitations':'LSH probabilistic recall; lexical near duplicates only. Transitive connected components may link endpoint pairs below .85. No author identity available.'})
    print('DEDUP DONE',df.family.nunique(),time.time()-start,flush=True)
if __name__=='__main__':main()
