"""Count/taxonomy/template audit only; never fit a predictive model."""
import sys, json, time
from functools import lru_cache
from pathlib import Path
import pandas as pd
import yaml
from datasketch import MinHash, MinHashLSH
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.deduplicate import tokens, shingles
from src.utils import ROOT, sha, dump_json

def main():
    start = time.time()
    out = ROOT / 'outputs/neural_preparation/credit_card'
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_json(ROOT/'data/processed/credit_card.jsonl.gz', lines=True, convert_dates=False)
    df = df.sort_values(['date','complaint_id']).reset_index(drop=True)
    counts = df.loc[df.period=='train','issue'].value_counts()
    # Same training-only threshold as the Debt Collection pilot; never drop a
    # selected class because future support or future model performance is poor.
    labels = sorted(x for x,n in counts.items() if x and n >= 200)
    entries = []
    for label in sorted(df.issue.unique()):
        entries.append({'original_label':label, 'included':label in labels,
            'training_count':int(counts.get(label,0)),
            'exclusion_reason':None if label in labels else 'Fewer than 200 original training-period narratives',
            'period_counts':{p:int(n) for p,n in df[df.issue==label].groupby('period').size().items()}})
    (ROOT/'configs/credit_card_issues.yaml').write_text(yaml.safe_dump({
        'selection_basis':'Original strings; >=200 training narratives; no future-performance selection or merging',
        'labels':entries},sort_keys=False),encoding='utf8')
    df['included'] = df.issue.isin(labels)
    parent = list(range(len(df)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i,j):
        i,j=find(i),find(j)
        if i!=j: parent[max(i,j)]=min(i,j)
    exact, canonical = {}, {}
    lsh = MinHashLSH(threshold=.7,num_perm=128)
    links, candidate_count, short = [], 0, 0
    @lru_cache(maxsize=4096)
    def get_shingles(i): return shingles(tokens(df.at[i,'narrative']))
    for i,text in enumerate(df.narrative):
        raw=df.at[i,'exact_hash']
        if raw in exact:
            union(i,exact[raw]); continue
        exact[raw]=i
        words=tokens(text); h=sha(' '.join(words))
        if h in canonical:
            union(i,canonical[h]); continue
        canonical[h]=i
        if len(words)<20:
            short+=1; continue
        ss=shingles(words); mh=MinHash(num_perm=128,seed=20261002)
        mh.update_batch(list(ss)); cand=lsh.query(mh); candidate_count+=len(cand)
        for j in sorted(cand):
            if find(i)==find(j): continue
            other=get_shingles(j)
            if min(len(ss),len(other))/max(len(ss),len(other))<.85: continue
            jac=len(ss&other)/len(ss|other)
            if jac>=.85:
                union(i,j); links.append({'id_a':df.at[j,'complaint_id'],'id_b':df.at[i,'complaint_id'],'jaccard':jac})
        lsh.insert(i,mh)
        if i%20000==0: print('Credit Card template audit',i,len(df),flush=True)
    df['family']=['cc:'+str(find(i)) for i in range(len(df))]
    df['train_keep']=(df.period=='train') & df.included & ~df.duplicated('exact_hash')
    hist=set(df[df.period.isin(['train','validation'])].family)
    df['novel_family']=~df.family.isin(hist)
    df['family_representative']=~df.duplicated(['period','family'])
    # No cross-task warm start is allowed. This audit checks that independent
    # specification is not confused with independent complaints/populations.
    dc=pd.read_csv(ROOT/'outputs/audits/debt_metadata.csv.gz',usecols=['exact_hash'])
    overlap=set(dc.exact_hash) & set(df.exact_hash)
    df['exact_seen_debt_domain']=df.exact_hash.isin(overlap)
    periods=[]; support=[]
    for period,g in df.groupby('period',sort=False):
        inc=g[g.included]; novel=inc[inc.novel_family]
        periods.append({'period':period,'all_narratives':len(g),'included_narratives':len(inc),
            'excluded_fraction':1-len(inc)/len(g),'families':inc.family.nunique(),
            'novel_families':novel.family.nunique(),'historical_family_fraction':float(inc.family.isin(hist).mean()),
            'largest_family':int(inc.family.value_counts().max()),'exact_cross_domain_rows':int(g.exact_seen_debt_domain.sum())})
        for issue,s in inc.groupby('issue'):
            support.append({'period':period,'issue':issue,'n':len(s),'families':s.family.nunique(),
                'novel_families':s[s.novel_family].family.nunique()})
    pd.DataFrame(periods).to_csv(out/'period_support.csv',index=False)
    pd.DataFrame(support).to_csv(out/'class_support.csv',index=False)
    pd.DataFrame(links).to_csv(out/'verified_near_duplicate_edges.csv',index=False)
    family=df.groupby('family').agg(n=('issue','size'),labels=('issue','nunique'),periods=('period','nunique'))
    family[family.n>1].to_csv(out/'template_families.csv')
    df.drop(columns='narrative').to_csv(out/'split_manifest.csv.gz',index=False)
    # Separate preparation dataset. No existing pilot file is overwritten.
    df.to_json(out/'rows.jsonl.gz',orient='records',lines=True,compression='gzip',force_ascii=False)
    summary={'records':len(df),'labels':labels,'training_exact_unique':int(df.train_keep.sum()),
        'exact_unique':len(exact),'families':int(df.family.nunique()),
        'cross_period_families':int((family.periods>1).sum()),
        'rows_in_cross_period_families':int(family.loc[family.periods>1,'n'].sum()),
        'conflicting_label_families':int((family.labels>1).sum()),
        'cross_domain_exact_texts':len(overlap),'cross_domain_rows':int(df.exact_seen_debt_domain.sum()),
        'short_texts_not_lsh':short,'lsh_candidates':candidate_count,
        'new_labels_after_training':sorted(set(df.issue)-set(counts.index)),
        'training_label_rule':'>=200 original training narratives; no merging',
        'predictive_results_inspected':False,'elapsed_seconds':time.time()-start}
    dump_json(out/'audit_summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
