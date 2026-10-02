"""Stream official CSV archives, audit all records, retain narrative-only task data."""
import csv,gzip,io,json,re,sys,zipfile
from collections import Counter,defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils import ROOT,period,quarter,sha,save_csv,dump_json

def main():
    csv.field_size_limit(100_000_000)
    out=ROOT/'outputs/audits';out.mkdir(parents=True,exist_ok=True)
    processed=ROOT/'data/processed';processed.mkdir(parents=True,exist_ok=True)
    monthly=defaultdict(Counter);daily=defaultdict(Counter);issues=Counter();products=Counter();file_stats=[]
    seen=set(); duplicate_ids=0;bad_dates=0;dc=[];cc=[]
    for path in sorted((ROOT/'data/raw').glob('*.zip')):
        counts=Counter();mind='9999';maxd='0000'
        with zipfile.ZipFile(path) as z:
            member=next(n for n in z.namelist() if n.endswith('.csv'))
            with z.open(member) as f:
                reader=csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig',newline=''))
                for row in reader:
                    counts['raw_records']+=1
                    d=row['Date received'];mind=min(mind,d);maxd=max(maxd,d)
                    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}',d):bad_dates+=1;continue
                    if not '2023-09-01'<=d<='2026-08-31':continue
                    cid=row['Complaint ID']
                    if cid in seen:duplicate_ids+=1;counts['duplicate_ids']+=1;continue
                    seen.add(cid)
                    t=row.get('Consumer complaint narrative','').strip();p=row['Product'];issue=row['Issue'];m=d[:7]
                    for ctr in [monthly[m],daily[d],counts]:
                        ctr['total_records']+=1;ctr['nonempty_narratives']+=bool(t)
                        ctr['debt_records']+=p=='Debt collection';ctr['debt_narratives']+=bool(t) and p=='Debt collection'
                        ctr['credit_card_records']+=p=='Credit card';ctr['credit_card_narratives']+=bool(t) and p=='Credit card'
                    products[(m,p)]+=1
                    if p in ('Debt collection','Credit card'):
                        issues[(p,period(d),issue,'all')]+=1
                        if t:
                            issues[(p,period(d),issue,'narratives')]+=1
                            rec={'complaint_id':cid,'date':d,'product':p,'issue':issue,'narrative':t,
                                 'period':period(d),'month':m,'quarter':quarter(d),'characters':len(t),
                                 'words':len(t.split()),'exact_hash':sha(t),'source_file':path.name}
                            (dc if p=='Debt collection' else cc).append(rec)
        file_stats.append({'file':path.name,'min_date':mind,'max_date':maxd,**dict(counts)})
        print(path.name,dict(counts),flush=True)
    for name,rows in [('debt_collection',dc),('credit_card',cc)]:
        with gzip.open(processed/f'{name}.jsonl.gz','wt',encoding='utf8') as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    def ct(rows,key):
        return [{key:k,**dict(c),'narrative_fraction':c['nonempty_narratives']/c['total_records'],
                 'debt_narrative_fraction':c['debt_narratives']/max(1,c['debt_records'])} for k,c in sorted(rows.items())]
    save_csv(out/'availability_monthly.csv',ct(monthly,'month'))
    save_csv(out/'availability_daily.csv',ct(daily,'date'))
    q=defaultdict(Counter)
    for m,c in monthly.items():q[quarter(m+'-01')].update(c)
    save_csv(out/'availability_quarterly.csv',ct(q,'quarter'))
    save_csv(out/'issue_counts_long.csv',[{'product':p,'period':w,'issue':i,'population':a,'n':n} for (p,w,i,a),n in sorted(issues.items())])
    save_csv(out/'product_counts_monthly.csv',[{'month':m,'product':p,'n':n} for (m,p),n in sorted(products.items())])
    dump_json(out/'ingestion_summary.json',{'unique_records':len(seen),'duplicate_ids':duplicate_ids,'invalid_dates':bad_dates,'debt_narratives':len(dc),'credit_card_narratives':len(cc),'files':file_stats})
    print('DONE',len(seen),len(dc),len(cc),flush=True)

if __name__=='__main__':main()
