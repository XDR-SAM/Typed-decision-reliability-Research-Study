import csv, io, json, zipfile, hashlib
from collections import Counter
from pathlib import Path

root = Path(__file__).parent
csv.field_size_limit(10_000_000)
results = []
for path in sorted((root / 'sources').glob('CCDB*.zip')):
    counts, narrative_products, dates, received_dates, issues = Counter(), Counter(), Counter(), Counter(), Counter()
    total = narratives = 0
    with zipfile.ZipFile(path) as z:
        member = next(n for n in z.namelist() if n.endswith('.csv'))
        with z.open(member) as raw:
            rows = csv.DictReader(io.TextIOWrapper(raw, encoding='utf-8-sig', newline=''))
            fields = rows.fieldnames
            for row in rows:
                total += 1
                narrative = row.get('Consumer complaint narrative', '').strip()
                product = row.get('Product', '')
                counts[product] += 1
                received_dates[row.get('Date received', '')] += 1
                if narrative:
                    narratives += 1
                    narrative_products[product] += 1
                    dates[row.get('Date received', '')] += 1
                    if product in {'Debt collection','Credit card','Checking or savings account','Mortgage'}:
                        issues[(product, row.get('Issue',''))] += 1
    result = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  fields=fields, rows=total, nonempty_narratives=narratives,
                  products=dict(counts), narrative_products=dict(narrative_products),
                  narrative_received_dates=dict(sorted(dates.items())),
                  received_date_min=min(received_dates),received_date_max=max(received_dates),
                  narrative_issues={p+' | '+i:n for (p,i),n in issues.items()})
    results.append(result)
    print(json.dumps({k:v for k,v in result.items() if k not in {'narrative_received_dates','narrative_issues','fields','products'}}, indent=2))
(root/'archive-audit.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
