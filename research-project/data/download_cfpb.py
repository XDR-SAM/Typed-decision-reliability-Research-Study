"""Download official CFPB archive parts 5-21 and record provenance/checksums."""
import argparse, hashlib, json, re, subprocess, zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
ARCHIVE = 'https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/'
TAXONOMY = 'https://files.consumerfinance.gov/f/documents/cfpb_consumer_complaint_form_product_issue_options_August_2023_FINAL.pdf'

def download(url, path):
    subprocess.run(['curl.exe','-L','--fail','--retry','3','--connect-timeout','30','--max-time','900','-o',str(path),url],check=True)

def main():
    RAW.mkdir(parents=True,exist_ok=True)
    manifest_path=ROOT/'data/source_manifest.json'
    previous=json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    old={x['filename']:x for x in previous}
    landing=RAW/'cfpb_archive.html'
    if not landing.exists(): download(ARCHIVE,landing)
    links=sorted(set(re.findall(r'https://files\.consumerfinance\.gov/[^"<>\s]+\.zip',landing.read_text(encoding='utf8'))))
    links=[u for u in links if (m:=re.search(r'Export_(\d+)_',u)) and 5<=int(m.group(1))<=21]
    assert len(links)==17, f'Expected 17 archive files, found {len(links)}'
    urls=[(ARCHIVE,landing),(TAXONOMY,RAW/'taxonomy_august_2023.pdf')]+[(u,RAW/u.rsplit('/',1)[1]) for u in links]
    records=[]
    for url,path in urls:
        if not path.exists(): download(url,path)
        if path.suffix=='.zip':
            with zipfile.ZipFile(path) as z:
                assert any(n.endswith('.csv') for n in z.namelist())
        h=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        rec={'url':url,'filename':path.name,'bytes':path.stat().st_size,'sha256':h.hexdigest(),
             'downloaded_at_utc':old.get(path.name,{}).get('downloaded_at_utc',datetime.now(timezone.utc).isoformat()),
             'source':'official CFPB','local_path':str(path.relative_to(ROOT))}
        records.append(rec)
        manifest_path.write_text(json.dumps(records,indent=2),encoding='utf8')
        print(f'RECORDED {path.name} {rec["bytes"]:,} bytes',flush=True)

if __name__=='__main__': main()
