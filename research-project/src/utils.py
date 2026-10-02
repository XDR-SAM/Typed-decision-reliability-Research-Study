import csv, hashlib, json
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
def config(): return yaml.safe_load((ROOT/'configs/data.yaml').read_text())
def dump_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,default=str,allow_nan=False),encoding='utf8')
def save_csv(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    rows=list(rows)
    if not rows and fields is None:return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def period(date):
    if date<='2024-06-30':return 'train'
    if date<='2024-09-30':return 'validation'
    if date[:7]=='2026-07':return '2026-07'
    if date[:7]=='2026-08':return '2026-08'
    return date[:4]+'Q'+str((int(date[5:7])-1)//3+1)
def quarter(date):return date[:4]+'Q'+str((int(date[5:7])-1)//3+1)
def sha(text):return hashlib.sha256(text.encode('utf8')).hexdigest()

