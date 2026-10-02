"""Role boundaries and target isolation used by preparation and GPU entrypoints."""
import json
from pathlib import Path
import numpy as np
import yaml
from functools import lru_cache
from src.utils import ROOT

@lru_cache(maxsize=16)
def load_config(name='neural_protocol'):
    return yaml.safe_load((ROOT/'configs'/f'{name}.yaml').read_text(encoding='utf8'))

def labels_for(domain):
    return [r['original_label'] for r in load_config(domain+'_issues')['labels'] if r['included']]

def role(date):
    cfg=load_config()
    for name in ['train','validation','temperature_fit','gate_selection','gate_verification']:
        a,b=cfg[name]
        if a<=str(date)<=b: return name
    q=str(date)[:4]+'Q'+str((int(str(date)[5:7])-1)//3+1)
    if q in cfg['future_quarters']: return q
    if q in cfg['exploratory_quarters']: return q+'_exploratory'
    return 'excluded'

def narrative_input(row):
    """Metadata can never affect the model-facing state."""
    return str(row['narrative']).replace('[MASK]',' ')

def question(labels):
    return {'t':'choice','ins':load_config()['question'],'crit':{x:'' for x in labels}}

def check_role(frame, expected):
    actual=frame.date.astype(str).map(role)
    if len(frame)==0 or not actual.eq(expected).all():
        raise ValueError(f'Expected nonempty {expected} data only')
    if not frame.complaint_id.is_unique: raise ValueError('Repeated Complaint IDs')

def disjoint(*frames, families=False):
    seen=set()
    key='family' if families else 'complaint_id'
    for frame in frames:
        values=set(frame[key].astype(str))
        if values & seen: raise ValueError(f'Overlapping {key} between roles')
        seen.update(values)

def historical_selection(history):
    """No future or calibration metrics may participate in epoch selection."""
    if not history or any(r['role']!='validation' for r in history):
        raise ValueError('Model selection is validation-only')
    return sorted(history,key=lambda r:(-r['macro_f1'],r['nll'],r['epoch']))[0]

def one_hot(label, labels):
    if label not in labels: raise ValueError('Unmapped official Issue')
    out=np.zeros(len(labels),dtype=np.float32);out[labels.index(label)]=1
    return out

def read_rows(domain):
    import pandas as pd
    path=ROOT/'outputs/neural_preparation'/domain/'rows.jsonl.gz'
    frame=pd.read_json(path,lines=True,convert_dates=False)
    frame['complaint_id']=frame.complaint_id.astype(str)
    frame['family']=frame.family.astype(str)
    return frame

def verify_freeze():
    import hashlib
    freeze=json.loads((ROOT/'research/protocol_freeze.json').read_text(encoding='utf8'))
    for rel,digest in {**freeze['files'],**freeze.get('data_files',{})}.items():
        h=hashlib.sha256()
        with (ROOT/rel).open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
        actual=h.hexdigest()
        if actual!=digest: raise ValueError('Frozen file changed: '+rel)
    return freeze
