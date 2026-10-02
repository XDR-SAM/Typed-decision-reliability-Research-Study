"""Hash final protocol/code/inputs after preparation tests and before training."""
import hashlib,json,subprocess,argparse
from datetime import datetime,timezone
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils import ROOT,dump_json

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--amend-reason')
    parser.add_argument('--exploratory-amendment',action='store_true')
    args=parser.parse_args()
    current=ROOT/'research/protocol_freeze.json'
    future=list((ROOT/'outputs/neural').glob('**/raw_2025*.csv.gz'))+list((ROOT/'outputs/neural').glob('**/raw_2026*.csv.gz'))
    if current.exists() and not args.amend_reason:
        raise SystemExit('Frozen protocol already exists: an explicit --amend-reason is required; never silently refreeze.')
    if future and not args.exploratory_amendment:
        raise SystemExit('Future neural predictions already exist: changes require an explicitly exploratory amendment.')
    previous=digest(current) if current.exists() else None
    if current.exists():
        history=ROOT/'research/protocol_freeze_history';history.mkdir(exist_ok=True)
        (history/f'{previous}.json').write_bytes(current.read_bytes())
    folders=['src/neural','configs']
    files=[p for folder in folders for p in (ROOT/folder).glob('*') if p.is_file()]
    files += [ROOT/p for p in ['requirements-neural.txt','data/audit_credit_replication.py','data/prepare_neural_rows.py',
        'research/full_protocol.md','research/preregistration_next_stage.md','research/neural_experiment_plan.md',
        'research/credit_card_replication_audit.md','research/freeze_protocol.py','research/novelty_update.md']]
    files += list((ROOT/'tests').glob('test_neural*.py'))
    files += list((ROOT/'research/sources/neural').glob('*'))
    files=[p for p in files if p.is_file()]
    tables=[p for p in (ROOT/'outputs/neural_preparation').rglob('*') if p.is_file() and p.name not in {'rows.jsonl.gz','train.jsonl.gz','validation.jsonl.gz'}]
    data=[p for p in (ROOT/'outputs/neural_preparation').rglob('*.jsonl.gz')]
    def mapping(paths):return {p.relative_to(ROOT).as_posix():digest(p) for p in sorted(set(paths))}
    freeze={'protocol_id':'cfpb-neural-v1','freeze_date_local':'2026-10-03','timezone':'Asia/Dhaka',
        'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'pilot_base_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),
        'externally_registered':False,'debt_pilot_inspected':True,'credit_predictive_results_inspected':False,
        'neural_training_executed':bool(list((ROOT/'outputs/neural').glob('**/best.pt'))),
        'future_neural_predictions_exist':bool(future),'exploratory_amendment':args.exploratory_amendment,
        'amendment_reason':args.amend_reason,'previous_freeze_sha256':previous,
        'files':mapping(files+tables),'data_files':mapping(data)}
    dump_json(ROOT/'research/protocol_freeze.json',freeze)
    print('Frozen',len(freeze['files']),'protocol/source/audit files and',len(freeze['data_files']),'data exports.')

if __name__=='__main__':main()
