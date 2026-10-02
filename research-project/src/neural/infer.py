"""Save uncalibrated logits; requires all prespecified domain runs completed."""
import argparse,json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.utils import ROOT
from src.neural.protocol import load_config,labels_for,read_rows,verify_freeze
from src.neural.models import build_adapter,load_tokenizer
from src.neural.encoding import encode_row,collate
from src.neural.environment import check_environment

def require_completed(domain):
    keys=['laya_ce','matched_head','modernbert']+(['laya_rlcd'] if domain=='debt_collection' else [])
    for key in keys:
        for seed in load_config()['seeds']:
            path=ROOT/'outputs/neural'/domain/key/f'seed_{seed}'/'completion.json'
            if not path.exists(): raise RuntimeError(f'Complete all frozen training runs before viewing future neural results: {path}')
    # Attribution comparison is permitted only with identical encoder initialization.
    for seed in load_config()['seeds']:
        digests=[]
        for key in ['laya_ce','matched_head']:
            path=ROOT/'outputs/neural'/domain/key/f'seed_{seed}'/'initialization.json'
            digests.append(json.loads(path.read_text())['encoder_sha256'])
        if digests[0]!=digests[1]:raise RuntimeError('Laya/matched encoder initialization differs')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--domain',required=True,choices=['debt_collection','credit_card'])
    parser.add_argument('--model',required=True,choices=['laya','laya_rlcd','matched_head','modernbert'])
    parser.add_argument('--seed',required=True,type=int,choices=[17,43,101]);parser.add_argument('--execute-inference',action='store_true')
    args=parser.parse_args()
    if not args.execute_inference:raise SystemExit('Explicit next-stage inference authorization flag required')
    check_environment()
    verify_freeze();require_completed(args.domain)
    if not torch.cuda.is_available():raise SystemExit('Use next-stage GPU session')
    cfg=load_config(args.model);labels=labels_for(args.domain);net=build_adapter(cfg,len(labels))
    out=ROOT/'outputs/neural'/args.domain/cfg['model_key']/f'seed_{args.seed}'
    checkpoint=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    if checkpoint['labels']!=labels:raise ValueError('Frozen label order mismatch')
    net.load_state_dict(checkpoint['model']);net.to('cuda');net.eval()
    tok=load_tokenizer()
    df=read_rows(args.domain);cfgp=load_config()
    periods=['validation','2024Q4']+cfgp['future_quarters']+cfgp['exploratory_quarters']
    for period in periods:
        g=df[(df.period==period)&df.included].copy()
        items=[encode_row(tok,r,labels,cfg['model_key']) for r in g.to_dict('records')]
        results=[]
        with torch.inference_mode():
            for batch in DataLoader(items,batch_size=16,collate_fn=lambda b:collate(b,tok.pad_token_id)):
                batch.pop('label');batch={k:v.to('cuda') for k,v in batch.items()}
                with torch.autocast('cuda',dtype=torch.float16):z=net(**batch)
                results.extend(z.float().cpu().numpy())
        z=np.asarray(results);p=torch.softmax(torch.from_numpy(z),-1).numpy()
        meta=[x for x in ['complaint_id','date','period','role','issue','family','novel_family','family_representative',
                          'strict_novel_family','calibration_role_eligible','exact_seen_debt_domain'] if x in g]
        predictions=g[meta].copy()
        predictions['state_tokens']=[x['state_tokens'] for x in items]
        predictions['used_state_tokens']=[x['used_state_tokens'] for x in items]
        for k in range(len(labels)):
            predictions[f'z_{k}']=z[:,k];predictions[f'p_raw_{k}']=p[:,k]
        target=out/f'raw_{period}.csv.gz'
        if target.exists():raise RuntimeError('Raw results already exist; do not silently overwrite')
        predictions.to_csv(target,index=False)

if __name__=='__main__':main()
