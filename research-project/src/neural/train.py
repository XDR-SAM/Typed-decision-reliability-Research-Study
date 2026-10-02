"""Explicitly launched GPU training only. No future rows are loaded by training."""
import argparse,os,json,random,time,sys,subprocess
from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler
from sklearn.metrics import f1_score
from scipy.special import logsumexp
from src.utils import ROOT,dump_json
from src.neural.protocol import load_config,labels_for,read_rows,check_role,historical_selection,verify_freeze
from src.neural.models import build_adapter,encoder_digest,load_tokenizer
from src.neural.encoding import encode_row,collate
from src.neural.losses import supervised_loss,rlcd_loss
from src.neural.environment import check_environment

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['laya','laya_rlcd','matched_head','modernbert'],required=True)
    parser.add_argument('--domain',choices=['debt_collection','credit_card'],required=True)
    parser.add_argument('--seed',type=int,choices=[17,43,101],required=True)
    parser.add_argument('--micro-batch',type=int,choices=[1,2,4],default=4)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--execute-training',action='store_true')
    args=parser.parse_args()
    if not args.execute_training: raise SystemExit('Preparation only: explicit --execute-training required')
    check_environment()
    verify_freeze()
    if not torch.cuda.is_available(): raise SystemExit('This entrypoint requires the next GPU phase')
    cfg=load_config();model_cfg=load_config(args.model);settings=cfg['training']
    if args.domain=='credit_card' and args.model=='laya_rlcd': raise SystemExit('RLCD is Debt Collection ablation only')
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    if world!=2: raise SystemExit('Frozen resource design uses torchrun --nproc_per_node=2')
    dist.init_process_group('nccl',timeout=timedelta(hours=2));torch.cuda.set_device(local);device=torch.device('cuda',local)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    # Read only the historical exports, never the complete future-data file.
    folder=ROOT/'outputs/neural_preparation'/args.domain
    train=pd.read_json(folder/'train.jsonl.gz',lines=True,convert_dates=False)
    val=pd.read_json(folder/'validation.jsonl.gz',lines=True,convert_dates=False)
    check_role(train,'train');check_role(val,'validation')
    labels=labels_for(args.domain)
    from transformers import get_cosine_schedule_with_warmup
    tok=load_tokenizer()
    def prepare(frame):return [encode_row(tok,r,labels,model_cfg['model_key']) for r in frame.to_dict('records')]
    items=prepare(train);valitems=prepare(val)
    sampler=DistributedSampler(items,num_replicas=world,rank=rank,shuffle=True,seed=args.seed)
    loader=DataLoader(items,batch_size=args.micro_batch,sampler=sampler,collate_fn=lambda b:collate(b,tok.pad_token_id))
    net=build_adapter(model_cfg,len(labels))
    initial_digest=encoder_digest(net.encoder) if rank==0 else None
    net.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    if hasattr(net,'decision'):net.decision.head_checkpointing=True
    net.to(device);ddp=DDP(net,device_ids=[local],find_unused_parameters=False)
    groups=[{'params':[p for n,p in net.named_parameters() if p.requires_grad and 'encoder.' in n],
             'lr':settings['encoder_learning_rate']},
            {'params':[p for n,p in net.named_parameters() if p.requires_grad and 'encoder.' not in n],
             'lr':settings['head_learning_rate']}]
    optimizer=torch.optim.AdamW(groups,betas=tuple(settings['betas']),eps=settings['epsilon'],weight_decay=settings['weight_decay'])
    accumulation=settings['effective_batch']//(world*args.micro_batch)
    total_steps=int(np.ceil(len(loader)/accumulation))*settings['epochs']
    scheduler=get_cosine_schedule_with_warmup(optimizer,int(total_steps*settings['warmup_fraction']),total_steps)
    scaler=torch.amp.GradScaler('cuda')
    out=ROOT/'outputs/neural'/args.domain/model_cfg['model_key']/f'seed_{args.seed}'
    if rank==0:
        out.mkdir(parents=True,exist_ok=True)
        if (out/'best.pt').exists() and not args.resume:raise RuntimeError('Existing neural run; use --resume explicitly')
        dump_json(out/'initialization.json',{'encoder_sha256':initial_digest,'model':model_cfg,'labels':labels,
            'training_rows':len(train),'validation_rows':len(val),'micro_batch':args.micro_batch,
            'effective_batch':settings['effective_batch'],'historical_only':True,
            'training_truncation_fraction':float(np.mean([x['state_tokens']>x['used_state_tokens'] for x in items])),
            'validation_truncation_fraction':float(np.mean([x['state_tokens']>x['used_state_tokens'] for x in valitems])),
            'package_versions':{n:__import__('importlib.metadata').metadata.version(n) for n in ['torch','transformers','laya']}})
        (out/'environment_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    dist.barrier();history=[];first_epoch=0
    if args.resume:
        resume=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        net.load_state_dict(resume['model']);optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler']);scaler.load_state_dict(resume['scaler'])
        history=resume['history'];first_epoch=resume['epoch']+1
        rng=resume['rng_by_rank'][rank]
        random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])
        torch.cuda.set_rng_state(rng['cuda'],device)
    started=time.time()
    for epoch in range(first_epoch,settings['epochs']):
        sampler.set_epoch(epoch);ddp.train();optimizer.zero_grad(set_to_none=True)
        for step,batch in enumerate(loader):
            y=batch.pop('label').to(device);batch={k:v.to(device) for k,v in batch.items()}
            # Last accumulation group has fewer microbatches; preserve average loss.
            group_start=(step//accumulation)*accumulation
            group_size=min(accumulation,len(loader)-group_start)
            with torch.autocast('cuda',dtype=torch.float16):logits=ddp(**batch)
            loss=(rlcd_loss(logits,y,epoch,settings['epochs'],model_cfg) if model_cfg['objective']=='rlcd_plus_ce'
                  else supervised_loss(logits,y))/group_size
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss; stop and diagnose before future results')
            scaler.scale(loss).backward()
            if (step+1)%accumulation==0 or step+1==len(loader):
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(net.parameters(),settings['gradient_clip'])
                scaler.step(optimizer);scaler.update();scheduler.step();optimizer.zero_grad(set_to_none=True)
        # Rank 0 validates the unwrapped model. Other ranks wait, no padded sampler evaluation.
        if rank==0:
            net.eval();z=[];ys=[]
            with torch.inference_mode():
                for batch in DataLoader(valitems,batch_size=args.micro_batch,collate_fn=lambda b:collate(b,tok.pad_token_id)):
                    ys.extend(batch.pop('label').tolist());batch={k:v.to(device) for k,v in batch.items()}
                    with torch.autocast('cuda',dtype=torch.float16):pred=net(**batch)
                    z.extend(pred.float().cpu().numpy())
            z=np.asarray(z);p=torch.softmax(torch.from_numpy(z),-1).numpy()
            row={'role':'validation','epoch':epoch,'macro_f1':float(f1_score(ys,p.argmax(1),labels=np.arange(len(labels)),average='macro',zero_division=0)),
                 'nll':float(np.mean(logsumexp(z.astype(np.float64),axis=1)-z[np.arange(len(ys)),ys]))}
            history.append(row);dump_json(out/'selection_history.json',history)
            if historical_selection(history)['epoch']==epoch:
                torch.save({'model':net.state_dict(),'epoch':epoch,'labels':labels,'model_config':model_cfg},out/'best.pt')
            print(args.domain,model_cfg['model_key'],args.seed,row,flush=True)
        dist.barrier()
        rng={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(device)}
        states=[None]*world;dist.all_gather_object(states,rng)
        if rank==0:
            torch.save({'model':net.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'scaler':scaler.state_dict(),'epoch':epoch,'history':history,'rng_by_rank':states},out/'latest.pt')
        dist.barrier()
    if rank==0: dump_json(out/'completion.json',{'complete':True,'seconds':time.time()-started,'selection':historical_selection(history)})
    dist.destroy_process_group()

if __name__=='__main__':main()
