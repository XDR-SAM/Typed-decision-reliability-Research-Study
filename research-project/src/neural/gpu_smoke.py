"""Explicit two-T4 integration checks; training rows only, updates discarded."""
import argparse, gc, hashlib, json, os, platform, random, subprocess, sys, time
from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from src.utils import ROOT, dump_json
from src.neural.environment import check_environment
from src.neural.protocol import verify_freeze, check_role, labels_for, load_config
from src.neural.models import build_adapter, load_tokenizer, encoder_digest
from src.neural.encoding import encode_row, collate
from src.neural.losses import supervised_loss, rlcd_loss

def tokenizer_evidence():
    folder = ROOT/'research/sources/neural'
    a = json.loads((folder/'laya_tokenizer.json').read_text())
    b = json.loads((folder/'modernbert_tokenizer.json').read_text())
    def full_vocab(t):
        out = dict(t['model']['vocab'])
        for x in t.get('added_tokens', []):
            if x['content'] in out: assert out[x['content']] == x['id']
            out[x['content']] = x['id']
        return out
    av, bv = full_vocab(a), full_vocab(b)
    mismatch = [{'token': k, 'laya_id': av.get(k), 'modernbert_id': bv.get(k)}
                for k in sorted(set(av)|set(bv)) if av.get(k) != bv.get(k)]
    def merges(t):
        return [x.split(' ') if isinstance(x, str) else x for x in t['model']['merges']]
    result = {'all_token_id_semantics_identical': not mismatch, 'vocabulary_entries': len(av),
              'vocabulary_mismatches': mismatch, 'bpe_merge_rules_identical': merges(a)==merges(b),
              'added_token_definitions_identical': a.get('added_tokens')==b.get('added_tokens'),
              'normalizer_identical': a.get('normalizer')==b.get('normalizer'),
              'pre_tokenizer_identical': a.get('pre_tokenizer')==b.get('pre_tokenizer'),
              'post_processor_identical': a.get('post_processor')==b.get('post_processor'),
              'decoder_identical': a.get('decoder')==b.get('decoder'),
              'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [folder/'laya_tokenizer.json',folder/'modernbert_tokenizer.json']}}
    dump_json(ROOT/'outputs/gpu_execution/tokenizer_compatibility.json', result)
    if mismatch: raise RuntimeError('Tokenizer semantic mismatch; stop ModernBERT before training')
    return result

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--micro-batch',type=int,choices=[4,2,1],default=4)
    args=parser.parse_args(); versions=check_environment(); verify_freeze()
    assert torch.cuda.device_count()==2, 'Exactly two GPUs required'
    assert all('T4' in torch.cuda.get_device_name(i) for i in range(2)), 'T4 devices required'
    rank=int(os.environ['RANK']); local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local); device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(hours=2))
    evidence=tokenizer_evidence() if rank==0 else None
    dist.barrier()
    signal=torch.tensor(float(rank+1),device=device); dist.all_reduce(signal)
    assert signal.item()==3, 'NCCL all-reduce failed'
    out=ROOT/'outputs/gpu_execution'; out.mkdir(parents=True,exist_ok=True)
    if rank==0:
        dump_json(out/'environment.json', {'python':sys.version,'platform':platform.platform(),
            'versions':versions,'torch_cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
            'gpu_names':[torch.cuda.get_device_name(i) for i in range(2)],
            'gpu_bytes':[torch.cuda.get_device_properties(i).total_memory for i in range(2)],
            'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True),
            'pip_freeze':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),
            'nccl_all_reduce_passed':True, 'future_predictive_results_observed':False})
    # Optimization examples are only from the frozen training export.
    path=ROOT/'outputs/neural_preparation/debt_collection/train.jsonl.gz'
    frame=pd.read_json(path,lines=True,convert_dates=False);check_role(frame,'train')
    longest=frame.loc[frame.narrative.str.len().nlargest(64).index].to_dict('records')
    labels=labels_for('debt_collection'); tok=load_tokenizer(); settings=load_config()['training']
    report=[]; original_digest=None
    for key in ['laya','matched_head','modernbert','laya_rlcd']:
        random.seed(17);np.random.seed(17);torch.manual_seed(17);torch.cuda.manual_seed_all(17)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
        cfg=load_config(key); net=build_adapter(cfg,len(labels))
        digest=encoder_digest(net.encoder) if rank==0 else None
        if rank==0 and key=='laya': original_digest=digest
        if rank==0 and key in ['matched_head','laya_rlcd']: assert digest==original_digest
        assert max(tok.get_vocab().values()) < net.encoder.get_input_embeddings().num_embeddings
        net.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        if hasattr(net,'decision'):net.decision.head_checkpointing=True
        net.to(device); ddp=DDP(net,device_ids=[local],find_unused_parameters=False)
        groups=[{'params':[p for n,p in net.named_parameters() if p.requires_grad and 'encoder.' in n], 'lr':settings['encoder_learning_rate']},
                {'params':[p for n,p in net.named_parameters() if p.requires_grad and 'encoder.' not in n], 'lr':settings['head_learning_rate']}]
        optim=torch.optim.AdamW(groups,betas=tuple(settings['betas']),eps=settings['epsilon'],weight_decay=settings['weight_decay'])
        from transformers import get_cosine_schedule_with_warmup
        steps=int(np.ceil(np.ceil(len(frame)/2/args.micro_batch)/(64//(2*args.micro_batch))))*4
        scheduler=get_cosine_schedule_with_warmup(optim,int(steps*settings['warmup_fraction']),steps)
        scaler=torch.amp.GradScaler('cuda'); accumulation=64//(2*args.micro_batch)
        items=[encode_row(tok,r,labels,cfg['model_key']) for r in longest]
        torch.cuda.reset_peak_memory_stats();dist.barrier();torch.cuda.synchronize();start=time.perf_counter()
        # Initial fp16 overflow is handled by the frozen adaptive GradScaler,
        # exactly as in the real trainer. It is not a persistent model failure.
        successful=0; skipped=0
        for attempt in range(12):
            optim.zero_grad(set_to_none=True)
            for i in range(accumulation):
                offset=(i*2+rank)*args.micro_batch
                batch=collate(items[offset:offset+args.micro_batch],tok.pad_token_id)
                y=batch.pop('label').to(device);batch={k:v.to(device) for k,v in batch.items()}
                with torch.autocast('cuda',dtype=torch.float16):z=ddp(**batch)
                assert z.shape==(args.micro_batch,len(labels)) and torch.isfinite(z).all()
                loss=rlcd_loss(z,y,0,4,cfg) if key=='laya_rlcd' else supervised_loss(z,y)
                assert torch.isfinite(loss), 'Nonfinite unscaled loss'
                scaler.scale(loss/accumulation).backward()
            scaler.unscale_(optim)
            finite=all(torch.isfinite(p.grad).all().item() for p in net.parameters() if p.grad is not None)
            flag=torch.tensor(int(finite),device=device);dist.all_reduce(flag,op=dist.ReduceOp.MIN)
            if not flag.item():
                before_scale=scaler.get_scale();scaler.step(optim);scaler.update();skipped+=1
                assert scaler.get_scale()<before_scale, 'GradScaler did not back off after overflow'
                if rank==0:print('SCALER_BACKOFF',key,attempt,scaler.get_scale(),flush=True)
                continue
            gradients={'encoder':[], 'head':[]}
            for n,p in net.named_parameters():
                if p.requires_grad and p.grad is not None and p.grad.abs().sum().item()>0:
                    gradients['encoder' if 'encoder.' in n else 'head'].append(n)
            assert gradients['encoder'] and gradients['head'], 'Missing encoder/head gradients'
            torch.nn.utils.clip_grad_norm_(net.parameters(),settings['gradient_clip'])
            scaler.step(optim);scaler.update();scheduler.step();successful+=1
            if successful==2:break
        assert successful==2, 'Persistent nonfinite gradients after adaptive scaling'
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        # Verify round-trip of model, optimizer, scaler, scheduler and per-rank RNG.
        net.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):expected=net(**batch).float().cpu()
        rng={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()}
        states=[None,None];dist.all_gather_object(states,rng)
        checkpoint=out/(cfg['model_key']+'_smoke_resume.pt')
        if rank==0:
            torch.save({'model':net.state_dict(),'optimizer':optim.state_dict(),'scaler':scaler.state_dict(),
                        'scheduler':scheduler.state_dict(),'rng_by_rank':states},checkpoint)
            with checkpoint.open('rb') as f: checksum=hashlib.file_digest(f,'sha256').hexdigest()
        dist.barrier();saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        net.load_state_dict(saved['model']);optim.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler']);scheduler.load_state_dict(saved['scheduler'])
        random.setstate(saved['rng_by_rank'][rank]['python']);np.random.set_state(saved['rng_by_rank'][rank]['numpy'])
        torch.set_rng_state(saved['rng_by_rank'][rank]['torch']);torch.cuda.set_rng_state(saved['rng_by_rank'][rank]['cuda'])
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):actual=net(**batch).float().cpu()
        assert torch.equal(expected,actual), 'Checkpoint restore changed raw logits'
        peak=torch.cuda.max_memory_allocated();peaks=[None,None];dist.all_gather_object(peaks,peak)
        if rank==0:
            row={'model':cfg['model_key'],'passed':True,'initial_encoder_sha256':digest,'micro_batch':args.micro_batch,
                 'accumulation':accumulation,'effective_batch':64,'optimizer_step_seconds':seconds,
                 'examples_per_second_including_scaler_backoff':64*(successful+skipped)/seconds,'peak_allocated_bytes_by_rank':peaks,
                 'checkpoint_sha256':checksum,'checkpoint_resume_roundtrip':True,'gradient_parameters':gradients,
                 'source_rows':'train.jsonl.gz only','sequence_length':int(batch['input_ids'].shape[1]),
                 'updates_discarded':True,'successful_optimizer_steps':successful,'scaler_skipped_steps':skipped,
                 'fp16_scaler':scaler.state_dict(),'nccl_all_reduce':True}
            report.append(row);dump_json(out/'smoke_results.json',{'passed':len(report)==4,'models':report})
            print('SMOKE_RESULT',json.dumps(row),flush=True)
        del saved;dist.barrier()
        if rank==0:checkpoint.unlink()  # Only this disposable, verified smoke checkpoint.
        dist.barrier();del ddp,net,optim,scheduler,scaler;gc.collect();torch.cuda.empty_cache()
    dist.destroy_process_group()

if __name__=='__main__':main()
