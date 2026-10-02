"""Logging-only wrapper for the unchanged frozen trainer; explicit launch required."""
import functools, json, os, runpy, sys, time
from pathlib import Path
import torch
from src.utils import ROOT, dump_json
from src.neural.protocol import load_config

def main():
    # This wrapper does not choose hyperparameters or load any evaluation data.
    rank=int(os.environ.get('RANK',0)); original=torch.optim.AdamW.step
    updates=0; started=None; previous=None; intervals=[]
    args=sys.argv[1:]
    def value(flag):return args[args.index(flag)+1]
    domain=value('--domain');key=load_config(value('--model'))['model_key'];seed=value('--seed')
    out=ROOT/'outputs/neural'/domain/key/f'seed_{seed}'
    train_rows=34343 if domain=='debt_collection' else 25182
    planned_steps=((train_rows+63)//64)*4
    @functools.wraps(original)
    def measured_step(optimizer,*a,**kw):
        nonlocal updates,started,previous
        result=original(optimizer,*a,**kw)
        if rank==0:
            torch.cuda.synchronize();now=time.perf_counter();updates+=1
            if started is None:started=now
            if previous is not None:intervals.append(now-previous)
            previous=now
            if updates==1 or updates%10==0:
                # Exclude the first two intervals; include actual accumulated training work.
                steady=intervals[2:][-50:]
                duration=sum(steady)/len(steady) if steady else None
                row={'optimizer_updates_this_process':updates,'elapsed_after_first_update_seconds':now-started,
                     'seconds_per_update':duration,'global_examples_per_second':64/duration if duration else None,
                     'training_only_remaining_hours_estimate':(planned_steps-updates)*duration/3600 if duration else None,
                     'estimate_excludes_validation_checkpoint_io':True,'peak_allocated_bytes_rank0':torch.cuda.max_memory_allocated(),
                     'peak_reserved_bytes_rank0':torch.cuda.max_memory_reserved(),'current_learning_rates':[g['lr'] for g in optimizer.param_groups]}
                out.mkdir(parents=True,exist_ok=True)
                dump_json(out/'runtime_progress.json',row)
                with (out/'runtime_progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print('TRAIN_PROGRESS',json.dumps(row),flush=True)
        return result
    torch.optim.AdamW.step=measured_step
    sys.argv=['src.neural.train']+args
    runpy.run_module('src.neural.train',run_name='__main__')

if __name__=='__main__':main()
