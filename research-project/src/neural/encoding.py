"""Identical state-token slice and identical typed/matched encoder inputs."""
from src.neural.protocol import narrative_input,question,load_config

def encode_row(tok,row,labels,model_key):
    from laya.common import build_head
    cfg=load_config();q=question(labels)
    head,markers,stats=build_head(tok,q,cfg['head_max_len'])
    if len(markers)!=len(labels) or stats['options_distinct']!=len(labels) or stats['tokens_per_option'] is not None:
        raise ValueError('Option truncation/collision; amend protocol before training')
    state=tok(narrative_input(row),add_special_tokens=False)['input_ids']
    room=cfg['max_total_tokens']-len(head)-1
    kept=state[:room]
    if not kept: raise ValueError('Narrative has no usable tokens')
    if model_key=='modernbert':
        ids=[tok.cls_token_id]+kept+[tok.sep_token_id];offset=1;positions=[0]*len(labels)
    else:
        ids=head+kept+[tok.sep_token_id];offset=len(head);positions=markers
    state_mask=[0]*len(ids)
    state_mask[offset:offset+len(kept)]=[1]*len(kept)
    return {'input_ids':ids,'attention_mask':[1]*len(ids),'state_mask':state_mask,
        'marker_pos':positions,'marker_mask':[True]*len(labels),'qtype':0,
        'label':labels.index(row['issue']), 'state_tokens':len(state),'used_state_tokens':len(kept)}

def collate(items,pad_id):
    import torch
    n=len(items);length=max(len(x['input_ids']) for x in items)
    batch={}
    for key in ['input_ids','attention_mask','state_mask']:
        fill=pad_id if key=='input_ids' else 0
        rows=[x[key]+[fill]*(length-len(x[key])) for x in items]
        batch[key]=torch.tensor(rows,dtype=torch.long)
    for key in ['marker_pos','marker_mask','qtype','label']:
        batch[key]=torch.tensor([x[key] for x in items],dtype=torch.bool if key=='marker_mask' else torch.long)
    return batch
