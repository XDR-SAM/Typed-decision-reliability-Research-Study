"""Raw-logit adapters. Model downloads happen only when explicitly constructed."""
import hashlib,json,copy
from pathlib import Path
import torch
from torch import nn

class FixedHead(nn.Module):
    def __init__(self,encoder,n_classes):
        super().__init__();self.encoder=encoder
        d=encoder.config.hidden_size
        self.classifier=nn.Sequential(nn.LayerNorm(d),nn.Dropout(.1),nn.Linear(d,n_classes))
        nn.init.normal_(self.classifier[-1].weight,std=.02);nn.init.zeros_(self.classifier[-1].bias)
    def forward(self,input_ids,attention_mask,state_mask,**ignored):
        h=self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        weight=state_mask.to(h.dtype).unsqueeze(-1)
        pooled=(h*weight).sum(1)/weight.sum(1).clamp_min(1)
        return self.classifier(pooled).float()

class TypedHead(nn.Module):
    def __init__(self,decision):
        super().__init__();self.decision=decision
        for parameter in self.decision.act_head.parameters(): parameter.requires_grad_(False)
    @property
    def encoder(self): return self.decision.encoder
    def forward(self,input_ids,attention_mask,marker_pos,marker_mask,qtype,**ignored):
        logits,_=self.decision(input_ids,attention_mask,marker_pos,marker_mask,qtype)
        return logits.float()  # No temperature, entropy-confidence or act head.

def encoder_digest(encoder):
    h=hashlib.sha256()
    for name,tensor in sorted(encoder.state_dict().items()):
        h.update(name.encode());h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def build_adapter(cfg,n_classes):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoModel
    from laya.common import build_model
    if cfg['model_key']=='modernbert':
        enc=AutoModel.from_pretrained(cfg['model_id'],revision=cfg['revision'],
            attn_implementation='sdpa',reference_compile=False)
        return FixedHead(enc,n_classes)
    path=Path(snapshot_download(cfg['model_id'],revision=cfg['revision'],
        allow_patterns=['model.safetensors','encoder/config.json','rl_agent_config.json']))
    agent_cfg=json.loads((path/'rl_agent_config.json').read_text())
    decision=build_model(agent_cfg,encoder_dir=str(path/'encoder'))
    decision.load_state_dict(load_file(str(path/'model.safetensors')),strict=True)
    decision.encoder.config.reference_compile=False
    if cfg['family']=='typed': return TypedHead(decision)
    # Reuse the already-loaded tensor objects. No second encoder initialization.
    return FixedHead(decision.encoder,n_classes)

def tokenizer_path():
    from huggingface_hub import snapshot_download
    return Path(snapshot_download('convaiinnovations/laya',
        revision='55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851',allow_patterns=['tokenizer/tokenizer.json']))/'tokenizer'

def load_tokenizer():
    from transformers import PreTrainedTokenizerFast
    # The snapshot was re-saved by Transformers 5. Construct from its pinned
    # tokenizer JSON and explicit known special tokens, so 4.48 does not try to
    # resolve the newer TokenizersBackend class or alter the shared Hub cache.
    tok=PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path()/'tokenizer.json'),
        cls_token='[CLS]',sep_token='[SEP]',mask_token='[MASK]',pad_token='[PAD]',unk_token='[UNK]')
    tok.backend_tokenizer.no_truncation();tok.backend_tokenizer.no_padding()
    return tok
