import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.utils import ROOT
from src.neural.protocol import role,narrative_input,one_hot,historical_selection,disjoint,load_config,verify_freeze
from src.neural.calibration import fit_temperature,select_gate,probabilities,adaptive_ece,nll_from_logits
from data.prepare_neural_rows import apply_roles

def test_neural_role_boundaries():
    assert role('2024-06-30')=='train'
    assert role('2024-07-01')=='validation'
    assert role('2024-10-01')=='temperature_fit'
    assert role('2024-11-01')=='gate_selection'
    assert role('2024-12-01')=='gate_verification'
    assert role('2025-01-01')=='2025Q1'
    assert role('2026-04-01')=='2026Q2_exploratory'
    assert role('2026-07-01')=='excluded'

def test_no_future_selection_or_temperature_fit():
    with pytest.raises(ValueError):historical_selection([{'role':'2025Q1','macro_f1':1,'nll':0,'epoch':1}])
    frame=pd.DataFrame({'date':['2025-01-01']*100,'complaint_id':list(map(str,range(100)))})
    with pytest.raises(ValueError):fit_temperature(np.ones((100,2)),np.arange(100)%2,frame)
    with pytest.raises(ValueError):fit_temperature(np.ones((100,2)),np.arange(100)%2,frame,fit_role='2025Q1')
    with pytest.raises(ValueError):fit_temperature(np.ones((100,2)),np.arange(100)%2,frame,target_quarter='2025Q3')

def test_metadata_is_not_input_and_target_is_authentic_one_hot():
    a={'narrative':'I dispute this debt.','issue':'one','product':'Debt collection'}
    b={**a,'issue':'two','product':'Credit card','company':'TARGET LEAK'}
    assert narrative_input(a)==narrative_input(b)
    assert one_hot('two',['one','two']).tolist()==[0.,1.]
    with pytest.raises(ValueError):one_hot('synthetic',['one','two'])

def test_calibration_and_verification_family_overlap_prevented():
    rows=[('2024-01-01','train','historical'),('2024-10-01','2024Q4','historical'),
          ('2024-10-02','2024Q4','shared'),('2024-11-02','2024Q4','shared'),
          ('2024-11-03','2024Q4','new_nov'),('2024-12-01','2024Q4','new_nov'),
          ('2024-12-02','2024Q4','new_dec')]
    df=pd.DataFrame(rows,columns=['date','period','family']);df['complaint_id']=list(map(str,range(len(df))))
    df['included']=True
    prepared=apply_roles(df)
    selected=prepared[prepared.calibration_role_eligible]
    assert selected.family.tolist()==['shared','new_nov','new_dec']
    assert not selected.duplicated('family').any()
    with pytest.raises(ValueError):disjoint(selected.iloc[:1],selected.iloc[:1],families=True)

def test_actual_prepared_role_integrity():
    for domain in ['debt_collection','credit_card']:
        df=pd.read_csv(ROOT/'outputs/neural_preparation'/domain/'split_manifest.csv.gz',dtype={'complaint_id':str,'family':str})
        selected=df[df.calibration_role_eligible]
        disjoint(*[selected[selected.role==r] for r in ['temperature_fit','gate_selection','gate_verification']],families=True)
        historical=set(df[df.role.isin(['train','validation'])].family)
        assert not set(selected.family)&historical
        assert selected.date.max()<='2024-12-31'

def test_credit_labels_chosen_using_training_counts_only():
    counts=pd.read_csv(ROOT/'outputs/audits/issue_counts_long.csv')
    historical=counts[(counts['product']=='Credit card')&(counts.period=='train')&(counts.population=='narratives')]
    expected=set(historical.loc[historical.n>=200,'issue'])
    from src.neural.protocol import labels_for
    assert set(labels_for('credit_card'))==expected

def test_temperature_keeps_argmax_and_fails_honestly_for_small_fit():
    z=np.array([[4.,0.],[4.,0.],[0.,4.],[0.,4.]])
    assert np.array_equal(probabilities(z,.2).argmax(1),probabilities(z,5).argmax(1))
    meta=pd.DataFrame({'date':['2024-10-02']*4,'complaint_id':['a','b','c','d']})
    fit=fit_temperature(z,np.array([0,1,1,0]),meta)
    assert not fit['success'] and fit['temperature']==1.

def test_no_gate_instead_of_manufacturing_target():
    meta=pd.DataFrame({'date':['2024-11-02']*1000,'complaint_id':list(map(str,range(1000))),'family':list(map(str,range(1000)))})
    gate=select_gate(np.zeros((1000,7)),np.arange(1000)%7,meta,1.)
    assert gate['threshold'] is None and gate['status']=='no_qualifying_gate'
    with pytest.raises(ValueError):select_gate(np.zeros((1000,7)),np.arange(1000)%7,meta.assign(date='2025-01-02'),1.)

def test_gate_and_adaptive_ece_preserve_ties():
    p=np.tile([.9,.1],(100,1));y=np.array([0]*90+[1]*10)
    e,bins=adaptive_ece(y,p)
    assert e<1e-12 and len(bins)==1

def test_neural_nll_does_not_clip_overconfident_errors():
    assert nll_from_logits([[100.,0.]],[1])==100.

def test_frozen_files_have_not_changed():
    assert verify_freeze()['protocol_id']=='cfpb-neural-v1'

def test_freeze_cannot_silently_overwrite_existing_plan():
    import subprocess
    result=subprocess.run([sys.executable,str(ROOT/'research/freeze_protocol.py')],capture_output=True,text=True)
    assert result.returncode!=0 and 'explicit --amend-reason' in result.stderr

def test_official_ce_math_and_typed_matched_encoding_without_weights():
    import torch
    import torch.nn.functional as F
    source=ROOT/'research/sources/neural/laya'
    if not source.exists():
        import zipfile
        target=ROOT/'research/sources/neural/laya_source.zip'
        with zipfile.ZipFile(target) as archive:
            prefix=archive.namelist()[0].split('/')[0]+'/'
            for name in archive.namelist():
                relative=name.removeprefix(prefix)
                if relative and not name.endswith('/'):
                    path=source/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(archive.read(name))
    sys.path.insert(0,str(source))
    from laya.common import proper_reward
    from transformers import PreTrainedTokenizerFast
    from src.neural.encoding import encode_row
    z=torch.tensor([[1.,2.,3.],[3.,2.,1.]],requires_grad=True);y=torch.tensor([2,0])
    target=F.one_hot(y,3).float()
    assert torch.allclose(F.cross_entropy(z,y),-(target*F.log_softmax(z,-1)).sum(-1).mean())
    reward=proper_reward(torch.softmax(z,-1),target,torch.zeros(2,dtype=torch.long),torch.ones_like(z,dtype=torch.bool))
    assert torch.isfinite(reward).all()
    tok=PreTrainedTokenizerFast(tokenizer_file=str(ROOT/'research/sources/neural/laya_tokenizer.json'),
        cls_token='[CLS]',sep_token='[SEP]',mask_token='[MASK]',pad_token='[PAD]',unk_token='[UNK]')
    from src.neural.protocol import labels_for
    labels=labels_for('debt_collection');r={'narrative':'This is not my debt. '*500,'issue':labels[0]}
    a=encode_row(tok,r,labels,'laya_ce');b=encode_row(tok,r,labels,'matched_head');c=encode_row(tok,r,labels,'modernbert')
    assert a['input_ids']==b['input_ids'] and len(a['input_ids'])==512
    assert a['used_state_tokens']==b['used_state_tokens']==c['used_state_tokens']==449
    assert [x for x,m in zip(a['input_ids'],a['state_mask']) if m]==[x for x,m in zip(c['input_ids'],c['state_mask']) if m]

def test_tiny_official_readout_is_raw_and_matched_encoder_is_identical():
    import copy,torch
    from types import SimpleNamespace
    from torch import nn
    sys.path.insert(0,str(ROOT/'research/sources/neural/laya'))
    from laya.common import DecisionModel
    from src.neural.models import FixedHead,TypedHead,encoder_digest
    class TinyEncoder(nn.Module):
        def __init__(self):
            super().__init__();self.config=SimpleNamespace(hidden_size=8);self.embedding=nn.Embedding(32,8)
        def forward(self,input_ids,attention_mask):
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))
    enc=TinyEncoder();typed=TypedHead(DecisionModel(enc,head_layers=1));matched=FixedHead(copy.deepcopy(enc),3)
    assert encoder_digest(typed.encoder)==encoder_digest(matched.encoder)
    typed.eval();matched.eval()
    batch={'input_ids':torch.tensor([[1,2,3,4,5,6,7]]),'attention_mask':torch.ones(1,7,dtype=torch.long),
        'state_mask':torch.tensor([[0,0,0,0,1,1,0]]),'marker_pos':torch.tensor([[1,2,3]]),
        'marker_mask':torch.ones(1,3,dtype=torch.bool),'qtype':torch.zeros(1,dtype=torch.long)}
    with torch.inference_mode():
        a=typed(**batch);typed.decision.temperature.fill_(20);b=typed(**batch);c=matched(**batch)
    assert a.shape==b.shape==c.shape==(1,3)
    assert torch.allclose(a,b) and torch.isfinite(c).all()
    assert all(not p.requires_grad for p in typed.decision.act_head.parameters())
