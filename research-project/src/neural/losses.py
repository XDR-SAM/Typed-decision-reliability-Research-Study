"""Hard labels throughout. The RLCD ablation follows the pinned upstream loop."""
import torch
import torch.nn.functional as F

def supervised_loss(logits,labels):
    return F.cross_entropy(logits.float(),labels)

def rlcd_loss(logits,labels,epoch,epochs,cfg):
    from laya.common import proper_reward
    z0=logits.float();mask=torch.ones_like(z0,dtype=torch.bool)
    target=F.one_hot(labels,num_classes=z0.shape[-1]).float()
    sigma=cfg['sigma_start']+(cfg['sigma_end']-cfg['sigma_start'])*epoch/max(1,epochs-1)
    eps=torch.randn((cfg['group_samples'],)+z0.shape,device=z0.device)*sigma
    eps=eps-eps.mean(-1,keepdim=True)
    projected=z0.detach().unsqueeze(0)+eps
    p=torch.softmax(projected,-1)
    with torch.no_grad():
        reward=proper_reward(p,target.unsqueeze(0),torch.zeros(len(labels),device=z0.device,dtype=torch.long),
            mask,w_sph=cfg['spherical_weight'],w_rps=0.0,log_floor=cfg['log_floor'])
        advantage=reward-reward.mean(0,keepdim=True)
        advantage=advantage/(advantage.std()+1e-6)
    log_density=-((projected-z0.unsqueeze(0))**2).sum(-1)/(2*sigma**2)
    return cfg['rl_weight']*(-(advantage*log_density).mean())+cfg['ce_weight']*supervised_loss(z0,labels)
