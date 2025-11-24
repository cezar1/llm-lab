
import torch, torch.nn.functional as F
def sample(logits,temp=1.0,top_k=None):
    l=logits/temp
    if top_k:
        v,_=torch.topk(l,top_k); thresh=v[..., -1].unsqueeze(-1)
        l=torch.where(l<thresh,-1e10,l)
    return torch.multinomial(F.softmax(l,dim=-1),1)
