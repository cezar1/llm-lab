
import torch
def sliding(q,k,v,window):
    T=q.size(1)
    out=[]
    for t in range(T):
        ks=k[:,max(0,t-window):t+1]
        vs=v[:,max(0,t-window):t+1]
        att=torch.softmax(q[:,t:t+1]@ks.transpose(-2,-1)/(q.size(-1)**0.5),dim=-1)
        out.append(att@vs)
    return torch.cat(out,dim=1)
