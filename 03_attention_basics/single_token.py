
import torch
def attention(q,k,v):
    w=torch.softmax(q@k.T/(q.shape[-1]**0.5),dim=-1)
    return w@v,w
