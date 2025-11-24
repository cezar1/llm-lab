
import torch, math
def rope(x):
    d=x.shape[-1]//2
    freqs=torch.exp(-math.log(10000)*torch.arange(0,d)/d)
    angles=torch.einsum('bd,d->bd', x[...,:d], freqs)
    return torch.cat([x[...,:d]*torch.cos(angles)-x[...,d:]*torch.sin(angles),
                      x[...,:d]*torch.sin(angles)+x[...,d:]*torch.cos(angles)], dim=-1)
