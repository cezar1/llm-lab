
import torch.nn as nn, torch
class LayerNorm(nn.Module):
    def __init__(self,d,eps=1e-5):
        super().__init__()
        self.g=nn.Parameter(torch.ones(d))
        self.b=nn.Parameter(torch.zeros(d))
        self.eps=eps
    def forward(self,x):
        m=x.mean(-1,keepdim=True); v=x.var(-1,keepdim=True)
        return self.g*(x-m)/torch.sqrt(v+self.eps)+self.b
