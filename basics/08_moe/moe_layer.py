
import torch, torch.nn as nn
class MoE(nn.Module):
    def __init__(self,d,experts=2):
        super().__init__()
        self.experts=nn.ModuleList([nn.Linear(d,d) for _ in range(experts)])
        self.router=nn.Linear(d,experts)
    def forward(self,x):
        w=torch.softmax(self.router(x),dim=-1)
        outs=sum(w[...,i:i+1]*self.experts[i](x) for i in range(len(self.experts)))
        return outs,w
