
import torch, torch.nn as nn
class GQA(nn.Module):
    def __init__(self,d,h_groups):
        super().__init__()
        self.Wq=nn.Linear(d,d)
        self.Wk=nn.Linear(d,d)
        self.Wv=nn.Linear(d,d)
        self.groups=h_groups
    def forward(self,x):
        q=self.Wq(x); k=self.Wk(x); v=self.Wv(x)
        B,T,D=q.shape
        g=self.groups; d=D//g
        qs=q.view(B,T,g,d); ks=k.view(B,T,g,d); vs=v.view(B,T,g,d)
        att=torch.softmax(qs@ks.transpose(-2,-1)/(d**0.5),dim=-1)
        return (att@vs).reshape(B,T,D)
