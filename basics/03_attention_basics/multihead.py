
import torch, torch.nn as nn
class MHA(nn.Module):
    def __init__(self,d,h):
        super().__init__()
        self.h=h; self.d=d
        self.Wq=nn.Linear(d,d); self.Wk=nn.Linear(d,d); self.Wv=nn.Linear(d,d)
        self.out=nn.Linear(d,d)
    def forward(self,x):
        B,T,_=x.size()
        q=self.Wq(x).view(B,T,self.h,-1)
        k=self.Wk(x).view(B,T,self.h,-1)
        v=self.Wv(x).view(B,T,self.h,-1)
        att=(q@k.transpose(-2,-1))/((self.d//self.h)**0.5)
        w=torch.softmax(att,dim=-1)
        out=(w@v).reshape(B,T,self.d)
        return self.out(out),w
