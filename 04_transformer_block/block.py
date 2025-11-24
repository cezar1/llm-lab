
import torch.nn as nn
class Block(nn.Module):
    def __init__(self,d,h):
        super().__init__()
        from ..03_attention_basics.multihead import MHA
        self.att=MHA(d,h)
        self.ln1=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
        self.ln2=nn.LayerNorm(d)
    def forward(self,x):
        x=x+self.att(x)[0]; x=self.ln1(x)
        return self.ln2(x+self.ff(x))
