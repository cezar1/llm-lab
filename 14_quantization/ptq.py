
import torch
def ptq(t):
    mn, mx=t.min(), t.max()
    scale=(mx-mn)/255
    qt=torch.clamp(((t-mn)/scale).round(),0,255)
    return qt,scale,mn
