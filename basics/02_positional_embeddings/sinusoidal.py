
import math, torch
def sinusoidal(pos, dim):
    return torch.tensor([math.sin(pos/(10000**(2*i/dim))) if i%2==0 else math.cos(pos/(10000**(2*i/dim))) for i in range(dim)])
