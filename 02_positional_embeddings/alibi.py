
import torch
def alibi_bias(seq_len, heads):
    slopes=torch.tensor([2**(-8*i/heads) for i in range(heads)])
    bias=torch.arange(seq_len).unsqueeze(0)-torch.arange(seq_len).unsqueeze(1)
    return slopes[:,None,None]*bias
