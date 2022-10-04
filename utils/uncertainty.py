import torch
from einops import rearrange

def kl(prob1, prob2):
    output = torch.sum(prob1 * (torch.log(prob1 + 1e-10) - torch.log(prob2 + 1e-10)), dim=1).detach()
    return output


def kl_mask(logits1, logits2, th=0.0005):
    prob1 = torch.softmax(logits1, dim=1)
    prob2 = torch.softmax(logits2, dim=1)
    kl_map = kl(prob1, prob2)
    mask = kl_map.ge(th)
    return mask
