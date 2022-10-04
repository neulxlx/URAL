import torch
import torch.nn.functional as F
from einops import rearrange
import torch.nn as nn


def calc_dice_loss(logits, label, num_classes, eps):

    label = torch.zeros(num_classes, label.shape[1]).cuda().scatter_(0, label, 1)
    label = label[1:, :]

    prob = torch.softmax(logits, dim=0)
    prob = prob[1:, :]
    
    intersection = label * prob
    loss = 1 - (2. * intersection.sum() + eps) / (prob.sum() + label.sum() + eps) 

    return loss


def dice_loss(outputs, label):

    prob = torch.softmax(outputs, dim=1)
    
    label = rearrange(label, 'b h w -> b 1 h w')
    label = torch.zeros(prob.shape).cuda().scatter_(1, label, 1)
    label = label[:, 1:, :, :]
    label = rearrange(label, 'b c h w -> b (c h w)')
    
    prob = prob[:, 1:, :, :]
    prob = rearrange(prob, 'b c h w -> b (c h w)')
    
    intersection = label * prob
    eps = 1e-3
    loss = 1 - (2. * intersection.sum() + eps) / (prob.sum() + label.sum() + eps)
    return loss


class DiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255, eps=1e-3):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.ignore_index=ignore_index
    def forward(self, logits, label):  

        label = rearrange(label, 'b h w -> 1 (b h w)')
        ind = torch.where(label!=self.ignore_index)[1]
        label = torch.index_select(label, 1, ind)

        logits = rearrange(logits, 'b c h w -> c (b h w)')
        logits = torch.index_select(logits, 1, ind)

        loss = calc_dice_loss(logits, label, self.num_classes, self.eps)

        return loss


def masked_ce_dc(outputs, label, mask):
    outputs = rearrange(outputs, 'b c h w -> c (b h w)')
    label = rearrange(label, 'b h w -> 1 (b h w)')
    mask = rearrange(mask, 'b h w -> (b h w)')
    ind = torch.where(mask)[0]
    
    outputs = torch.index_select(outputs, 1, ind)
    label = torch.index_select(label, 1, ind)
    outputs = rearrange(outputs, 'c l -> 1 c l 1')
    label = rearrange(label, '1 l -> 1 l 1')
    loss_ce = F.cross_entropy(outputs, label)
    loss_dc = dice_loss(outputs, label)
    return loss_ce + loss_dc