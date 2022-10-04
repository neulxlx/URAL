import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np


def negative_index_sampler(samp_num, seg_num_list):
    negative_index = []
    for i in range(samp_num.shape[0]):
        for j in range(samp_num.shape[1]):
            negative_index += np.random.randint(low=sum(seg_num_list[:j]),
                                                high=sum(seg_num_list[:j+1]),
                                                size=int(samp_num[i, j])).tolist()
    return negative_index


def reco_loss(proto_feats_list, feats_list, ind_feats_list, num_feats_list, anchor_feats_list, ind_anchor_feats_list, num_anchor_feats_list, num_neg, temp=2):

    num_classes_current = len(ind_anchor_feats_list)
    seg_len = torch.arange(num_classes_current)
    proto_feats = torch.stack(proto_feats_list)
    ind_anchor_feats_list_ = [ind_feats_list.index(i) for i in ind_anchor_feats_list]

    reco_loss = torch.tensor(0.0)
    if len(ind_anchor_feats_list) <= 1:
        return reco_loss
    else:
        for ind, i in enumerate(ind_anchor_feats_list_):
            anchor_feats = anchor_feats_list[ind].unsqueeze(2)
            with torch.no_grad():
                seg_mask = torch.cat(([seg_len[i:], seg_len[:i]]))
                proto_sim = torch.cosine_similarity(proto_feats[seg_mask[0]], proto_feats[seg_mask[1:]])
                proto_prob = torch.softmax(proto_sim / temp, dim=0)
                proto_prob = rearrange(proto_prob, 'h w -> w h')
                negative_dist = torch.distributions.categorical.Categorical(probs=proto_prob)
                samp_class = negative_dist.sample(sample_shape=[num_anchor_feats_list[ind], num_neg])
                samp_num = torch.stack([(samp_class == c).sum(0) for c in range(num_classes_current-1)], dim=1)            

                negative_num_list = num_feats_list[i+1:] + num_feats_list[:i]
                negative_index = negative_index_sampler(samp_num, negative_num_list)  
                negative_feats_all = torch.cat(feats_list[i+1:] + feats_list[:i], dim=1)
                negative_feats = negative_feats_all[:, negative_index].reshape(64, num_anchor_feats_list[ind], num_neg)            

                positive_feats = repeat(proto_feats_list[i], 'c 1 -> c n 1', n=num_anchor_feats_list[ind])
                all_feats = torch.cat([positive_feats.detach(), negative_feats.detach()], dim=2)

            logits = torch.cosine_similarity(anchor_feats, all_feats, dim=0)
            reco_loss = reco_loss + F.cross_entropy(logits/temp, torch.zeros(logits.shape[0]).long().cuda())
    reco_loss = reco_loss / num_classes_current    
    return reco_loss



def calc_mask_proto(feats, mask):
    feats = rearrange(feats.detach(), 'b c h w -> c (b h w)')
    mask = rearrange(mask, 'b h w -> (b h w) 1')
    f = torch.mm(feats, mask.to(torch.float))
    f = f/mask.sum(dim=0, keepdims=True)
    return f

def to_one_hot(label, num_classes=7):
    b, h, w = label.shape
    label = rearrange(label, 'b h w -> b 1 h w')
    label = torch.zeros(b, num_classes, h, w, dtype=torch.int64).cuda().scatter_(1, label, 1)
    return label


def get_feats(feats, mask):
    
    feats = rearrange(feats, 'b c h w -> c (b h w)')
    mask_ = rearrange(mask, 'b h w -> (b h w)')
    ind = torch.where(mask_)[0]

    feats = torch.index_select(feats, 1, ind)

    return feats

def get_anchor_feats_list(feats, label, mask):
    mask = rearrange(mask, 'b h w -> b 1 h w')
    label = to_one_hot(label) * mask
    num_feats_list = []
    feats_list = []
    ind_feats_list = []
    for i in range(1, 7):
        label_i = label[:, i]
        if label_i.sum() == 0:
            continue
        feats_i = get_feats(feats, label_i) 
        ind_feats_list.append(i)
        feats_list.append(feats_i)
        num_feats_list.append(int(label_i.sum().item()))
    return feats_list, ind_feats_list, num_feats_list



def get_proto_feats_list(feats, label, mask):
    mask = rearrange(mask, 'b h w -> b 1 h w')
    label = to_one_hot(label) * mask
    proto_feats_list = []
    for i in range(1, 7):
        label_i = label[:, i]  # select binary mask for i-th class
        if label_i.sum() == 0:  # not all classes would be available in a mini-batch
            continue
        proto_feats_list.append(calc_mask_proto(feats, label_i))
    return proto_feats_list


def get_proto_feats(feats, label, mask):
    mask = rearrange(mask, 'b h w -> b 1 h w')
    label = to_one_hot(label) * mask
    proto_feats_list = []
    for i in range(7):
        label_i = label[:, i]  # select binary mask for i-th class
        if label_i.sum() == 0:  # not all classes would be available in a mini-batch
            proto_feats_list.append(torch.zeros(feats.shape[1], 1).cuda())
            continue
        proto_feats_list.append(calc_mask_proto(feats, label_i))
        
    proto_feats = torch.cat(proto_feats_list, dim=1)
    return proto_feats


def get_feats_list(feats, label, mask):
    mask = rearrange(mask, 'b h w -> b 1 h w')
    label_ = to_one_hot(label) * mask
    num_feats_list = []
    feats_list = []
    ind_feats_list = []
    for i in range(1, 7):
        label_i = label_[:, i]
        if label_i.sum() == 0:
            continue
        ind_feats_list.append(i)
        feats_i = get_feats(feats, label_i)
        feats_list.append(feats_i)
        num_feats_list.append(int(label_i.sum().item()))
    return feats_list, ind_feats_list, num_feats_list

