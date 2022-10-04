import time
import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from torchvision import models as models
from models.U_Net import U_Net_4
from utils.transforms import random_flip_rotate_corrupt
from utils.metrics import DiceScore
from utils.loss import DiceLoss, masked_ce_dc
from utils.utils import setup_seed, set_requires_grad
from utils.reco_loss import *
from utils.uncertainty import *
from build_dataset import build_dataset
from experiment_config import EXPERIMENTS
import higher
from einops import rearrange


NUM_CLASSES = 7

def main(config):
    setup_seed(config.seed)
    TRAIN_CONFIG = EXPERIMENTS[config.experiment]
    train_dataset, test_dataset, eval_dataset = build_dataset(config.dataset, TRAIN_CONFIG, config.noise_rate, config.beta, random_flip_rotate_corrupt)
    train_loader = data.DataLoader(train_dataset, batch_size=4, num_workers=1, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False)
    eval_loader  = data.DataLoader(eval_dataset, batch_size=1, num_workers=1, shuffle=False)

    model = U_Net_4().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.init_lr, momentum=0.9, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    dl = DiceLoss(NUM_CLASSES)

    meta_model = meta_net(64, 7).cuda()
    meta_opt = torch.optim.SGD(meta_model.parameters(), lr=config.init_lr)

    teacher_model = U_Net_4().cuda()
    set_requires_grad(teacher_model, False)

    fixed_model = U_Net_4().cuda()
    set_requires_grad(fixed_model, False)

    t_start = time.time()
    for epoch in range(config.n_epochs):
        print('epoch: {}'.format(epoch))
        model.train()
        meta_model.train()
        teacher_model.train()
        t_epoch = time.time()
        train(model, meta_model, teacher_model, fixed_model, train_loader, optimizer, meta_opt, dl, config)    
  
        scheduler.step() 
        t_train = time.time()
        print('epoch: {}'.format(epoch))('cost {:.2f} seconds in this train epoch'.format(t_train - t_epoch))
        score_eval = validation(model, eval_loader)

    t_end = time.time()
    print('epoch: {}'.format(epoch))('cost {:.2f} minutes'.format((t_end - t_start)/60))


def train(model, meta_model, teacher_model, fixed_model, train_loader, optimizer, meta_opt, dl, config):
    
    for inputs in train_loader:
        img, label = inputs[0].cuda(), inputs[1].cuda()
        meta_img, meta_label = inputs[3].cuda(), inputs[4].cuda()
        meta_img = rearrange(meta_img, 'b c h w -> (b c) 1 h w')
        meta_label = rearrange(meta_label, 'b c h w -> (b c) h w')        
        model.zero_grad()
        meta_model.zero_grad()
        update_meta_parameters(meta_model, model)

        with torch.no_grad():

            for param_teacher, param_student in zip(teacher_model.parameters(), model.parameters()):
                param_teacher.data = param_teacher.data * config.m + param_student.data * (1. - config.m)       
                
            t_logits, t_feats = teacher_model(img)
            meta_t_logits, _ = teacher_model(meta_img)

            f_logits, _ = fixed_model(img)
            fixed_soft_pseudo = torch.softmax(f_logits, dim=1)  

            logits, feats = model(img)
            meta_logits, meta_feats = model(meta_img)

            e_mask = kl_mask(t_logits, logits, config.th)
            meta_e_mask = kl_mask(meta_t_logits, meta_logits, config.th)

            mask_feats, mask_label = get_feats_label(feats, label, e_mask)
            mask_meta_feats, mask_meta_label = get_feats_label(meta_feats, meta_label, meta_e_mask)

        with higher.innerloop_ctx(meta_model, meta_opt) as (meta_model_, meta_opt_):
            logits = meta_model_(mask_feats)
            eps = torch.zeros_like(mask_label, dtype=torch.double, requires_grad=True)
            loss = weighted_cross_entropy_loss(logits, mask_label, eps)
            meta_opt_.step(loss)
            meta_logits = meta_model_(mask_meta_feats)
            meta_loss = F.cross_entropy(meta_logits, mask_meta_label)
            g = torch.autograd.grad(meta_loss, eps)[0].detach().to(torch.float)

        with torch.no_grad():
            mask = get_mask(g, e_mask)
            uncertain_mask = e_mask.clone()
            uncertain_mask[torch.where(mask)] = 0
            proto_mask = e_mask.clone()
            
            uncertain_t_feats = get_feats(t_feats, uncertain_mask)
            uncertain_soft_pseudo = get_feats(fixed_soft_pseudo, uncertain_mask)

            proto_feats = get_proto_feats(t_feats, label, proto_mask)
            proto_feats = repeat(proto_feats, 'c l -> c l n', n=uncertain_t_feats.shape[1])

            uncertain_t_feats = rearrange(uncertain_t_feats, 'c n -> c 1 n')
            sim = torch.cosine_similarity(uncertain_t_feats, proto_feats, dim=0)
            w = torch.softmax(sim, dim=0)

            refined_uncertain_soft_pseudo = uncertain_soft_pseudo.mul(w)
            refined_uncertain_pseudo = refined_uncertain_soft_pseudo.max(0)[1]
            refined_uncertain_pseudo = rearrange(refined_uncertain_pseudo, 'l -> 1 l')
                
        logits, feats = model(img)
        uncertain_logits = get_feats(logits, uncertain_mask)
        uncertain_logits = rearrange(uncertain_logits, 'c l -> 1 c l')
        loss_uncertain = F.cross_entropy(uncertain_logits, refined_uncertain_pseudo)
        loss_seg = F.cross_entropy(logits, label) + dl(logits, label)
        loss_meta = masked_ce_dc(logits, label, mask)
                     
        anchor_mask = e_mask.clone()
        valid_mask = e_mask.clone()
        proto_mask = e_mask.clone()
        
        re_label = label.clone()
        re_label[torch.where(uncertain_mask)] = refined_uncertain_pseudo
        
        loss_reco = torch.tensor(0.0).cuda()
        for i in range(4):
            anchor_feats_list, \
            ind_anchor_feats_list, \
            num_anchor_feats_list = get_anchor_feats_list(feats[i:i+1], re_label[i:i+1], anchor_mask[i:i+1])

            proto_feats_list = get_proto_feats_list(t_feats[i:i+1], re_label[i:i+1], proto_mask[i:i+1])
            if len(proto_feats_list) == 0:
                continue
            
            feats_list, ind_feats_list, num_feats_list = get_feats_list(t_feats[i:i+1], re_label[i:i+1], valid_mask[i:i+1])

            loss_reco += reco_loss(proto_feats_list, \
                                feats_list, ind_feats_list, num_feats_list, \
                                anchor_feats_list, ind_anchor_feats_list, num_anchor_feats_list, config.num, config.temp)    
            
        loss = loss_seg + config.r_meta * loss_meta + config.r_reco * loss_reco + config.r_uncertain * loss_uncertain
        loss.backward()
        optimizer.step()    


def validation(model, eval_loader):
    model.eval()
    pred_all  = []
    label_all = []
    for inputs in eval_loader:
        img, label = inputs
        img = img.cuda()
        with torch.no_grad():
            logits = model(img)[0, :, :, :]
        pred = logits.data.max(0)[1].cpu()
        pred_all.append(pred)
        label_all.append(label)       
    pred_all  = torch.stack(pred_all, dim=0)
    label_all = torch.cat(label_all, dim=0)
    score     = DiceScore(pred_all, label_all, NUM_CLASSES)

    print('eval:')
    print('Mean Dice: {}'.format(score['Mean Dice']))        
    print('Thalamus: {}'.format(score['Dice'][0]))
    print('Caudate: {}'.format(score['Dice'][1]))
    print('Putamen: {}'.format(score['Dice'][2]))
    print('Pallidum: {}'.format(score['Dice'][3]))
    print('Hippocampus: {}'.format(score['Dice'][4]))
    print('Amygdala: {}'.format(score['Dice'][5]))  

    return score


class meta_net(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(meta_net, self).__init__()
        
        self.cls = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True),
                              nn.ReLU(),
                              nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True))
    def forward(self, x):
        return self.cls(x)


def update_meta_parameters(meta_model, model):
    for name, param in meta_model.named_parameters():
        param.data = torch.clone(model.get_parameter(name))


def get_feats_label(feats, label, mask):
    
    feats = rearrange(feats, 'b c h w -> c (b h w)')
    label = rearrange(label, 'b h w -> 1 (b h w)')
    mask_ = rearrange(mask, 'b h w -> (b h w)')
    ind = torch.where(mask_)[0]

    feats = torch.index_select(feats, 1, ind)
    label = torch.index_select(label, 1, ind)

    feats = rearrange(feats, 'c l -> 1 c l 1')
    label = rearrange(label, '1 l -> 1 l 1')

    return feats, label


def get_mask(g, mask):
    g = torch.clamp(-g.cpu(), min=0)
    g = rearrange(g, 'b c h -> (b c h)')
    t = torch.zeros_like(mask, dtype=torch.float)
    t[torch.where(mask)] = g.cuda()
    return t.greater(0)


def weighted_cross_entropy_loss(outputs, label, weight):
    loss = F.cross_entropy(outputs, label, reduction='none')
    loss = loss * weight
    return loss.sum()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--seed', nargs='?', type=int, default=20)
    parser.add_argument('--batch_size', nargs='?', type=int, default=4)
    parser.add_argument('--n_epochs', nargs='?', type=int, default=40)
    parser.add_argument('--init_lr', nargs='?', type=float, default=1e-2)
    parser.add_argument('--step_size', nargs='?', type=int, default=20)
    parser.add_argument('--gamma', nargs='?', type=float, default=0.1)
    parser.add_argument('--dataset', nargs='?', type=str, default='IBSR')

    parser.add_argument('--experiment', nargs='?', type=int, default=0)
    parser.add_argument('--noise_rate', nargs='?', type=float, default=0.3)
    parser.add_argument('--beta', nargs='?', type=int, default=7)

    parser.add_argument('--r_meta', nargs='?', type=float, default=0.5)
    parser.add_argument('--r_uncertain', nargs='?', type=float, default=0.5)
    parser.add_argument('--r_reco', nargs='?', type=float, default=0.25)
    parser.add_argument('--th', nargs='?', type=float, default=0.01)
    parser.add_argument('--m', nargs='?', type=float, default=0.999)
    parser.add_argument('--num', nargs='?', type=int, default=256)
    parser.add_argument('--temp', nargs='?', type=float, default=2)


    config = parser.parse_args()
    main(config)
