from II2.II2_transform import *
import sys
import torch.nn as nn


EPS = 1e-10

def II2_loss(x1_out, x2_out, inver_mat, T=10):

    x2_outs_inv = perform_affine_tf(x2_out, inver_mat)

    bn, k, h, w = x1_out.shape

    x1_out = x1_out.permute(1, 0, 2, 3).contiguous()
    x2_out = x2_out.permute(1, 0, 2, 3).contiguous()

    TT = 2*T + 1

    p_i_j = F.conv2d(x1_out, weight=x2_out, padding=(T, T))
    p_i_j = p_i_j.sum(dim=2, keepdim=False).sum(dim=2, keepdim=False)

    p_i_j = (p_i_j + p_i_j.t()) / 2.0

    p_i_mat = p_i_j.sum(dim=1).unsqueeze(1)
    p_j_mat  =p_i_j.sum(dim=0).unsqueeze(0)

    p_i_j[(p_i_j < EPS).data] = EPS
    p_i_mat[(p_i_mat < EPS).data] = EPS
    p_i_mat[(p_j_mat < EPS).data] = EPS

    loss = (-p_i_j * (torch.log(p_i_j) - torch.log(p_i_mat) - torch.log(p_j_mat))).sum() / (bn * TT * TT)

    return loss

def compute_joint(x_out, x_tf_out):

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)

    p_i_j = p_i_j.sum(dim=0)

    p_i_j = (p_i_j + p_i_j.t()) / 2.

    p_i_j = p_i_j / p_i_j.sum()

    return p_i_j

def IID_loss(x_out, x_tf_out, EPS=sys.float_info.epsilon):

    bs, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)

    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    p_i_j = torch.where(p_i_j < EPS, torch.tensor(
        [EPS], device=p_i_j.device), p_i_j)
    
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

    alpha = 1.0

    loss = -1*(p_i_j * (torch.log(p_i_j) - alpha * torch.log(p_j) - alpha * torch.log(p_i))).sum()

    return loss


def KL_divergence(x_out, x_tf_out, EPS=sys.float_info.epsilon):
    p = x_out
    q = x_tf_out

    p = torch.where(p < EPS, torch.tensor([EPS], device=p.device), p)
    q = torch.where(q < EPS, torch.tensor([EPS], device=q.device), q)

    kl = (p * torch.log(p) - p * torch.log(q)).sum()

    return kl

def JS_divergence(x_out, x_tf_out, EPS=sys.float_info.epsilon):

    p = x_out
    q = x_tf_out

    p = torch.where(p < EPS, torch.tensor([EPS], device=p.device), p)
    q = torch.where(q < EPS, torch.tensor([EPS], device=q.device), q)
    r = (p + q) / 2.0

    kl1 = (p * (torch.log(r) - torch.log(p))).sum()
    kl2 = (q * (torch.log(r) - torch.log(q))).sum()

    js = -0.5 * (kl1 + kl2)

    return js

def crossview_contrastive_Loss(view1, view2, EPS=sys.float_info.epsilon):
    _, k = view1.size()
    p_i_j = compute_joint(view1, view2)
    assert (p_i_j.size() == (k, k))
    lamb_1 = -10

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    p_i_j = torch.where(p_i_j < EPS, torch.tensor(
        [EPS], device=p_i_j.device), p_i_j)
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

    loss = - p_i_j * (torch.log(p_i_j)
                      - lamb_1 * torch.log(p_j)
                      - lamb_1 * torch.log(p_i))

    loss = loss.sum()

    return loss


class Loss_FMI(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ff):

        norm_fx = ff / (ff ** 2).sum(0, keepdim=True).sqrt()
        coef_mat = torch.mm(norm_fx.t(), norm_fx)
        k = coef_mat.size(0)
        lamb = 10
        EPS = sys.float_info.epsilon
        p_i = coef_mat.sum(dim=1).view(k, 1).expand(k, k)
        p_j = coef_mat.sum(dim=0).view(1, k).expand(k, k)
        p_i_j = torch.where(coef_mat < EPS, torch.tensor(
            [EPS], device=coef_mat.device), coef_mat)
        p_j = torch.where(p_j < EPS, torch.tensor(
            [EPS], device=p_j.device), p_j)
        p_i = torch.where(p_i < EPS, torch.tensor(
            [EPS], device=p_i.device), p_i)

        loss_fmi = (p_i_j * (torch.log(p_i_j)
                             - (lamb + 1) * torch.log(p_j)
                             - (lamb + 1) * torch.log(p_i))) / (k**2)

        loss_fmi = loss_fmi.sum()

        return loss_fmi


