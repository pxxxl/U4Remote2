import torch
import torch.nn as nn
import numpy as np
from custom.recorder import record

class Low_bound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        x = torch.clamp(x, min=1e-6)
        return x

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        grad1 = g.clone()
        grad1[x < 1e-6] = 0
        pass_through_if = np.logical_or(
            x.cpu().numpy() >= 1e-6, g.cpu().numpy() < 0.0)
        t = torch.Tensor(pass_through_if+0.0).cuda()
        return grad1 * t


def entropy_skipping(x, mean, scale, Q=None, x_mean=None, gaussian_skipping_ratio=0.0, mask=None, inplace=False):
    if gaussian_skipping_ratio != 0.0 and mask is not None:
        raise ValueError("gaussian_skipping_ratio is 0.0, but mask is not None")
    if Q is None:
        Q = 1
    if x_mean is None:
        x_mean = x.mean()
    x_min = x_mean - 15_000 * Q
    x_max = x_mean + 15_000 * Q
    if inplace:
        x[:] = torch.clamp(x, min=x_min.detach(), max=x_max.detach())
    else:
        x = torch.clamp(x, min=x_min.detach(), max=x_max.detach())
    scale = torch.clamp(scale, min=1e-9)

    if gaussian_skipping_ratio > 0.0:
        threshold = torch.quantile(scale, gaussian_skipping_ratio)
        mask_gr = scale <= threshold
        x = torch.where(mask_gr, mean, x)

    if mask is not None:
        if inplace:
            x[:] = torch.where(mask, mean, x)
        else:
            x = torch.where(mask, mean, x)

    m1 = torch.distributions.normal.Normal(mean, scale)
    lower = m1.cdf(x - 0.5*Q)
    upper = m1.cdf(x + 0.5*Q)
    likelihood = torch.abs(upper - lower)
    likelihood = Low_bound.apply(likelihood)
    bits = -torch.log2(likelihood)
    return bits
    

def evaluate_entropy_skipping(x_raw, x_skip, mean, scale):
    """
    评估entropy skipping是否有效
    有如下指标：
    1. 前后x_raw, x_skip的差别，用MSE衡量
    2. 两者的bits差别，用entropy skipping衡量
    3. 预测准确度，用x_raw, mean, scale计算出来，计算方法是1 - 2 * |cdf(x_raw) - 0.5|，这个值越大越好
    """
    metric_1 = nn.MSELoss()(x_raw, x_skip)
    metric_2 = entropy_skipping(x_raw, mean, scale) - entropy_skipping(x_skip, mean, scale)
    metric_2 = metric_2.mean().item()
    scale = torch.clamp(scale, min=1e-9)
    m1 = torch.distributions.normal.Normal(mean, scale)
    cdf_raw = m1.cdf(x_raw)
    metric_3 = (1 - 2 * torch.abs(cdf_raw - 0.5)).mean().item()
    return metric_1, metric_2, metric_3


def conduct_entropy_skipping_in_place(pc, visible_mask=None, is_training=False, step=0):
    ## view frustum filtering for acceleration
    bit_per_feat_param = None
    Q_feat = 1
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)

    anchor = pc.get_anchor[visible_mask]
    feat = pc._anchor_feat[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]
    if pc.mode == 'I_frame':
        if is_training:
            if step > pc.step_flag3:
                feat_context = pc.calc_interp_feat(anchor)
                feat_context_A = pc.get_grid_mlp(feat_context)
                if pc.enable_entropy_skipping_mask:
                    entropy_mask = pc.get_mask_mlp(feat_context)
                    entropy_mask = entropy_mask > pc.entropy_skipping_mask_threshold
                else:
                    entropy_mask = None
                mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                    torch.split(feat_context_A, split_size_or_sections=[pc.feat_dim, pc.feat_dim, 6, 6, 3*pc.n_offsets, 3*pc.n_offsets, 1, 1, 1], dim=-1)

                Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
                feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat

                choose_idx = torch.rand_like(anchor[:, 0]) <= 0.05
                feat_chosen = feat[choose_idx]
                mean = mean[choose_idx]
                scale = scale[choose_idx]
                entropy_mask = entropy_mask[choose_idx]
                Q_feat = Q_feat[choose_idx]
                # bit_feat = pc.entropy_gaussian.forward(feat_chosen, mean, scale, Q_feat, pc._anchor_feat.mean())
                raw_feat_chosen = feat_chosen.clone()
                bit_feat = entropy_skipping(feat_chosen, mean, scale, Q_feat, pc._anchor_feat.mean(), pc.entropy_skipping_ratio, entropy_mask, inplace=True)

                mse, be, ac = evaluate_entropy_skipping(raw_feat_chosen, feat_chosen, mean, scale)
                record(['ES', 'MSE'], mse.item())
                record(['ES', 'Bit Edge'], be)
                record(['ES', 'Prediction Accuracy'], ac)
                bit_per_feat_param = torch.sum(bit_feat) / bit_feat.numel()

        return bit_per_feat_param