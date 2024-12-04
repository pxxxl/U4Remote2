import torch
import torch.nn as nn
import numpy as np

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


def entropy_skipping(x, mean, scale, Q=None, x_mean=None, gaussian_skipping_ratio=0.0, mask=None):
    if gaussian_skipping_ratio != 0.0 and mask is not None:
        raise ValueError("gaussian_skipping_ratio is 0.0, but mask is not None")
    if Q is None:
        Q = 1
    if x_mean is None:
        x_mean = x.mean()
    x_min = x_mean - 15_000 * Q
    x_max = x_mean + 15_000 * Q
    x = torch.clamp(x, min=x_min.detach(), max=x_max.detach())
    scale = torch.clamp(scale, min=1e-9)

    if gaussian_skipping_ratio > 0.0:
        threshold = torch.quantile(scale, gaussian_skipping_ratio)
        mask_gr = scale <= threshold
        x = torch.where(mask_gr, mean, x)

    if mask is not None:
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