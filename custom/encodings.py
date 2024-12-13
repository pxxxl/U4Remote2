import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import numpy as np
import torchac
import math
import multiprocessing
import struct

class STE_binary_with_ratio(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, ratio=0.5):
        ctx.save_for_backward(input)
        thres = torch.quantile(input, ratio)
        p = (input >= thres) * 1.0
        n = (input < thres) * -1.0
        return p + n

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        i2 = input.clone().detach()
        i3 = torch.clamp(i2, -1, 1)
        mask = (i3 == i2).float()
        return grad_output * mask, None