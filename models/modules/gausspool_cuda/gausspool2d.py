import torch
from torch import nn
from torch.autograd import Function

from .build import gausspool2d_cuda

class gausspool2d_func(Function):
    @staticmethod
    def forward(ctx, input, weights, kernel_size, stride, padding, ceil_mode, count_include_pad):
        output = gausspool2d_cuda.forward(input, weights, kernel_size, stride, padding, ceil_mode, count_include_pad)
        ctx.save_for_backward(input, weights)
        ctx.params = [kernel_size, stride, padding, ceil_mode, count_include_pad]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        d_input, d_weights = gausspool2d_cuda.backward( grad_output.contiguous(),  *ctx.saved_variables, *ctx.params)
        return d_input, d_weights, None, None, None, None, None
