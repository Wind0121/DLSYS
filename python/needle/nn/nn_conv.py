"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
import math
from .nn_basic import Parameter, Module, Sequential, BatchNorm2d, ReLU


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_channels * kernel_size * kernel_size, out_channels * kernel_size * kernel_size, 
                                                (kernel_size, kernel_size, in_channels, out_channels), 
                                                nonlinearity="relu", device=device, dtype=dtype, requires_grad=True))
        interval = 1 / math.sqrt(in_channels * kernel_size * kernel_size)
        self.bias = Parameter(init.rand(out_channels, low=-interval, high=interval, device=device, dtype=dtype, requires_grad=True)) if bias else None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        N, C_in, H, W = x.shape
        x = x.transpose((1, 2)).transpose((2, 3))

        x = ops.conv(x, self.weight, self.stride, self.kernel_size // 2) # (N, H, W, C_out)
        
        if self.bias:
            bias = self.bias.reshape((1, 1, 1, self.out_channels))
            bias = bias.broadcast_to(x.shape)
            x = x + bias

        return x.transpose((2, 3)).transpose((1, 2))
        ### END YOUR SOLUTION

class ConvBN(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.module = Sequential(
            Conv(in_channels, out_channels, kernel_size, stride=stride, device=device),
            BatchNorm2d(out_channels, device=device),
            ReLU()
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.module(x)