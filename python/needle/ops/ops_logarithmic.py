from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z_original = array_api.max(Z, axis=self.axes, keepdims=True)
        max_z_reduce = array_api.max(Z, axis=self.axes)
        return array_api.log(array_api.sum(array_api.exp(Z - max_z_original), axis=self.axes)) + max_z_reduce
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        max_z = z.realize_cached_data().max(axis=self.axes, keepdims=True)
        exp_z = exp(z - max_z)
        sum_exp_z = summation(exp_z, axes=self.axes)
        
        # grad1
        grad_log_z = out_grad / sum_exp_z

        # grad2
        new_shape = list(exp_z.shape)
        axes = list(range(len(exp_z.shape))) if self.axes is None else self.axes
        for i in axes:
            new_shape[i] = 1
        grad_sum_z = broadcast_to(reshape(grad_log_z, new_shape), exp_z.shape)

        # grad3
        grad_exp_z = grad_sum_z * exp_z

        return grad_exp_z

        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

