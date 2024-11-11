from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z = Z.max(axis=(1,), keepdims=True)
        exp_z = array_api.exp(Z - max_z)
        sum_exp_z = array_api.sum(exp_z, axis=(1,), keepdims=True)
        log_sum_exp_z = array_api.log(sum_exp_z)
        return Z - array_api.broadcast_to(max_z, Z.shape) - array_api.broadcast_to(log_sum_exp_z, Z.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]  # 输入 Z
        max_z = Tensor(z.realize_cached_data().max(axis=(1,), keepdims=True), device=z.device)
        exp_z = exp(z - max_z)
        sum_exp_z = summation(exp_z, axes=(1,))
        sum_exp_z = reshape(sum_exp_z, (sum_exp_z.shape[0], 1))
        softmax_z = exp_z / broadcast_to(sum_exp_z, exp_z.shape)  # 计算 softmax(Z)
        
        sum_out_grad = summation(out_grad, axes=(1,))
        grad = out_grad - broadcast_to(reshape(sum_out_grad, (out_grad.shape[0], 1)), softmax_z.shape) * softmax_z
        return grad
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z_original = Z.max(axis=self.axes, keepdims=True)
        max_z_reduce = Z.max(axis=self.axes)
        return array_api.log(array_api.sum(array_api.exp(Z - array_api.broadcast_to(max_z_original, Z.shape)), axis=self.axes)) + max_z_reduce
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        max_z = Tensor(z.realize_cached_data().max(axis=self.axes, keepdims=True), device=z.device)
        exp_z = exp(z - max_z.broadcast_to(z.shape))
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

