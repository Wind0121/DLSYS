"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (node.inputs[1] * power(node.inputs[0], node.inputs[1] - 1)), out_grad * (power(node.inputs[0], node.inputs[1]) * log(node.inputs[0]))
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * self.scalar * (node.inputs[0] ** (self.scalar - 1))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad / rhs, out_grad * -1 * lhs / (rhs ** 2) 
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes:
            new_axes = list(range(a.ndim))
            new_axes[self.axes[0]], new_axes[self.axes[1]] = new_axes[self.axes[1]], new_axes[self.axes[0]]
            return a.permute(tuple(new_axes))
        else:
            new_axes = list(range(a.ndim))
            new_axes[a.ndim - 1], new_axes[a.ndim - 2] = new_axes[a.ndim - 2], new_axes[a.ndim - 1]
            return a.permute(tuple(new_axes))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.axes:
            return transpose(out_grad, self.axes)
        else:
            return transpose(out_grad)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.compact().reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape).compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        shrink_dims = [i for i in range(len(self.shape))]
        for i, (ori, cur) in enumerate(zip(reversed(node.inputs[0].shape), reversed(self.shape))):
            if ori == cur:
                shrink_dims[len(self.shape) - i - 1] = -1
        shrink_dims = tuple(filter(lambda x: x >= 0, shrink_dims))
        if not shrink_dims:
            return out_grad
        else:
            return summation(out_grad, shrink_dims).reshape(node.inputs[0].shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if isinstance(self.axes, (list, tuple)) and len(self.axes) > 1:
            # multiple axes case
            for axis in reversed(sorted(self.axes)):
                a = array_api.sum(a, axis = axis)
            return a
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        new_shape = list(node.inputs[0].shape)
        axes = tuple(range(len(new_shape))) if self.axes is None else self.axes
        
        if isinstance(axes, (tuple, list)):
            for i in axes:
                new_shape[i] = 1
        else:
            new_shape[axes] = 1

        return broadcast_to(reshape(out_grad, new_shape), node.inputs[0].shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lgrad, rgrad = matmul(out_grad, transpose(rhs)), matmul(transpose(lhs), out_grad)
        if len(lgrad.shape) > len(lhs.shape):
            lgrad = summation(lgrad, tuple([i for i in range(len(lgrad.shape) - len(lhs.shape))]))
        if len(rgrad.shape) > len(rhs.shape):
            rgrad = summation(rgrad, tuple([i for i in range(len(rgrad.shape) - len(rhs.shape))]))
        return lgrad, rgrad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * -1
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return divide(out_grad, node.inputs[0])
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return multiply(out_grad, exp(node.inputs[0]))
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out = node.realize_cached_data()
        return out_grad * Tensor(out > 0, device=node.device, requires_grad=True)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (-(tanh(node.inputs[0]) ** 2) + 1)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        shape = args[0].shape
        for array in args:
            assert shape == array.shape
        new_shape = list(shape)
        n = len(args)
        new_shape.insert(self.axis, n)
        new_array = array_api.empty(tuple(new_shape), device=args[0].device)

        slices = [slice(0, s) for s in new_shape]
        for i, array in enumerate(args):
            slices[self.axis] = slice(i, i + 1)
            new_array[tuple(slices)] = args[i]
        return new_array
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        n = A.shape[self.axis]
        new_shape = list(A.shape)
        new_shape.pop(self.axis)
        new_array_list = []
        new_slice = [slice(0, s) for s in new_shape]
        old_slice = [slice(0, s) for s in A.shape]
        for i in range(n):
            new_array = array_api.empty(tuple(new_shape), device=A.device)
            old_slice[self.axis] = slice(i, i + 1)
            new_array[tuple(new_slice)] = A[tuple(old_slice)]
            new_array_list.append(new_array)
        return tuple(new_array_list)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        for axis in self.axes:
            new_shape[axis] = new_shape[axis] * (self.dilation + 1)
        new_shape = tuple(new_shape)

        new_array = array_api.full(new_shape, 0, dtype="float32", device=a.device)

        slices = [slice(0, s, self.dilation + 1 if i in self.axes else 1) for i, s in enumerate(new_shape)]

        new_array[tuple(slices)] = a
        
        return new_array
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        for axis in self.axes:
            new_shape[axis] = new_shape[axis] // (self.dilation + 1)
        new_shape = tuple(new_shape)

        new_array = array_api.empty(new_shape, dtype="float32", device=a.device)

        new_slice = [slice(0, s) for s in new_shape]
        old_slice = [slice(0, s, self.dilation + 1 if i in self.axes else 1) for i, s in enumerate(a.shape)]

        new_array[tuple(new_slice)] = a[tuple(old_slice)]

        return new_array
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        pad_axes = ((0,0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))
        pad_A = A.pad(pad_axes)

        N, H, W, C_in = pad_A.shape
        K, _, _, C_out = B.shape
        Ns, Hs, Ws, Cs = pad_A.strides

        strided_H, strided_W = (H - K) // self.stride + 1, (W - K) // self.stride + 1
        strided_Hs, strided_Ws = Hs * self.stride, Ws * self.stride

        strided_A = pad_A.as_strided((N, strided_H, strided_W, K, K, C_in), (Ns, strided_Hs, strided_Ws, Hs, Ws, Cs))
        inner_dim = K * K * C_in
        outer_dim = N * strided_H * strided_W
        strided_A = strided_A.compact().reshape((outer_dim, inner_dim))
        B = B.compact().reshape((inner_dim, C_out))

        Z = strided_A @ B
        Z = Z.compact().reshape((N, strided_H, strided_W, C_out))

        return Z
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # X (N, H, W, C_in)
        # W (K, K, C_in, C_out)
        # out_grad: (N, (H - K) // self.stride + 1, (W - K) // self.stride + 1, C_out)
        X, W = node.inputs
        K, _, _, _ = W.shape

        out_grad = dilate(out_grad, (1, 2), self.stride - 1)

        flip_W = transpose(flip(W, (0, 1)), (2, 3)) # (K, K, C_out, C_in)
        x_grad = conv(out_grad, flip_W, padding=K - 1 - self.padding) # (N, H, W, C_in)

        permute_out_grad = transpose(transpose(out_grad, (0, 1)), (1, 2)) # (H - K + 1, W - K + 1, N, C_out)
        permute_X = transpose(X, (0, 3)) # (C_in, H, W, N)
        w_grad = conv(permute_X, permute_out_grad, padding=self.padding) # (C_in, K, K, C_out)
        w_grad = transpose(transpose(w_grad, (0, 1)), (1, 2))

        return x_grad, w_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1): 
    return Conv(stride, padding)(a, b)


