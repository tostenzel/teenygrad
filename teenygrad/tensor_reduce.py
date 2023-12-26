from __future__ import annotations
from typing import List, Tuple, Type, Optional, Union

from teenygrad.helpers import dtypes, prod, all_int
from teenygrad.function import Function
import teenygrad.function as function

# reduce ops

def _reduce(tensor: 'Tensor', fxn:Type[Function], axis:Optional[Union[int, Tuple[int, ...]]]=None, keepdim=False) -> 'Tensor':
    from teenygrad.tensor import Tensor
    axis_: List[int] = list(range(len(tensor.shape))) if axis is None else ([axis] if isinstance(axis, int) else list(axis))
    axis_ = [x if x >= 0 else x+len(tensor.shape) for x in axis_]
    shape = tuple(s for i,s in enumerate(tensor.shape) if i not in axis_)
    if 0 in tensor.shape and 0 not in shape: return Tensor.full(tuple(1 if s == 0 else s for s in tensor.shape) if keepdim else shape, {function.Sum: 0, function.Max: -float("inf")}[fxn])
    ret = fxn.apply(tensor, new_shape=tuple([1 if i in axis_ else s for i,s in enumerate(tensor.shape)]))
    return ret if keepdim else ret.reshape(shape=shape)

def sum(tensor: 'Tensor', axis=None, keepdim=False): return tensor._reduce(function.Sum, axis, keepdim)


def max(tensor: 'Tensor', axis=None, keepdim=False): return tensor._reduce(function.Max, axis, keepdim)


def min(tensor: 'Tensor', axis=None, keepdim=False): return -((-tensor).max(axis=axis, keepdim=keepdim))


def mean(tensor: 'Tensor', axis=None, keepdim=False):
    assert all_int(tensor.shape), "does not support symbolic shape"
    out = tensor.sum(axis=axis, keepdim=keepdim)
    return out.mul(prod(out.shape)/prod(tensor.shape)) if 0 not in tensor.shape else out

def std(tensor: 'Tensor', axis=None, keepdim=False, correction=1):
    assert all_int(tensor.shape), "does not support symbolic shape"
    square_sum = ((tensor - tensor.mean(axis=axis, keepdim=True)).square()).sum(axis=axis, keepdim=keepdim)
    return square_sum.div(prod(tensor.shape)/prod(square_sum.shape)-correction).sqrt()


def _softmax(tensor: 'Tensor', axis):
    m = tensor - tensor.max(axis=axis, keepdim=True)
    e = m.exp()
    return m, e, e.sum(axis=axis, keepdim=True)


def softmax(tensor: 'Tensor', axis=-1):
    _, e, ss = tensor._softmax(axis)
    return e.div(ss)


def log_softmax(tensor: 'Tensor', axis=-1):
    m, _, ss = tensor._softmax(axis)
    return m - ss.log()


def argmax(tensor: 'Tensor', axis=None, keepdim=False):
    if axis is None:
        idx = (tensor == tensor.max(axis)) * Tensor.arange(prod(tensor.shape)-1,-1,-1, dtype=dtypes.int32, requires_grad=False).reshape(tensor.shape)
        return prod(tensor.shape) - idx.max() - 1
    axis = axis + len(tensor.shape) if axis < 0 else axis
    m = tensor == tensor.max(axis=axis, keepdim=True)
    idx = m * Tensor.arange(tensor.shape[axis]-1,-1,-1, dtype=dtypes.int32, requires_grad=False).reshape(tensor.shape[axis], *[1]*(tensor.ndim-axis-1))
    return tensor.shape[axis]-idx.max(axis=axis, keepdim=keepdim)-1


def argmin(tensor: 'Tensor', axis=None, keepdim=False): return (-tensor).argmax(axis=axis, keepdim=keepdim)
