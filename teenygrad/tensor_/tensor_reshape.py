from __future__ import annotations
from typing import Tuple, Optional, Union, List

from teenygrad.helpers import argfix, prod, shape_int

import teenygrad.function as function


# movement mlops

def reshape(tensor: 'Tensor', shape, *args) -> 'Tensor':
    new_shape = argfix(shape, *args)
    return function.Reshape.apply(tensor, shape=tuple([-prod(tensor.shape) // prod(new_shape) if s == -1 else (s if s is not None else tensor.shape[i]) for i,s in enumerate(new_shape)]))


def expand(tensor: 'Tensor', shape, *args) -> 'Tensor':
    return function.Expand.apply(tensor, shape=tuple([x if x != -1 else s for s,x in zip(tensor.shape, argfix(shape, *args))]))


def permute(tensor: 'Tensor', order, *args) -> 'Tensor': return function.Permute.apply(tensor, order=argfix(order, *args))


def flip(tensor: 'Tensor', axis, *args) -> 'Tensor':
    return function.Flip.apply(tensor, axis=[x if x >= 0 else x+len(tensor.shape) for x in argfix(axis, *args)])

def shrink(tensor: 'Tensor', arg:Tuple[Optional[Tuple[shape_int, shape_int]], ...]) -> 'Tensor':
    return function.Shrink.apply(tensor, arg=tuple(x if x is not None else (0,s) for x,s in zip(arg, tensor.shape))) if any(x is not None and x != (0,s) for x,s in zip(arg, tensor.shape)) else tensor

def pad(tensor: 'Tensor', arg:Tuple[Optional[Tuple[int, int]], ...], value:float) -> 'Tensor':
    if all(x is None or x == (0,0) for x in arg): return tensor
    ret = function.Pad.apply(tensor, arg=(narg:=tuple(x if x is not None else (0,0) for x in arg)))
    return ret if 0 == value else ret + function.Pad.apply('Tensor'.ones_like(tensor), arg=narg).where(0, value)


# (padding_left, padding_right, padding_top, padding_bottom)
def pad2d(tensor: 'Tensor', padding:Union[List[int], Tuple[int, ...]], value:float) -> 'Tensor':
    slc = [(-p0, s+p1) for p0,p1,s in zip(padding[::2], padding[1::2], tensor.shape[::-1])][::-1]
    return tensor.slice([(0,s) for s in tensor.shape[:-(len(padding)//2)]] + slc, value=value)


def transpose(tensor: 'Tensor', ax1, ax2) -> 'Tensor':
    order = list(range(len(tensor.shape)))
    order[ax1], order[ax2] = order[ax2], order[ax1]
    return tensor.permute(order)


def flatten(tensor: 'Tensor', start_dim): return tensor.reshape(shape=tensor.shape[:start_dim] + (-1,))


def squeeze(tensor: 'Tensor', dim) -> 'Tensor':
    if dim is None: return tensor if 1 not in tensor.shape else tensor.reshape(*[size for size in tensor.shape if size != 1])
    if dim <= 0 and tensor.ndim == 0: return tensor # This is to match PyTorch behavior
    if not -tensor.ndim <= dim < tensor.ndim: raise IndexError(f"Dimension out of range (expected to be in range of [{-tensor.ndim if tensor.ndim > 0 else tensor.ndim-1}, {tensor.ndim-1 if tensor.ndim > 0 else tensor.ndim}], but got {dim})")
    if dim < 0: dim += tensor.ndim
    return tensor if tensor.shape[dim] != 1 else tensor.reshape(*[size for idx, size in enumerate(tensor.shape) if idx != dim])


def unsqueeze(tensor: 'Tensor', dim) -> 'Tensor':
    if dim < 0: dim = len(tensor.shape) + dim + 1
    return tensor.reshape(tensor.shape[:dim] + (1,) + tensor.shape[dim:])
