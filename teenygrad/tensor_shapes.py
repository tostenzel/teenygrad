# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
from __future__ import annotations
from typing import Tuple, Optional, Sequence, Union, List

from teenygrad.helpers import argfix, prod, shape_int

import teenygrad.function as function


# ***** movement mlops *****


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


# NOTE: using slice is discouraged and things should migrate to pad and shrink
def slice(tensor: 'Tensor', arg:Sequence[Optional[Tuple[int, shape_int]]], value:float) -> 'Tensor':
    arg_ = tuple([a if a is not None else (0,s) for s,a in zip(tensor.shape, arg)])
    padding = tuple([(max(0, -p[0]), max(0, p[1]-tensor.shape[i])) for i,p in enumerate(arg_)])
    return pad(tensor, padding, value=value).shrink(tuple([(p[0] + padding[i][0], p[1] + padding[i][0]) for i,p in enumerate(arg_)]))
    # FIXME: tensor.pad(padding, value=value)... returns None...

def transpose(tensor: 'Tensor', ax1, ax2) -> 'Tensor':
    order = list(range(len(tensor.shape)))
    order[ax1], order[ax2] = order[ax2], order[ax1]
    return tensor.permute(order)

def flatten(tensor: 'Tensor', start_dim): return tensor.reshape(shape=tensor.shape[:start_dim] + (-1,))