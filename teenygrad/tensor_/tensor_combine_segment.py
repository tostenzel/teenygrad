from __future__ import annotations
import  math
from typing import List
from functools import reduce
from itertools import accumulate

from teenygrad.helpers import all_int


def cat(tensor, *args, dim) -> 'Tensor':
    from teenygrad.tensor_ import Tensor
    dim = (dim + len(tensor.shape)) if dim < 0 else dim
    assert all(len(y.shape) == len(tensor.shape) and all(y.shape[i] == s for i,s in enumerate(tensor.shape) if i != dim) for y in args)
    catargs = [tensor, *args]
    assert all(t.shape for t in catargs), "zero-dimensional tensor cannot be concatenated"
    shapes = [s.shape[dim] for s in catargs]
    shape_cumsum = [0, *accumulate(shapes)]
    slc = [[(0, 0) for _ in tensor.shape] for _ in catargs]
    for shp,k,s in zip(shapes, shape_cumsum[:-1], slc): s[dim] = (k, shape_cumsum[-1] - k - shp)
    return reduce(Tensor.__add__, [arg.pad(tuple(s)) for arg,s in zip(catargs, slc)])


@staticmethod
def stack(tensors, dim) -> 'Tensor':
    first = tensors[0].unsqueeze(dim)
    unsqueezed_tensors = [tensor.unsqueeze(dim) for tensor in tensors[1:]]
    # checks for shapes and number of Falsedimensions delegated to cat
    return first.cat(*unsqueezed_tensors, dim=dim)


def repeat(tensor: 'Tensor', repeats) -> 'Tensor':
    base_shape = (1,) * (len(repeats) - tensor.ndim) + tensor.shape
    new_shape = [x for b in base_shape for x in [1, b]]
    expand_shape = [x for rs in zip(repeats, base_shape) for x in rs]
    final_shape = [r*s for r,s in zip(repeats, base_shape)]
    return tensor.reshape(new_shape).expand(expand_shape).reshape(final_shape)


def chunk(tensor: 'Tensor', num:int, dim:int) -> List['Tensor']:
    assert all_int(tensor.shape), f"does not support symbolic shape {tensor.shape}"
    dim, step = dim + tensor.ndim if dim < 0 else dim, math.ceil(tensor.shape[dim]/num)
    slice_params = [[slice(None)]*dim + [slice(k, k + step)] for k in range(0, tensor.shape[dim], step)]
    return [tensor[tuple(sl)] for sl in slice_params]


