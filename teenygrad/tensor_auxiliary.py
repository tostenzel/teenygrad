from __future__ import annotations
import  math
from typing import List, Tuple, Optional, Union
from functools import reduce
from itertools import accumulate

from teenygrad.helpers import getenv, DEBUG, DType, dtypes, prod, all_int


def assign(tensor, x) -> 'Tensor':
    from teenygrad.tensor import Tensor
    # TODO: this is a hack for writing to DISK
    if x.__class__ is not Tensor: x = Tensor(x, dtype=tensor.dtype)
    assert tensor.shape == x.shape, f"assign shape mismatch {tensor.shape} != {x.shape}"
    assert not x.requires_grad    # tensor requires_grad is okay?
    if DEBUG >= 4: print(f"assign {tensor.data} <- {x.data}")
    if tensor.dtype == x.dtype and tensor.data is not None and not getenv("DISALLOW_ASSIGN"): x.data.output_buffer = tensor.data
    tensor.data = x.data
    return tensor



# advanced tensor ops

def multinomial(tensor:'Tensor', num_samples:int = 1, replacement:bool = False) -> 'Tensor':
    from teenygrad.tensor import Tensor
    assert 1 <= tensor.ndim <= 2 and num_samples > 0, f"{tensor.ndim=} must be 1 or 2 dim, {num_samples=} must be positive"
    assert replacement or num_samples == 1, "no replacement only supports num_samples = 1"
    weight = tensor.unsqueeze(0) if tensor.ndim == 1 else tensor
    cdf = (cw := weight.cumsum(1)) / cw[:, -1].unsqueeze(1)
    unif_samples = Tensor.rand(num_samples, cdf.shape[0], 1)
    indices = (unif_samples.expand((-1, -1, cdf.shape[1])) >= cdf).sum(2).permute((1, 0))
    return (indices.squeeze(0) if tensor.ndim == 1 else indices).cast(dtypes.int32)


def gather(tensor: 'Tensor', idx: 'Tensor', dim: int) -> 'Tensor':
    from teenygrad.tensor import Tensor
    assert idx.ndim == tensor.ndim, "tensor.ndim must equal idx.ndim"
    assert all(s >= i for s,i in zip(tensor.shape, idx.shape)), "all dim of idx.shape must be smaller than tensor.shape"
    if dim < 0: dim += tensor.ndim
    idx = idx.transpose(ax1=dim, ax2=0).unsqueeze(-1)
    permarg = list(range(tensor.ndim))
    permarg = permarg[1:dim] + [permarg[0]] + permarg[dim+1:] + [permarg[dim]] if dim != 0 else permarg[1:] + [permarg[0]]
    return ((idx == Tensor.arange(tensor.shape[dim], dtype=dtypes.int32, requires_grad=False)) * tensor.permute(*permarg).shrink(tuple([*[(0,sh) for sh in idx.shape[1:-1]], (0,tensor.shape[dim])])).unsqueeze(0)).sum(-1).transpose(ax1=0, ax2=dim)


def cat(tensor, *args, dim) -> 'Tensor':
    from teenygrad.tensor import Tensor
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


def squeeze(tensor: 'Tensor', dim) -> 'Tensor':
    if dim is None: return tensor if 1 not in tensor.shape else tensor.reshape(*[size for size in tensor.shape if size != 1])
    if dim <= 0 and tensor.ndim == 0: return tensor # This is to match PyTorch behavior
    if not -tensor.ndim <= dim < tensor.ndim: raise IndexError(f"Dimension out of range (expected to be in range of [{-tensor.ndim if tensor.ndim > 0 else tensor.ndim-1}, {tensor.ndim-1 if tensor.ndim > 0 else tensor.ndim}], but got {dim})")
    if dim < 0: dim += tensor.ndim
    return tensor if tensor.shape[dim] != 1 else tensor.reshape(*[size for idx, size in enumerate(tensor.shape) if idx != dim])


def unsqueeze(tensor: 'Tensor', dim) -> 'Tensor':
    if dim < 0: dim = len(tensor.shape) + dim + 1
    return tensor.reshape(tensor.shape[:dim] + (1,) + tensor.shape[dim:])
