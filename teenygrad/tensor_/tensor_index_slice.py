from typing import Sequence, Optional, Tuple
from collections import defaultdict

from teenygrad.helpers import shape_int, dtypes
from teenygrad.tensor_.tensor_reshape import pad, flatten


# ***** movement high level ops *****

# - Negative indices are taken relative to the end of the sequence, so X[-2] returns the 2nd-to-last element
# - A slice i:j returns the elements with indices in [i, j)
#        - If omitted, i and j will default to 0 and N, respectively, where N is the length of the sequence
#        - Negative values for i and j are taken relative to the end of the sequence
#        - Both i and j will be clamped to the range (-N, N], where N in the length of the sequence
# - Indexing with None on a given axis will add a new dimension of size one before that axis
# - Empty slices are not allowed (tensors with 0s in shape have to be supported first, for all backends).
# - For a slice [i:j:k] finding the correct indices is delegated to slice.indices(len).
# - Strides > 1 and < 0 are now allowed!:
#        - This works by applying Shrink -> [[Flip -> ] Pad -> Reshape -> Shrink] -> Reshape (ops in brackets are optional)
#        - Idea of stride < 0 support:
#                - Do the slice first, flip the axes were slice.step is negative, do slice.step -> -slice.step. Go to steps below.
#        - Idea of stride `s` > 1 support (Pad -> Reshape -> Shrink):
#                - Instead of doing [::s] on axis [dim_sz], do [:, 0] on axes [dim_sz_padded // s, s].
#                - So pad dim_sz with as many zeros as needed (dim_sz -> dim_sz_padded) so that reshape to [dim_sz_padded // s, s]
#                    is possible.
#                - Apply Shrink to do the slice [:, 0] on axes of shapes [dim_sz_padded // s, s].
# - Fancy indexing and combined indexing is supported
#        - Combined indexing works by letting regular slicing finish first -> computing the resulting dims w.r.t to Tensors passed in -> fancy indexing
#        - Any Tensors passed in __getitem__ will perform (CMPEQ with arange -> MUL with tensor -> SUM_REDUCE) iteratively
#                - The first iteration will expand the dim of tensor while consecutive iterations will reduce the dim
#        - There's a special case where a permute is needed at the end:
#                - if first Tensor passed in (expand dims) is not at dim 0
#                - and following Tensors does not follow consecutively to the end of fancy indexing's dims
def __getitem__(tensor: 'Tensor', val) -> 'Tensor': # val: Union[int, slice, Tensor, None, Ellipsis, Tuple[Union[int, slice, Tensor, None, Ellipsis], ...]]
    from teenygrad.tensor_ import Tensor
    def normalize_int(e, i, dim_sz):
        if -dim_sz <= e < dim_sz: return e if e != -1 else dim_sz-1
        raise IndexError(f"index {e} is out of bounds for dimension {i} with size {tensor.shape[i]}")

    orig_slices = list(val) if isinstance(val, tuple) else [val]
    count = defaultdict(list)
    for i,v in enumerate(orig_slices): count[type(v)].append(i)

    if (num_slices := len(count[int]) + len(count[slice]) + len(count[Tensor])) > len(tensor.shape): raise IndexError(f"too many indices for tensor of dimension {len(tensor.shape)}")
    if len(ellipsis_found := count[type(Ellipsis)]) > 1: raise IndexError("an index can only have a single ellipsis ('...')")

    ellipsis_idx = ellipsis_found[0] if ellipsis_found else len(orig_slices)
    orig_slices[ellipsis_idx:ellipsis_idx+1] = [slice(None)] * (len(tensor.shape) - num_slices)

    valid_slices = [v for v in orig_slices if v is not None]
    valid_slices = [v if isinstance(v, slice) else slice(y_ := normalize_int(v, i, dim_sz), y_+1) if isinstance(v, int) else slice(None) for i, (v, dim_sz) in enumerate(zip(valid_slices, tensor.shape))]

    start, stop, strides = zip(*y) if (y := [s.indices(dim_sz) for s, dim_sz in zip(valid_slices, tensor.shape)]) else ((), (), ())
    new_slice = tuple(((0, 0) if e < s else (s, e)) if st > 0 else ((0, 0) if e > s else (e+1, s+1)) for s, e, st in zip(start, stop, strides))
    sliced_tensor = tensor.shrink(new_slice).flip(axis=[i for i, s in enumerate(strides) if s < 0])
    new_shape = sliced_tensor.shape
    if any(abs(s) != 1 for s in strides):
        strides = tuple(abs(s) for s in strides)
        # Pad: add pad at the end: [dim_sz] -> [dim_sz_padded]
        padded_tensor = sliced_tensor.pad(tuple((0, s-(dim_sz % s) if dim_sz % s != 0 else 0) for s, dim_sz in zip(strides, sliced_tensor.shape)))
        # Reshape: [dim_sz_padded] -> [dim_sz_padded // s, s]
        reshaped_tensor = padded_tensor.reshape(flatten([sh // s, s] for sh, s in zip(padded_tensor.shape, strides)))
        new_shape = reshaped_tensor.shape[::2]
        # Shrink: do [:, 0]
        sliced_tensor = reshaped_tensor.shrink(tuple(flatten(((0, sh), (0, 1)) for sh in new_shape)))

    final_shape, it_shape, dim, tensors, dim_collapsed = [], iter(new_shape), [], [], 0
    for i,s in enumerate(orig_slices):
        if s is None: final_shape.append(1)
        else: # s is int or slice or Tensor
            dim_shape = next(it_shape)
            if isinstance(s, int):
                dim_collapsed += 1
            else:
                assert isinstance(dim_shape, int), f"does not support symbolic shape {dim_shape}"
                final_shape.append(dim_shape)
                if isinstance(s, Tensor):
                    tensors.append(s)
                    dim.append(i-dim_collapsed)
    ret = sliced_tensor.reshape(tuple(final_shape))

    if tensors: # Fancy/tensor indexing
        # normalize idx
        # TODO: first contiguous fixes torch+cpu_only CI, but it causes llvm to fail. Second one fixes llvm
        idx = [t.sign().__neg__().relu() * ret.shape[d] + t for d,t in zip(dim, tensors)]
        max_dim = max(i.ndim for i in idx)
        # compute sum_dim, arange, and idx
        sum_dim = [d if n==0 else d+max_dim-n for n,d in enumerate(dim)]
        arange = [Tensor.arange(ret.shape[d], dtype=dtypes.int32, requires_grad=False).reshape(*[1]*sd, ret.shape[d], *[1]*(ret.ndim + max_dim - n - sd - 1)) for n,(sd,d) in enumerate(zip(sum_dim, dim))]
        first_idx = [idx[0].reshape(*[1]*dim[0], *[1]*(1 + max_dim - idx[0].ndim), *idx[0].shape, *[1]*(ret.ndim - dim[0] - 1))]
        rest_idx = [i.reshape(*[1]*dim[0], *[1]*(max_dim - i.ndim), *i.shape, *[1]*(ret.ndim - dim[0] - n)) for n,i in enumerate(idx[1:], 1)]
        idx = first_idx + rest_idx
        ret = ret.reshape(*ret.shape[:sum_dim[0]+1], *[1]*max_dim, *ret.shape[sum_dim[0]+1:])
        # iteratively fancy index
        for a,i,sd in zip(arange, idx, sum_dim): ret = (a==i).mul(ret).sum(sd)
        # special permute case
        if dim[0] != 0 and len(dim) != 1 and dim != list(range(dim[0], dim[-1]+1)):
            ret_dims = list(range(ret.ndim))
            ret = ret.permute(ret_dims[dim[0]:dim[0]+max_dim] + ret_dims[:dim[0]] + ret_dims[dim[0]+max_dim:])
    return ret

def __setitem__(tensor: 'Tensor',s,v): return tensor.__getitem__(s).assign(v)




# NOTE: using slice is discouraged and things should migrate to pad and shrink
def slice(tensor: 'Tensor', arg:Sequence[Optional[Tuple[int, shape_int]]], value:float) -> 'Tensor':
    arg_ = tuple([a if a is not None else (0,s) for s,a in zip(tensor.shape, arg)])
    padding = tuple([(max(0, -p[0]), max(0, p[1]-tensor.shape[i])) for i,p in enumerate(arg_)])
    return pad(tensor, padding, value=value).shrink(tuple([(p[0] + padding[i][0], p[1] + padding[i][0]) for i,p in enumerate(arg_)]))
    # FIXME: tensor.pad(padding, value=value)... returns None...

def gather(tensor: 'Tensor', idx: 'Tensor', dim: int) -> 'Tensor':
    from teenygrad.tensor_ import Tensor
    assert idx.ndim == tensor.ndim, "tensor.ndim must equal idx.ndim"
    assert all(s >= i for s,i in zip(tensor.shape, idx.shape)), "all dim of idx.shape must be smaller than tensor.shape"
    if dim < 0: dim += tensor.ndim
    idx = idx.transpose(ax1=dim, ax2=0).unsqueeze(-1)
    permarg = list(range(tensor.ndim))
    permarg = permarg[1:dim] + [permarg[0]] + permarg[dim+1:] + [permarg[dim]] if dim != 0 else permarg[1:] + [permarg[0]]
    return ((idx == Tensor.arange(tensor.shape[dim], dtype=dtypes.int32, requires_grad=False)) * tensor.permute(*permarg).shrink(tuple([*[(0,sh) for sh in idx.shape[1:-1]], (0,tensor.shape[dim])])).unsqueeze(0)).sum(-1).transpose(ax1=0, ax2=dim)

