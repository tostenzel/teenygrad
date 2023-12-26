# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
from __future__ import annotations
import time, math
from typing import List, Tuple, Optional, ClassVar, Union, Sequence, Any, Iterable
from collections import defaultdict
import numpy as np

from teenygrad.helpers import argfix, getenv, DEBUG, flatten, DType, dtypes, prod, all_int, round_up, shape_int
from teenygrad.data import TensorData
from teenygrad.ops import LoadOps
from teenygrad.function import Function
import teenygrad.function as function

from teenygrad.tensor_autograd import backward
from teenygrad.tensor_auxiliary import assign
from teenygrad.tensor_auxiliary import randn, randint, normal, uniform, scaled_uniform
from teenygrad.tensor_auxiliary import multinomial, gather, cat, stack, repeat, chunk, squeeze, unsqueeze
from teenygrad.tensor_shapes import reshape, expand, permute, flip, shrink, pad, pad2d, slice, transpose, flatten
from teenygrad.tensor_nn import _pool, avg_pool2d, max_pool2d, conv2d, linear, binary_crossentropy, binary_crossentropy_logits, sparse_categorical_crossentropy


class Tensor:
    __slots__ = "data", "requires_grad", "grad", "_ctx"
    __deletable__ = ('_ctx',)
    training: ClassVar[bool] = False
    class train:
        def __init__(self, val=True): self.val = val
        def __enter__(self): self.prev, Tensor.training = Tensor.training, self.val
        def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any): Tensor.training = self.prev

    no_grad: ClassVar[bool] = False
    default_type: ClassVar[DType] = dtypes.float32

    def __init__(self, data:Union[None, int, float, list, TensorData, np.ndarray, bytes], dtype:Optional[DType]=None, requires_grad:Optional[bool]=None):
        
        assert dtype is None or isinstance(dtype, DType), f"invalid dtype {dtype}"
        # tensors have gradients, buffers do not
        self.grad: Optional[Tensor] = None

        # NOTE: this can be in three states. False and None: no gradient, True: gradient
        # None (the default) will be updated to True if it's put in an optimizer
        self.requires_grad: Optional[bool] = requires_grad

        # internal variables used for autograd graph construction
        self._ctx: Optional[Function] = None
        if isinstance(data, TensorData): assert dtype is None or dtype == data.dtype, "dtype doesn't match, and casting isn't supported"
        elif isinstance(data, (int, float)):
            data = TensorData.loadop(LoadOps.CONST, tuple(), dtype or Tensor.default_type, data)
        elif data is None or data.__class__ is list:
            assert dtype is None or dtype.np is not None, f"{dtype} doesn't have a numpy dtype"
            data = TensorData(np.array([] if data is None else data, dtype=(dtype or Tensor.default_type).np))
        elif isinstance(data, bytes):
            data = TensorData(np.frombuffer(data, np.uint8))
        elif isinstance(data, np.ndarray):
            assert dtype is None or dtype.np is not None, f"{dtype} doesn't have a numpy dtype"
            if data.shape == ():
                data = TensorData.loadop(LoadOps.CONST, tuple(), dtype or dtypes.from_np(data.dtype), data.item())
            else:
                data = TensorData(data.astype(dtype.np) if dtype is not None and dtype.np is not None else data)

        if not isinstance(data, TensorData): raise RuntimeError(f"can't create Tensor from {data!r} with type {type(data)}")
        self.data = data

    # ------------------------------------------------------------------------------------------------------------------
    # basic properties

    def __repr__(self):
        return f"<Tensor {self.data!r} with grad {(self.grad.data if self.grad else None)!r}>"
    # Python has a non moving garbage collector, so this should be okay
    def __hash__(self): return id(self)
    @property
    def shape(self) -> Tuple[shape_int, ...]: return self.data.shape
    @property
    def dtype(self) -> DType: return self.data.dtype

    # ***** data handlers ****

    def assign(self, x) -> Tensor: return assign(self, x)

    #-------------------------------------------------------------------------------------------------------------------
    # basic tensor manipulations 

    def detach(self) -> Tensor: return Tensor(self.data, requires_grad=False)
    def numpy(self) -> np.ndarray:
        assert all_int(self.shape), f"no numpy if shape is symbolic, {self.shape=}"
        assert self.dtype.np is not None, f"no numpy dtype for {self.dtype}"
        return self.detach().cast(dtypes.from_np(self.dtype.np)).data.data.reshape(self.shape)
    def item(self) -> Union[float, int]: return self.numpy().item()

    # ***** creation low-level op entrypoint *****

    @staticmethod
    def _loadop(op, sz, dtype:Optional[DType]=None, arg=None, **kwargs):
        assert isinstance(sz, int), f"cannot create with symbolic size {sz}"
        return Tensor(TensorData.loadop(op, (sz,), Tensor.default_type if dtype is None else dtype, arg), dtype=dtype, **kwargs)

    @staticmethod
    def empty(*shape, **kwargs):
        return Tensor._loadop(LoadOps.EMPTY, prod((shape:=argfix(*shape))), **kwargs).reshape(shape)

    _seed: int = int(time.time())
    @staticmethod
    def manual_seed(seed=0): Tensor._seed = seed

    @staticmethod
    def rand(*shape, **kwargs):
        Tensor._seed += 1
        return Tensor._loadop(LoadOps.RAND, prod((shape:=argfix(*shape))), arg=Tensor._seed, **kwargs).reshape(shape)

    # ***** creation helper functions *****

    @staticmethod
    def full(shape:Tuple[shape_int, ...], fill_value, **kwargs): return Tensor(fill_value, **kwargs).reshape([1]*len(new_shape := argfix(shape))).expand(new_shape)

    @staticmethod
    def zeros(*shape, **kwargs): return Tensor.full(argfix(*shape), 0, **kwargs)

    @staticmethod
    def ones(*shape, **kwargs): return Tensor.full(argfix(*shape), 1, **kwargs)

    @staticmethod
    def arange(start, stop=None, step=1, **kwargs):
        if stop is None: stop, start = start, 0
        return Tensor.full((math.ceil((stop-start)/step),), step, **kwargs).cumsum() + (start - step)

    @staticmethod
    def eye(dim:int, **kwargs): return Tensor.full((dim,1),1,**kwargs).pad(((0,0),(0,dim))).reshape(dim*(dim+1)).shrink(((0,dim*dim),)).reshape(dim, dim)

    def full_like(self, fill_value, **kwargs): return Tensor.full(self.shape, fill_value=fill_value, dtype=kwargs.pop("dtype", self.dtype), **kwargs)
    def zeros_like(self, **kwargs): return self.full_like(0, **kwargs)
    def ones_like(self, **kwargs): return self.full_like(1, **kwargs)

    # ***** random number generation high level ops *****

    @staticmethod
    def randn(*shape, dtype:Optional[DType]=None, **kwargs) -> Tensor: return randn(*shape, dtype=dtype, **kwargs)
    @staticmethod
    def randint(*shape, low=0, high=10, **kwargs) -> Tensor: return randint(*shape, low=low, high=high, **kwargs) 
    @staticmethod
    def normal(*shape, mean=0.0, std=1.0, **kwargs) -> Tensor:  return normal(*shape, mean=mean, std=std, **kwargs) 
    @staticmethod
    def uniform(*shape, low=0.0, high=1.0, **kwargs) -> Tensor:
        return uniform(*shape, low=low, high=high, **kwargs)
    @staticmethod
    def scaled_uniform(*shape, **kwargs) -> Tensor: return scaled_uniform(*shape, **kwargs)
    def multinomial(self:Tensor, num_samples:int = 1, replacement:bool = False) -> Tensor:
        return multinomial(self, num_samples, replacement)

    # ***** toposort and backward pass *****


    def backward(self):
        return backward(self)

    # ***** movement mlops *****

    def reshape(self, shape, *args) -> Tensor: return reshape(self, shape, *args)
    def expand(self, shape, *args) -> Tensor: return expand(self, shape, *args)
    def permute(self, order, *args) -> Tensor: return permute(self, order, *args)
    def flip(self, axis, *args) -> Tensor: return flip(self, axis, *args)
    def shrink(self, arg:Tuple[Optional[Tuple[shape_int, shape_int]], ...]) -> Tensor: return shrink(self, arg)
    def pad(self, arg:Tuple[Optional[Tuple[int, int]], ...], value:float=0.0) -> Tensor: pad(self, arg, value)

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
    #        - Any Tensors passed in __getitem__ will perform (CMPEQ with arange -> MUL with self -> SUM_REDUCE) iteratively
    #                - The first iteration will expand the dim of self while consecutive iterations will reduce the dim
    #        - There's a special case where a permute is needed at the end:
    #                - if first Tensor passed in (expand dims) is not at dim 0
    #                - and following Tensors does not follow consecutively to the end of fancy indexing's dims
    def __getitem__(self, val) -> Tensor: # val: Union[int, slice, Tensor, None, Ellipsis, Tuple[Union[int, slice, Tensor, None, Ellipsis], ...]]
        def normalize_int(e, i, dim_sz):
            if -dim_sz <= e < dim_sz: return e if e != -1 else dim_sz-1
            raise IndexError(f"index {e} is out of bounds for dimension {i} with size {self.shape[i]}")

        orig_slices = list(val) if isinstance(val, tuple) else [val]
        count = defaultdict(list)
        for i,v in enumerate(orig_slices): count[type(v)].append(i)

        if (num_slices := len(count[int]) + len(count[slice]) + len(count[Tensor])) > len(self.shape): raise IndexError(f"too many indices for tensor of dimension {len(self.shape)}")
        if len(ellipsis_found := count[type(Ellipsis)]) > 1: raise IndexError("an index can only have a single ellipsis ('...')")

        ellipsis_idx = ellipsis_found[0] if ellipsis_found else len(orig_slices)
        orig_slices[ellipsis_idx:ellipsis_idx+1] = [slice(None)] * (len(self.shape) - num_slices)

        valid_slices = [v for v in orig_slices if v is not None]
        valid_slices = [v if isinstance(v, slice) else slice(y_ := normalize_int(v, i, dim_sz), y_+1) if isinstance(v, int) else slice(None) for i, (v, dim_sz) in enumerate(zip(valid_slices, self.shape))]

        start, stop, strides = zip(*y) if (y := [s.indices(dim_sz) for s, dim_sz in zip(valid_slices, self.shape)]) else ((), (), ())
        new_slice = tuple(((0, 0) if e < s else (s, e)) if st > 0 else ((0, 0) if e > s else (e+1, s+1)) for s, e, st in zip(start, stop, strides))
        sliced_tensor = self.shrink(new_slice).flip(axis=[i for i, s in enumerate(strides) if s < 0])
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

    def __setitem__(self,s,v): return self.__getitem__(s).assign(v)

    # NOTE: using slice is discouraged and things should migrate to pad and shrink
    def slice(self, arg:Sequence[Optional[Tuple[int, shape_int]]], value:float=0) -> Tensor:
        return slice(self, arg, value)

    def gather(self: Tensor, idx: Tensor, dim: int) -> Tensor: return gather(self, idx, dim)

    def cat(self, *args, dim=0) -> Tensor: return cat(self, *args, dim)

    @staticmethod
    def stack(tensors, dim=0) -> Tensor: stack(tensors, dim)
    def repeat(self, repeats) -> Tensor: repeat(self, repeats)
    def chunk(self, num:int, dim:int=0) -> List[Tensor]: chunk(self, num, dim)
    def squeeze(self, dim=None) -> Tensor: squeeze(self, dim)
    def unsqueeze(self, dim) -> Tensor: return unsqueeze(self, dim)
    # (padding_left, padding_right, padding_top, padding_bottom)
    def pad2d(self, padding:Union[List[int], Tuple[int, ...]], value:float=0) -> Tensor: return pad2d(self, padding, value)

    @property
    def T(self) -> Tensor: return self.transpose()
    def transpose(self, ax1=1, ax2=0) -> Tensor: return transpose(self, ax1, ax2)
    def flatten(self, start_dim=0): return flatten(self, start_dim)

    # ***** reduce ops *****

    def _reduce(self, fxn:Type[Function], axis:Optional[Union[int, Tuple[int, ...]]]=None, keepdim=False) -> Tensor:
        axis_: List[int] = list(range(len(self.shape))) if axis is None else ([axis] if isinstance(axis, int) else list(axis))
        axis_ = [x if x >= 0 else x+len(self.shape) for x in axis_]
        shape = tuple(s for i,s in enumerate(self.shape) if i not in axis_)
        if 0 in self.shape and 0 not in shape: return Tensor.full(tuple(1 if s == 0 else s for s in self.shape) if keepdim else shape, {function.Sum: 0, function.Max: -float("inf")}[fxn])
        ret = fxn.apply(self, new_shape=tuple([1 if i in axis_ else s for i,s in enumerate(self.shape)]))
        return ret if keepdim else ret.reshape(shape=shape)

    def sum(self, axis=None, keepdim=False): return self._reduce(function.Sum, axis, keepdim)
    def max(self, axis=None, keepdim=False): return self._reduce(function.Max, axis, keepdim)
    def min(self, axis=None, keepdim=False): return -((-self).max(axis=axis, keepdim=keepdim))

    def mean(self, axis=None, keepdim=False):
        assert all_int(self.shape), "does not support symbolic shape"
        out = self.sum(axis=axis, keepdim=keepdim)
        return out.mul(prod(out.shape)/prod(self.shape)) if 0 not in self.shape else out
    def std(self, axis=None, keepdim=False, correction=1):
        assert all_int(self.shape), "does not support symbolic shape"
        square_sum = ((self - self.mean(axis=axis, keepdim=True)).square()).sum(axis=axis, keepdim=keepdim)
        return square_sum.div(prod(self.shape)/prod(square_sum.shape)-correction).sqrt()
    def _softmax(self, axis):
        m = self - self.max(axis=axis, keepdim=True)
        e = m.exp()
        return m, e, e.sum(axis=axis, keepdim=True)

    def softmax(self, axis=-1):
        _, e, ss = self._softmax(axis)
        return e.div(ss)

    def log_softmax(self, axis=-1):
        m, _, ss = self._softmax(axis)
        return m - ss.log()

    def argmax(self, axis=None, keepdim=False):
        if axis is None:
            idx = (self == self.max(axis)) * Tensor.arange(prod(self.shape)-1,-1,-1, dtype=dtypes.int32, requires_grad=False).reshape(self.shape)
            return prod(self.shape) - idx.max() - 1
        axis = axis + len(self.shape) if axis < 0 else axis
        m = self == self.max(axis=axis, keepdim=True)
        idx = m * Tensor.arange(self.shape[axis]-1,-1,-1, dtype=dtypes.int32, requires_grad=False).reshape(self.shape[axis], *[1]*(self.ndim-axis-1))
        return self.shape[axis]-idx.max(axis=axis, keepdim=keepdim)-1
    def argmin(self, axis=None, keepdim=False): return (-self).argmax(axis=axis, keepdim=keepdim)

    # ***** processing ops *****

    def _pool(self, k_:Tuple[shape_int, ...], stride:Union[Tuple[int, ...], int]=1, dilation:Union[Tuple[int, ...], int]=1) -> Tensor:
      return _pool(self, k_, stride, dilation)


    # NOTE: these work for more than 2D
    def avg_pool2d(self, kernel_size=(2,2), stride=None, dilation=1): return avg_pool2d(self, kernel_size, stride, dilation)
    def max_pool2d(self, kernel_size=(2,2), stride=None, dilation=1): return max_pool2d(self, kernel_size, stride, dilation)

    wino = int(getenv("WINO", "0"))
    def conv2d(self, weight:Tensor, bias:Optional[Tensor]=None, groups=1, stride=1, dilation=1, padding=0) -> Tensor:
        return conv2d(self, weight, bias, groups, stride, dilation, padding)

    def dot(self, w:Tensor) -> Tensor:
        n1, n2 = len(self.shape), len(w.shape)
        assert n1 != 0 and n2 != 0, f"both arguments to matmul need to be at least 1D, but they are {n1}D and {n2}D"
        assert self.shape[-1] == w.shape[-min(n2, 2)], f"Input Tensor shapes {self.shape} and {w.shape} cannot be multiplied ({self.shape[-1]} != {w.shape[-min(n2, 2)]})"
        x = self.reshape(*self.shape[0:-1], *[1]*min(n1-1, n2-1, 1), self.shape[-1])
        w = w.reshape(*w.shape[0:-2], *[1]*min(n1-1, n2-1, 1), *w.shape[-min(n2, 2):]).transpose(-1, -min(n2, 2))
        return (x*w).sum(-1)

    def _cumsum(self, axis:int=0, _first_zero=False) -> Tensor: return self.transpose(axis,-1).pad2d((self.shape[axis]-int(not _first_zero),0))._pool((self.shape[axis],)).sum(-1).transpose(axis,-1)
    def cumsum(self, axis:int=0) -> Tensor:
        # TODO: someday the optimizer will find this on it's own
        # for now this is a two stage cumsum
        SPLIT = 256
        if self.shape[axis] <= SPLIT*2: return self._cumsum(axis)
        ret = self.transpose(axis,-1).pad2d((round_up(self.shape[axis], SPLIT)-self.shape[axis], 0))
        ret = ret.reshape(*ret.shape[0:-1], ret.shape[-1]//SPLIT, SPLIT)._cumsum(-1)
        base_add = ret[..., -1]._cumsum(-1, _first_zero=True)[..., :-1]
        base_add = base_add.unsqueeze(-1).expand(*base_add.shape, ret.shape[-1])
        def fix(x:Tensor): return x.reshape(*ret.shape[0:-2], ret.shape[-2] * ret.shape[-1])[..., -self.shape[axis]:].transpose(axis,-1)
        return fix(ret) + fix(base_add)

    # ***** mlops (unary) *****

    def neg(self): return function.Neg.apply(self)
    def log(self): return function.Log.apply(self)
    def exp(self): return function.Exp.apply(self)
    def relu(self): return function.Relu.apply(self)
    def sigmoid(self): return function.Sigmoid.apply(self)
    def sqrt(self): return function.Sqrt.apply(self)

    # ***** math functions (unary) *****

    # ***** activation functions (unary) *****

    # ***** broadcasted binary mlops *****

    def _broadcasted(self, y:Union[Tensor, float], reverse:bool=False) -> Tuple[Tensor, Tensor]:
        x: Tensor = self
        if not isinstance(y, Tensor):
            if 0 in x.shape: return x, x.full_like(y)
            y = Tensor(y, requires_grad=False, dtype=self.dtype if self.dtype != dtypes.bool else dtypes.float32)
        if reverse: x, y = y, x
        if (xshape:=x.shape) == (yshape:=y.shape): return (x, y)

        shape_delta = len(xshape) - len(yshape)
        if shape_delta > 0: y = y.reshape((1,) * shape_delta + yshape)
        elif shape_delta < 0: x = x.reshape((1,) * -shape_delta + xshape)
        if (xshape:=x.shape) == (yshape:=y.shape): return (x, y)

        shape_ret = tuple([max(x, y) for x, y in zip(xshape, yshape)])
        if xshape != shape_ret: x = x.expand(shape_ret)
        if yshape != shape_ret: y = y.expand(shape_ret)
        return (x, y)

    def _to_float(self, x:Union[Tensor, float]):
        return x.data.base.op.arg if isinstance(x, Tensor) and x.data.is_unrealized_contiguous_const() \
            and not x.requires_grad and self._broadcasted(x)[0].shape == self.shape else x

    def add(self, x:Union[Tensor, float], reverse=False) -> Tensor:
        x = self._to_float(x)
        return function.Add.apply(*self._broadcasted(x, reverse)) if x.__class__ is Tensor or x else self
    def sub(self, x:Union[Tensor, float], reverse=False) -> Tensor:
        x = self._to_float(x)
        return function.Sub.apply(*self._broadcasted(x, reverse)) if x.__class__ is Tensor or x else (-self if reverse else self)
    def mul(self, x:Union[Tensor, float], reverse=False) -> Tensor:
        x = self._to_float(x)
        if x.__class__ is not Tensor and x == 0.0: return function.Zero.apply(self)
        if x.__class__ is not Tensor and x == -1.0: return -self
        return function.Mul.apply(*self._broadcasted(x, reverse)) if x.__class__ is Tensor or x != 1.0 else self
    def div(self, x:Union[Tensor, float], reverse=False) -> Tensor:
        x = self._to_float(x)
        return function.Div.apply(*self._broadcasted(x, reverse)) if x.__class__ is Tensor or reverse or not x or not dtypes.is_float(self.dtype) else self.mul(1/x)
    def pow(self, x:Union[Tensor, float], reverse=False) -> Tensor:
        x = self._to_float(x)
        if x.__class__ is not Tensor and not reverse:
            # simple pow identities
            if x < 0: return self.reciprocal().pow(-x)
            if x == 3.0: return self*self*self
            if x == 2.0: return self*self
            if x == 1.0: return self
            if x == 0.5: return self.sqrt()
        if not isinstance(x, Tensor) and reverse and x > 0: return self.mul(math.log(x)).exp()
        ar = self.abs().log().mul(x).exp() if not reverse or isinstance(x, Tensor) else self.mul(math.log(abs(x))).exp()
        # correct sign of negative numbers raised to a power (cos has a period of 2pi so we use it here to get the oddness of the power)
        sign = (x * math.pi).cos() if isinstance(x, Tensor) else math.cos(x * math.pi) if not reverse else (self * math.pi).cos()
        # we only need to correct the sign if the base is negative
        base_sign = ((self.sign() if not reverse else x.sign() if isinstance(x, Tensor) else math.copysign(1, x)) - 1) / -2
        # we need 0 to be positive so we need to correct base_sign when the base is 0
        base_sign = base_sign - (1.5 * (1 - (self.sign().abs() if not reverse else x.sign().abs() if isinstance(x, Tensor) else abs(int(bool(x))))))
        # inject nan if the base is negative and the power is not an integer
        to_nan = (((x - x.trunc()) * 1e10).abs().clip(0, 1) if isinstance(x, Tensor) else int(bool(x - int(x))) if not reverse else ((self - self.trunc()) * 1e10).abs().clip(0, 1)) * base_sign
        inject_nan = ((((-to_nan) * 2) + 1)).log().add(1) if isinstance(to_nan, Tensor) else 1 if not to_nan else float("nan")
        return ar.mul(sign * base_sign + (1 - base_sign)).mul(inject_nan)
    def matmul(self, x:Tensor, reverse=False) -> Tensor: return x.dot(self) if reverse else self.dot(x)

    def maximum(self, x:Union[Tensor, float]) -> Tensor: return (self<x).detach().where(x, (self>x).detach().where(self, (self+x)/2))
    def minimum(self, x:Union[Tensor, float]) -> Tensor: return -((-self).maximum(-x))

    def where(self:Tensor, input_:Union[Tensor, float], other:Union[Tensor, float]):
        x_,y = self._broadcasted(input_)
        x,z = x_._broadcasted(other)
        return function.Where.apply(x, *y._broadcasted(z))

    # ***** op wrappers (wasted lines to make the typechecker happy) *****

    def __neg__(self) -> Tensor: return self.neg()

    def __add__(self, x) -> Tensor: return self.add(x)
    def __sub__(self, x) -> Tensor: return self.sub(x)
    def __mul__(self, x) -> Tensor: return self.mul(x)
    def __pow__(self, x) -> Tensor: return self.pow(x)
    def __truediv__(self, x) -> Tensor: return self.div(x)
    def __matmul__(self, x) -> Tensor: return self.matmul(x)

    def __radd__(self, x) -> Tensor: return self.add(x, True)
    def __rsub__(self, x) -> Tensor: return self.sub(x, True)
    def __rmul__(self, x) -> Tensor: return self.mul(x, True)
    def __rpow__(self, x) -> Tensor: return self.pow(x, True)
    def __rtruediv__(self, x) -> Tensor: return self.div(x, True)
    def __rmatmul__(self, x) -> Tensor: return self.matmul(x, True)

    def __iadd__(self, x) -> Tensor: return self.assign(self.add(x))
    def __isub__(self, x) -> Tensor: return self.assign(self.sub(x))
    def __imul__(self, x) -> Tensor: return self.assign(self.mul(x))
    def __ipow__(self, x) -> Tensor: return self.assign(self.pow(x))
    def __itruediv__(self, x) -> Tensor: return self.assign(self.div(x))
    def __imatmul__(self, x) -> Tensor: return self.assign(self.matmul(x))

    def __lt__(self, x) -> Tensor: return function.Less.apply(*self._broadcasted(x, False))
    def __gt__(self, x) -> Tensor: return function.Less.apply(*self._broadcasted(x, True))
    def __ge__(self, x) -> Tensor: return 1.0-(self<x)
    def __le__(self, x) -> Tensor: return 1.0-(self>x)
    def __ne__(self, x) -> Tensor: return (self<x) + (self>x)     # type: ignore
    def __eq__(self, x) -> Tensor: return 1.0-(self != x)             # type: ignore

    # ***** functional nn ops *****

    def linear(self, weight:Tensor, bias:Optional[Tensor]=None): return linear(self, weight, bias)

    def binary_crossentropy(self, y:Tensor) -> Tensor: return binary_crossentropy(self, y)

    def binary_crossentropy_logits(self, y:Tensor) -> Tensor: return binary_crossentropy_logits(self, y)

    def sparse_categorical_crossentropy(self, Y, ignore_index=-1) -> Tensor: return sparse_categorical_crossentropy(self, Y, ignore_index)

    # ***** cast ops *****

    def cast(self, dtype:DType) -> Tensor: return function.Cast.apply(self, dtype=dtype) if self.dtype != dtype else self
    def bitcast(self, dtype:DType) -> Tensor:
        assert self.dtype.itemsize == dtype.itemsize, "can't bitcast mismatched dtype itemsizes"
        return function.Cast.apply(self, dtype=dtype, bitcast=True) if self.dtype != dtype else self
    def float(self) -> Tensor: return self.cast(dtypes.float32)
    def half(self) -> Tensor: return self.cast(dtypes.float16)

    # ***** convenience stuff *****

    @property
    def ndim(self) -> int: return len(self.shape)
    def numel(self) -> shape_int: return prod(self.shape)
    def element_size(self) -> int: return self.dtype.itemsize
    def nbytes(self) -> int: return self.numel() * self.element_size()
    def is_floating_point(self) -> bool: return dtypes.is_float(self.dtype)