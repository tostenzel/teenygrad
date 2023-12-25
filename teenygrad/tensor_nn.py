# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
from __future__ import annotations
import math
from typing import Tuple, Optional, Union

from teenygrad.helpers import make_pair, getenv, flatten, dtypes, all_int, shape_int



# ***** processing ops *****

def _pool(tensor: 'Tensor', k_:Tuple[shape_int, ...], stride:Union[Tuple[int, ...], int], dilation:Union[Tuple[int, ...], int]) -> 'Tensor':
    assert len(tensor.shape) >= len(k_), f"can't pool {tensor.shape} with {k_}"
    assert all_int(tensor.shape) and all_int(k_), f"does not support symbolic {tensor.shape=}, {k_=}"
    s_, d_ = make_pair(stride, len(k_)), make_pair(dilation, len(k_))
    assert len(k_) == len(s_) and len(k_) == len(d_), f"stride/dilation mismatch kernel:{k_} stride:{s_} dilation:{d_}"
    slc_prefix, prefix, i_ = [(0,x) for x in tensor.shape[0:-len(k_)]], tensor.shape[0:-len(k_)], tensor.shape[-len(k_):]
    if any(k > s for k,s in zip(k_, s_)) or any(d != 1 for d in d_):
        o_ = [(i - d * (k-1) - 1)//s + 1 for i,d,k,s in zip(i_, d_, k_, s_)]
        e_ = [math.ceil(k*(i+d) / i) for k,i,d in zip(k_, i_, d_)]    # expands such that we don't need padding
        xup = tensor.reshape(*prefix, *flatten((1,i) for i in i_)).expand(*prefix, *flatten((e,i) for e,i in zip(e_, i_))).reshape(*prefix, *[e*i for e,i in zip(e_, i_)])
        # slide by dilation
        xup = xup.slice(slc_prefix + [(0,k*(i+d)) for k,i,d in zip(k_, i_, d_)])
        xup = xup.reshape(*prefix, *flatten((k,i+d) for k,i,d in zip(k_, i_, d_)))
        xup = xup.slice(slc_prefix + flatten(((0,k), (0,o*s)) for k,o,s in zip(k_, o_, s_)))
        # handle stride, and permute to move reduce to the end
        xup = xup.reshape(*prefix, *flatten((k,o,s) for k,o,s in zip(k_, o_, s_)))
        xup = xup.slice(slc_prefix + flatten(((0,k), (0,o), (0,1)) for k,o in zip(k_, o_)))
        xup = xup.reshape(*prefix, *flatten((k,o) for k,o in zip(k_, o_)))
        return xup.permute(*range(len(prefix)), *[len(prefix)+i*2+1 for i in range(len(k_))], *[len(prefix)+i*2 for i in range(len(k_))])
    # TODO: once the shapetracker can optimize well, remove this alternative implementation. or not if the CPU implementation doesn't use ShapeTracker
    o_ = [(i+(s-k))//s for i,s,k in zip(i_, s_, k_)]
    xup = tensor.slice(slc_prefix + [(0,o*s) for o,s in zip(o_, s_)])
    xup = xup.reshape(*prefix, *flatten(((o, s) for o,s in zip(o_, s_))))
    xup = xup.slice(slc_prefix + flatten(((0,o), (0,k)) for o,k in zip(o_, k_)))
    return xup.permute(*range(len(prefix)), *[len(prefix)+i*2 for i in range(len(k_))], *[len(prefix)+i*2+1 for i in range(len(k_))])

# NOTE: these work for more than 2D
def avg_pool2d(tensor: 'Tensor', kernel_size, stride, dilation): return tensor._pool(make_pair(kernel_size), stride if stride is not None else kernel_size, dilation).mean(axis=tuple(range(0-len(make_pair(kernel_size)), 0)))
def max_pool2d(tensor: 'Tensor', kernel_size, stride, dilation): return tensor._pool(make_pair(kernel_size), stride if stride is not None else kernel_size, dilation).max(axis=tuple(range(0-len(make_pair(kernel_size)), 0)))

wino = int(getenv("WINO", "0"))
def conv2d(tensor: 'Tensor', weight:'Tensor', bias:Optional['Tensor']=None, groups=1, stride=1, dilation=1, padding=0) -> 'Tensor':
    from teenygrad.tensor import Tensor
    (bs,cin_), (cout,cin), HW = tensor.shape[:2], weight.shape[:2], weight.shape[2:]
    assert groups*cin == cin_ and len(tensor.shape) == len(weight.shape), f"Input 'Tensor' shape {tensor.shape} does not match the shape of the weights {weight.shape}. ({groups*cin} vs. {cin_})"
    if isinstance(padding, (tuple,list)): assert len(padding) == 2*len(HW) or len(padding) == len(HW), f"Expected padding of length {2*len(HW)} or {len(HW)}, but got {len(padding)} for 'Tensor' of shape {tensor.shape}"
    padding_ = [padding]*2*len(HW) if isinstance(padding, int) else (padding if len(padding) == 2*len(HW) else [p for p in padding for _ in range(2)][::-1])

    # conv2d is a pooling op (with padding)
    x = tensor.pad2d(padding_)._pool(HW, stride, dilation)     # (bs, groups*cin, oy, ox, H, W)
    rcout, oyx = cout//groups, x.shape[2:-len(HW)]
    if not all(x == 3 for x in HW) or stride != 1 or dilation != 1 or not Tensor.wino:
        # normal conv
        x = x.reshape(bs, groups, cin, 1, *oyx, *HW).expand(bs, groups, cin, rcout, *oyx, *HW).permute(0,1,3,*[4+i for i in range(len(oyx))],2,*[4+len(oyx)+i for i in range(len(HW))])

        # conv! broadcasted to (bs, groups, rcout, *oyx, cin, *HW)
        ret = (x * weight.reshape(1, groups, rcout, *[1] * len(oyx), cin, *HW)).sum([-1-i for i in range(1+len(oyx))], keepdim=True).reshape(bs, cout, *oyx)
        return ret if bias is None else ret.add(bias.reshape(1, -1, *[1] * len(HW)))


# ***** functional nn ops *****

def linear(tensor: 'Tensor', weight:'Tensor', bias:Optional['Tensor']=None):
    x = tensor.mul(weight) if len(weight.shape) == 1 else tensor.dot(weight)
    return x.add(bias) if bias is not None else x

def binary_crossentropy(tensor: 'Tensor', y:'Tensor') -> 'Tensor':
    return (-y*tensor.log() - (1-y)*(1-tensor).log()).mean()

def binary_crossentropy_logits(tensor: 'Tensor', y:'Tensor') -> 'Tensor':
    return (tensor.maximum(0) - y * tensor + (1 + tensor.abs().__neg__().exp()).log()).mean()

def sparse_categorical_crossentropy(tensor: 'Tensor', Y, ignore_index=-1) -> 'Tensor':
    from teenygrad.tensor import Tensor
    # NOTE: tensor is a logits input
    loss_mask = Y != ignore_index
    y_counter = Tensor.arange(tensor.shape[-1], dtype=dtypes.int32, requires_grad=False).unsqueeze(0).expand(Y.numel(), tensor.shape[-1])
    y = ((y_counter == Y.flatten().reshape(-1, 1)).where(-1.0, 0) * loss_mask.reshape(-1, 1)).reshape(*Y.shape, tensor.shape[-1])
    return tensor.log_softmax().mul(y).sum() / loss_mask.sum()