from __future__ import annotations

from typing import Tuple, Union

import math

from teenygrad.helpers import dtypes
import teenygrad.function as function


# broadcasted binary mlops

def _broadcasted(tensor: 'Tensor', y:Union['Tensor', float], reverse:bool=False) -> Tuple['Tensor', 'Tensor']:
    from teenygrad.tensor import Tensor
    x: Tensor = tensor
    if not isinstance(y, Tensor):
        if 0 in x.shape: return x, x.full_like(y)
        y = Tensor(y, requires_grad=False, dtype=tensor.dtype if tensor.dtype != dtypes.bool else dtypes.float32)
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

def _to_float(tensor: 'Tensor', x:Union['Tensor', float]):
    from teenygrad.tensor import Tensor
    return x.data.base.op.arg if isinstance(x, Tensor) and x.data.is_unrealized_contiguous_const() \
        and not x.requires_grad and tensor._broadcasted(x)[0].shape == tensor.shape else x

def add(tensor: 'Tensor', x:Union['Tensor', float], reverse=False) -> 'Tensor':
    from teenygrad.tensor import Tensor
    x = tensor._to_float(x)
    return function.Add.apply(*tensor._broadcasted(x, reverse)) if x.__class__ is Tensor or x else tensor

def sub(tensor: 'Tensor', x:Union['Tensor', float], reverse=False) -> 'Tensor':
    from teenygrad.tensor import Tensor
    x = tensor._to_float(x)
    return function.Sub.apply(*tensor._broadcasted(x, reverse)) if x.__class__ is Tensor or x else (-tensor if reverse else tensor)

def mul(tensor: 'Tensor', x:Union['Tensor', float], reverse=False) -> 'Tensor':
    from teenygrad.tensor import Tensor
    x = tensor._to_float(x)
    if x.__class__ is not Tensor and x == 0.0: return function.Zero.apply(tensor)
    if x.__class__ is not Tensor and x == -1.0: return -tensor
    return function.Mul.apply(*tensor._broadcasted(x, reverse)) if x.__class__ is Tensor or x != 1.0 else tensor

def div(tensor: 'Tensor', x:Union['Tensor', float], reverse=False) -> 'Tensor':
    from teenygrad.tensor import Tensor
    x = tensor._to_float(x)
    return function.Div.apply(*tensor._broadcasted(x, reverse)) if x.__class__ is Tensor or reverse or not x or not dtypes.is_float(tensor.dtype) else tensor.mul(1/x)

def pow(tensor: 'Tensor', x:Union['Tensor', float], reverse=False) -> 'Tensor':
    from teenygrad.tensor import Tensor
    x = tensor._to_float(x)
    if x.__class__ is not Tensor and not reverse:
        # simple pow identities
        if x < 0: return tensor.reciprocal().pow(-x)
        if x == 3.0: return tensor*tensor*tensor
        if x == 2.0: return tensor*tensor
        if x == 1.0: return tensor
        if x == 0.5: return tensor.sqrt()
    if not isinstance(x, Tensor) and reverse and x > 0: return tensor.mul(math.log(x)).exp()
    ar = tensor.abs().log().mul(x).exp() if not reverse or isinstance(x, Tensor) else tensor.mul(math.log(abs(x))).exp()
    # correct sign of negative numbers raised to a power (cos has a period of 2pi so we use it here to get the oddness of the power)
    sign = (x * math.pi).cos() if isinstance(x, Tensor) else math.cos(x * math.pi) if not reverse else (tensor * math.pi).cos()
    # we only need to correct the sign if the base is negative
    base_sign = ((tensor.sign() if not reverse else x.sign() if isinstance(x, Tensor) else math.copysign(1, x)) - 1) / -2
    # we need 0 to be positive so we need to correct base_sign when the base is 0
    base_sign = base_sign - (1.5 * (1 - (tensor.sign().abs() if not reverse else x.sign().abs() if isinstance(x, Tensor) else abs(int(bool(x))))))
    # inject nan if the base is negative and the power is not an integer
    to_nan = (((x - x.trunc()) * 1e10).abs().clip(0, 1) if isinstance(x, Tensor) else int(bool(x - int(x))) if not reverse else ((tensor - tensor.trunc()) * 1e10).abs().clip(0, 1)) * base_sign
    inject_nan = ((((-to_nan) * 2) + 1)).log().add(1) if isinstance(to_nan, Tensor) else 1 if not to_nan else float("nan")
    return ar.mul(sign * base_sign + (1 - base_sign)).mul(inject_nan)

def matmul(tensor: 'Tensor', x:'Tensor', reverse=False) -> 'Tensor': return x.dot(tensor) if reverse else tensor.dot(x)

def maximum(tensor: 'Tensor', x:Union['Tensor', float]) -> 'Tensor': return (tensor<x).detach().where(x, (tensor>x).detach().where(tensor, (tensor+x)/2))

def minimum(tensor: 'Tensor', x:Union['Tensor', float]) -> 'Tensor': return -((-tensor).maximum(-x))

def where(tensor: 'Tensor', input_:Union['Tensor', float], other:Union['Tensor', float]):
    x_,y = tensor._broadcasted(input_)
    x,z = x_._broadcasted(other)
    return function.Where.apply(x, *y._broadcasted(z))