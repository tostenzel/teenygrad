"""Contains the core differentiable functions with forward and backward passes.

These functions are compositions of basic numpy operations wrapped in data.TensorData methods.
Many other methods in the tensor.Tensor class are composed of these functions and can therefore be
backpropagated, too.

All these functions are applied to and return TensorData objects that can be called with Tensor.data.

"""
import math
from typing import Tuple, Optional, cast
from teenygrad.helpers import argsort, DType, shape_int
from teenygrad.ops import UnaryOps, BinaryOps, TernaryOps, ReduceOps
from teenygrad.data import TensorData

from typing import Type


class Function:
    """Base class for all differentiable operations in the autograd system.

    This class represents a mathematical operation and serves as a node in the computational graph.
    Subclasses of `Function` implement the forward and backward passes of differentiable operations.

    Attributes:
        needs_input_grad (list[bool]): List indicating whether each input tensor requires gradient computation.
        requires_grad (bool or None): True if any input tensor requires a gradient, False otherwise.
        parents (tuple[Tensor]): Tuple of input tensors from which this operation is derived.

    """
    def __init__(self, *tensors: 'Tensor'):
        """Initializes the Function with input tensors, determining if gradients are required.

        Args:
            *tensors (Tensor): Input tensors for the function. The function will check these tensors to determine
                               if it should compute gradients during the backward pass.

        """
        # Determine if gradients are needed for each input tensor
        self.needs_input_grad = [t.requires_grad for t in tensors]
        # If any input requires grad, or if it's unknown (None), set this function to require grad
        self.requires_grad = True if any(self.needs_input_grad) else None if None in self.needs_input_grad else False
        # Store parent tensors for gradient computation in the backward pass if needed
        if self.requires_grad:
            self.parents = tensors

    def forward(self, *args, **kwargs):
        """Forward pass of the function.

        Computes the output Tensor from input Tensors. This method should be implemented by all subclasses.
        It defines the actual operation performed at this node of the computational graph.

        """
        raise NotImplementedError(f"forward not implemented for {type(self)}")

    def backward(self, *args, **kwargs):
        """Backward pass (gradient computation) of the function.

        Computes the gradient of the function with respect to its inputs. This method should be implemented by all subclasses.
        It is used during the backward phase of automatic differentiation.

        """
        raise RuntimeError(f"backward not implemented for {type(self)}")

    @classmethod
    def apply(fxn: Type['Function'], *x: 'Tensor', **kwargs) -> 'Tensor':
        """Apply the function to the given tensors and return the result.

        This method handles the setup of the computational graph by creating a context for the function,
        performing the forward pass, and setting up the backward pass if necessary.

        Args:
            fxn (Type[Function]): The function class to be applied.
            *x (Tensor): Input tensors for the function.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: The result of applying the function.
        """
        from teenygrad.tensor import Tensor
        # Create a context (an instance of the function) for the computation
        ctx = fxn(*x)
        # Compute the forward pass
        ret = Tensor(ctx.forward(*[t.data for t in x], **kwargs), requires_grad=ctx.requires_grad)
        # If gradients are required and global gradient computation is not turned off, store the context
        if ctx.requires_grad and not Tensor.no_grad:
            ret._ctx = ctx  # Context is stored for use by the autograd engine
        return ret


class Cast(Function):
    """Handles casting of TensorData to a different data type."""
    def forward(self, x: TensorData, dtype: DType, bitcast: bool = False) -> TensorData:
        self.input_dtype, self.bitcast = x.dtype, bitcast
        return x.cast(dtype, bitcast)

    def backward(self, grad_output: TensorData) -> TensorData:
        return grad_output.cast(self.input_dtype, self.bitcast)


# ----------------------------------------------------------------------------------------------------------------------
# unary ops

class Zero(Function):
    """Represents a function that returns zero regardless of the input."""
    def forward(self, x: TensorData) -> TensorData:
        return x.const(0)

    def backward(self, grad: TensorData) -> TensorData:
        return grad.const(0)


class Neg(Function):
    def forward(self, x: TensorData) -> TensorData:
        return x.elementwise(UnaryOps.NEG)

    def backward(self, grad: TensorData) -> TensorData:
        return grad.elementwise(UnaryOps.NEG)


class Sin(Function):
    def forward(self, x: TensorData) -> TensorData:
        self.x = x
        return x.elementwise(UnaryOps.SIN)

    def backward(self, grad: TensorData) -> TensorData:
        return self.x.const(math.pi / 2).elementwise(BinaryOps.SUB, self.x).elementwise(UnaryOps.SIN).elementwise(BinaryOps.MUL, grad)


class Relu(Function):
    def forward(self, x: TensorData) -> TensorData:
        self.ret = x.elementwise(BinaryOps.MAX, x.const(0))
        return self.ret

    def backward(self, grad_output: TensorData) -> TensorData:
        return self.ret.const(0).elementwise(BinaryOps.CMPLT, self.ret).elementwise(BinaryOps.MUL, grad_output)


class Log(Function):
    def forward(self, x: TensorData) -> TensorData:
        self.x = x
        return x.elementwise(UnaryOps.LOG2).elementwise(BinaryOps.MUL, x.const(math.log(2)))

    def backward(self, grad_output: TensorData) -> TensorData:
        return grad_output.elementwise(BinaryOps.DIV, self.x)


class Exp(Function):
    def forward(self, x: TensorData) -> TensorData:
        self.ret = x.elementwise(BinaryOps.MUL, x.const(1/math.log(2))).elementwise(UnaryOps.EXP2)
        return self.ret

    def backward(self, grad_output: TensorData) -> TensorData:
        return self.ret.elementwise(BinaryOps.MUL, grad_output)


class Sqrt(Function):
    def forward(self, x: TensorData) -> TensorData:
        self.ret = x.elementwise(UnaryOps.SQRT)
        return self.ret

    def backward(self, grad_output: TensorData) -> TensorData:
        return grad_output.elementwise(BinaryOps.DIV, self.ret.elementwise(BinaryOps.MUL, self.ret.const(2)))


# NOTE: the implicit derivative of sigmoid is not stable
# https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
# TODO: have the backend automatically find this
class Sigmoid(Function):
    def forward(self, x: TensorData) -> TensorData:
        self.ret = x.const(1).elementwise(BinaryOps.DIV, x.const(1).elementwise(BinaryOps.ADD, x.elementwise(BinaryOps.MUL, x.const(-1/math.log(2))).elementwise(UnaryOps.EXP2)))
        return self.ret

    def backward(self, grad_output: TensorData) -> TensorData:
        return self.ret.elementwise(BinaryOps.MUL, self.ret.const(1).elementwise(BinaryOps.SUB, self.ret)).elementwise(BinaryOps.MUL, grad_output)

#-----------------------------------------------------------------------------------------------------------------------
# binary ops

class Less(Function):
    def forward(self, x: TensorData, y: TensorData) -> TensorData:
        return x.elementwise(BinaryOps.CMPLT, y)


class Add(Function):
    def forward(self, x: TensorData, y: TensorData) -> TensorData:
        return x.elementwise(BinaryOps.ADD, y)

    def backward(self, grad_output: TensorData) -> Tuple[Optional[TensorData], Optional[TensorData]]:
        return grad_output if self.needs_input_grad[0] else None, \
            grad_output if self.needs_input_grad[1] else None


class Sub(Function):
    def forward(self, x: TensorData, y: TensorData) -> TensorData:
        return x.elementwise(BinaryOps.SUB, y)

    def backward(self, grad_output: TensorData) -> Tuple[Optional[TensorData], Optional[TensorData]]:
        return grad_output if self.needs_input_grad[0] else None, \
            grad_output.elementwise(UnaryOps.NEG) if self.needs_input_grad[1] else None


class Mul(Function):
    def forward(self, x: TensorData, y: TensorData) -> TensorData:
        self.x, self.y = x, y
        return x.elementwise(BinaryOps.MUL, y)

    def backward(self, grad_output: TensorData) -> Tuple[Optional[TensorData], Optional[TensorData]]:
        return self.y.elementwise(BinaryOps.MUL, grad_output) if self.needs_input_grad[0] else None, \
            self.x.elementwise(BinaryOps.MUL, grad_output) if self.needs_input_grad[1] else None


class Div(Function):
    def forward(self, x: TensorData, y: TensorData) -> TensorData:
        self.x, self.y = x, y
        return x.elementwise(BinaryOps.DIV, y)

    def backward(self, grad_output: TensorData) -> Tuple[Optional[TensorData], Optional[TensorData]]:
        return grad_output.elementwise(BinaryOps.DIV, self.y) if self.needs_input_grad[0] else None, \
            grad_output.elementwise(UnaryOps.NEG).elementwise(BinaryOps.MUL, self.x).elementwise(BinaryOps.DIV, self.y.elementwise(BinaryOps.MUL, self.y)) if self.needs_input_grad[1] else None


# ----------------------------------------------------------------------------------------------------------------------
# ternary ops

class Where(Function):
    """Ternary conditional operation."""
    def forward(self, x: TensorData, y: TensorData, z: TensorData) -> TensorData:
        self.x = x
        return x.elementwise(TernaryOps.WHERE, y, z)

    def backward(self, grad_output: TensorData) -> Tuple[None, Optional[TensorData], Optional[TensorData]]:
        return None, \
            self.x.elementwise(TernaryOps.WHERE, grad_output, grad_output.const(0)) if self.needs_input_grad[1] else None, \
            self.x.elementwise(TernaryOps.WHERE, grad_output.const(0), grad_output) if self.needs_input_grad[2] else None


# ----------------------------------------------------------------------------------------------------------------------
# reduce ops
  
class Sum(Function):
    def forward(self, x: TensorData, new_shape: Tuple[int, ...]) -> TensorData:
        self.input_shape = x.shape
        return x.reduce(ReduceOps.SUM, new_shape)

    def backward(self, grad_output: TensorData) -> TensorData:
        return grad_output.expand(self.input_shape)


class Max(Function):
    def forward(self, x: TensorData, new_shape: Tuple[int, ...]) -> TensorData:
        self.x, self.ret = x, x.reduce(ReduceOps.MAX, new_shape)
        return self.ret

    def backward(self, grad_output: TensorData) -> TensorData:
        # 1s in locations where the max was chosen (can be two locations)
        max_is_1s = self.x.const(1.0).elementwise(BinaryOps.SUB, self.x.elementwise(BinaryOps.CMPLT, self.ret.expand(self.x.shape)))
        div = max_is_1s.reduce(ReduceOps.SUM, grad_output.shape).expand(self.x.shape)
        return max_is_1s.elementwise(BinaryOps.DIV, div).elementwise(BinaryOps.MUL, grad_output.expand(self.x.shape))


# ----------------------------------------------------------------------------------------------------------------------
# movement ops

# NOTE: this is sum in reverse
class Expand(Function):
    def forward(self, x: TensorData, shape: Tuple[int, ...]) -> TensorData:
        self.input_shape = x.shape
        return x.expand(shape)

    def backward(self, grad_output: TensorData) -> TensorData:
        return grad_output.reduce(ReduceOps.SUM, self.input_shape)


class Reshape(Function):
    def forward(self, x: TensorData, shape: Tuple[int, ...]) -> TensorData:
        self.input_shape = x.shape
        return x.reshape(shape)

    def backward(self, grad_output: TensorData) -> TensorData:
        return grad_output.reshape(self.input_shape)


class Permute(Function):
    def forward(self, x: TensorData, order: Tuple[int, ...]) -> TensorData:
        self.input_order = order
        return x.permute(order)

    def backward(self, grad_output: TensorData) -> TensorData:
        return grad_output.permute(argsort(self.input_order))


class Pad(Function):
    def forward(self, x: TensorData, arg: Tuple[Tuple[int, int], ...]) -> TensorData:
        self.narg = tuple([(p[0], s+p[0]) for s,p in zip(x.shape, arg)])
        return x.pad(arg)

    def backward(self, grad_output: TensorData) -> TensorData:
        return grad_output.shrink(self.narg)


class Shrink(Function):
    """Shrink operation."""
    def forward(self, x: TensorData, arg: Tuple[Tuple[shape_int, shape_int], ...]) -> TensorData:
        self.narg = tuple([(p[0], s-p[1]) for s,p in zip(x.shape, arg)])
        return x.shrink(arg)

    def backward(self, grad_output: TensorData) -> TensorData:
        assert all(isinstance(x[0], int) and isinstance(x[1], int) for x in self.narg), "symbolic shrink does not support backward"
        # need this cast because mypy cannot narrow the type even with assert
        return grad_output.pad(cast(Tuple[Tuple[int, int], ...], self.narg))
