from __future__ import annotations
from teenygrad.helpers import DType, dtypes, DEBUG
from teenygrad.ops import UnaryOps, BinaryOps, ReduceOps, TernaryOps, LoadOps
import numpy as np


class TensorData:
    """Class representing a buffer that delays computation until necessary (lazy evaluation)."""

    def __init__(self, data: np.ndarray):
        """Initialize the LazyBuffer with a numpy array."""
        self.data = data

    @property
    def dtype(self):
        """Returns the data type of the buffer."""
        return dtypes.from_np(self.data.dtype)

    @property
    def shape(self):
        """Returns the shape of the numpy array."""
        return self.data.shape

    def __repr__(self):
        return f"<TD {self.shape} {self.dtype}>"

    def is_unrealized_contiguous_const(self):
        """Checks if the buffer is an unrealized contiguous constant."""
        return False

    @staticmethod
    def loadop(op, shape, dtype, arg=None, src=None) -> TensorData:
        """
        Load operation for creating a LazyBuffer based on specified operation. 
        Supported operations: RAND, CONST, EMPTY.
        """
        if op == LoadOps.RAND:
            return TensorData(np.random.default_rng(arg).random(size=shape, dtype=dtype.np))
        elif op == LoadOps.CONST:
            return TensorData(np.full(shape, arg, dtype=dtype.np))
        elif op == LoadOps.EMPTY:
            return TensorData(np.empty(shape, dtype=dtype.np))
        else:
            raise NotImplementedError(op)

    def const(self, x) -> TensorData:
        """Returns a new LazyBuffer with a constant value."""
        return TensorData(np.full_like(self.data, x))

    def cast(self, dtype: DType, bitcast: bool = False) -> TensorData:
        """Casts the buffer to a different data type."""
        if bitcast:
            return TensorData(self.data.view(dtype.np))
        else:
            return TensorData(self.data.astype(dtype.np))

    def exec(self, op, *srcs: TensorData):
        """Execute a unary, binary, or ternary operation on the buffer."""
        unary_ops = {
            UnaryOps.NEG: np.negative,
            UnaryOps.EXP2: np.exp2,
            UnaryOps.LOG2: np.log2,
            UnaryOps.SIN: np.sin,
            UnaryOps.SQRT: np.sqrt,
        }
        binary_ops = {
            BinaryOps.ADD: np.add,
            BinaryOps.SUB: np.subtract,
            BinaryOps.MUL: np.multiply,
            BinaryOps.DIV: np.divide,
            BinaryOps.MAX: np.maximum,
            BinaryOps.CMPLT: np.less,
        }
        ternary_ops = {
            TernaryOps.WHERE: np.where,
        }

        if op in unary_ops:
            return TensorData(unary_ops[op](self.data))
        elif op in binary_ops and srcs:
            return TensorData(binary_ops[op](self.data, srcs[0].data))
        elif op in ternary_ops and len(srcs) == 2:
            return TensorData(ternary_ops[op](self.data, srcs[0].data, srcs[1].data))
        else:
            raise NotImplementedError(f"Operation {op} not implemented or wrong number of sources")

    def reduce(self, op, new_shape):
        """
        Perform reduction operations on the buffer.
        Supported operations: ReduceOps.
        """
        if DEBUG >= 1: 
            print(op, self, new_shape)
        assert len(self.shape) == len(new_shape), "reduce shapes must have same dimensions"
        axis = tuple(i for i, (a, b) in enumerate(zip(self.shape, new_shape)) if a != b)

        # Reduction operations
        if op == ReduceOps.SUM: 
            return TensorData(self.data.sum(axis, dtype=self.data.dtype, keepdims=True))
        elif op == ReduceOps.MAX: 
            return TensorData(self.data.max(axis, keepdims=True))
        else: 
            raise NotImplementedError(op)

    # Movement operations
    def reshape(self, arg):
        """Reshape the buffer to a new shape."""
        return TensorData(self.data.reshape(arg))

    def expand(self, arg):
        """Expand the buffer to a new shape by broadcasting."""
        return TensorData(np.broadcast_to(self.data, arg))

    def shrink(self, arg):
        """Shrink the buffer by slicing it."""
        return TensorData(self.data[tuple(slice(p[0], p[1], None) for p in arg)])

    def permute(self, arg):
        """Permute the axes of the buffer."""
        return TensorData(self.data.transpose(arg))

    def pad(self, arg):
        """Pad the buffer with specified padding."""
        return TensorData(np.pad(self.data, arg))

    def stride(self, arg):
        """Apply striding to the buffer."""
        return TensorData(self.data[tuple(slice(None, None, i) for i in arg)])
