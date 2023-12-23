from __future__ import annotations
from teenygrad.helpers import DType, dtypes, DEBUG
from teenygrad.ops import UnaryOps, BinaryOps, ReduceOps, TernaryOps, LoadOps
import numpy as np

class RawCPUBuffer:
    """Wrapper class for numpy array to be used as a CPU buffer."""
    def __init__(self, x):
        self.x = x

    def toCPU(self):
        """Returns the underlying numpy array."""
        return self.x

class LazyBuffer:
    """Class representing a buffer that delays computation until necessary (lazy evaluation)."""

    def __init__(self, buf: np.ndarray):
        """Initialize the LazyBuffer with a numpy array."""
        self._np = buf

    @property
    def base(self):
        """Returns the base buffer (itself, as this is the base class)."""
        return self

    @property
    def dtype(self):
        """Returns the data type of the buffer."""
        return dtypes.from_np(self._np.dtype)

    @property
    def realized(self):
        """Returns a RawCPUBuffer wrapping the numpy array."""
        return RawCPUBuffer(self._np)

    @property
    def shape(self):
        """Returns the shape of the numpy array."""
        return self._np.shape

    def __repr__(self):
        return f"<LB {self.shape} {self.dtype}>"

    def schedule(self, seen=None):
        """Returns an empty schedule as there's no delayed computation in this buffer."""
        return []

    def is_unrealized_contiguous_const(self):
        """Checks if the buffer is an unrealized contiguous constant."""
        return False

    @staticmethod
    def fromCPU(x):
        """Creates a LazyBuffer from a numpy array."""
        return LazyBuffer(x)

    @staticmethod
    def loadop(op, shape, dtype, arg=None, src=None) -> LazyBuffer:
        """
        Load operation for creating a LazyBuffer based on specified operation. 
        Supported operations: RAND, CONST, EMPTY.
        """
        if op == LoadOps.RAND:
            return LazyBuffer(np.random.default_rng(arg).random(size=shape, dtype=dtype.np))
        elif op == LoadOps.CONST:
            return LazyBuffer(np.full(shape, arg, dtype=dtype.np))
        elif op == LoadOps.EMPTY:
            return LazyBuffer(np.empty(shape, dtype=dtype.np))
        else:
            raise NotImplementedError(op)

    def contiguous(self):
        """Returns the buffer as it is always contiguous."""
        return self

    def const(self, x) -> LazyBuffer:
        """Returns a new LazyBuffer with a constant value."""
        return LazyBuffer(np.full_like(self._np, x))

    def cast(self, dtype: DType, bitcast: bool = False) -> LazyBuffer:
        """Casts the buffer to a different data type."""
        if bitcast:
            return LazyBuffer(self._np.view(dtype.np))
        else:
            return LazyBuffer(self._np.astype(dtype.np))

    def e(self, op, *srcs: LazyBuffer):
        """
        Execute a unary, binary, or ternary operation on the buffer.
        Supported operations: UnaryOps, BinaryOps, TernaryOps.
        """
        if DEBUG >= 1: 
            print(op, self, srcs)

        # Unary operations
        if op == UnaryOps.NEG: 
            ret = -self._np
        elif op == UnaryOps.EXP2: 
            ret = np.exp2(self._np)
        elif op == UnaryOps.LOG2: 
            ret = np.log2(self._np)
        elif op == UnaryOps.SIN: 
            ret = np.sin(self._np)
        elif op == UnaryOps.SQRT: 
            ret = np.sqrt(self._np)

        # Binary operations
        elif op == BinaryOps.ADD: 
            ret = self._np + srcs[0]._np
        elif op == BinaryOps.SUB: 
            ret = self._np - srcs[0]._np
        elif op == BinaryOps.MUL: 
            ret = self._np * srcs[0]._np
        elif op == BinaryOps.DIV: 
            ret = self._np / srcs[0]._np
        elif op == BinaryOps.MAX: 
            ret = np.maximum(self._np, srcs[0]._np)
        elif op == BinaryOps.CMPLT: 
            ret = self._np < srcs[0]._np

        # Ternary operations
        elif op == TernaryOps.WHERE: 
            ret = np.where(self._np, srcs[0]._np, srcs[1]._np)
        else: 
            raise NotImplementedError(op)

        return LazyBuffer(ret.astype(self.dtype.np if len(srcs) == 0 else max(self.dtype, *[x.dtype for x in srcs]).np, copy=False))

    def r(self, op, new_shape):
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
            return LazyBuffer(self._np.sum(axis, dtype=self._np.dtype, keepdims=True))
        elif op == ReduceOps.MAX: 
            return LazyBuffer(self._np.max(axis, keepdims=True))
        else: 
            raise NotImplementedError(op)

    # Movement operations
    def reshape(self, arg):
        """Reshape the buffer to a new shape."""
        return LazyBuffer(self._np.reshape(arg))

    def expand(self, arg):
        """Expand the buffer to a new shape by broadcasting."""
        return LazyBuffer(np.broadcast_to(self._np, arg))

    def shrink(self, arg):
        """Shrink the buffer by slicing it."""
        return LazyBuffer(self._np[tuple(slice(p[0], p[1], None) for p in arg)])

    def permute(self, arg):
        """Permute the axes of the buffer."""
        return LazyBuffer(self._np.transpose(arg))

    def pad(self, arg):
        """Pad the buffer with specified padding."""
        return LazyBuffer(np.pad(self._np, arg))

    def stride(self, arg):
        """Apply striding to the buffer."""
        return LazyBuffer(self._np[tuple(slice(None, None, i) for i in arg)])
