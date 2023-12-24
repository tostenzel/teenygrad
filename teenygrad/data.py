"""Defines the TensorData class, a container for tensor data (tensor.Tensor.data), represented as numpy arrays.

It facilitates direct manipulation of tensor data through a range of basic operations. These operations are building
blocks for defining forward and backward passes of differentiable Functions in the computational graph.
The ops are executed immediately on the CPU using numpy. This approach contrasts with deferred computation models that
analyze subsequent delayed operations in order to find an optimized equivalent final optimization at the point where
execution is actually required. The deferred model is common in industrial-scale libraries.

"""
from typing import Tuple
import numpy as np
from teenygrad.ops import UnaryOps, BinaryOps, TernaryOps, ReduceOps, LoadOps   # consider reading the docs there
from teenygrad.helpers import DType, dtypes, DEBUG


class TensorData:
    """A class that encapsulates numpy array data and provides methods for direct tensor operations."""

    def __init__(self, data: np.ndarray):
        """Initialize the TensorData with a numpy array."""
        self.data = data

    @property
    def dtype(self) -> DType:
        """Return the data type of the numpy array."""
        return dtypes.from_np(self.data.dtype)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the numpy array."""
        return self.data.shape

    def __repr__(self) -> str:
        """Return a string representation of the TensorData object."""
        return f"<TensorData shape={self.shape} dtype={self.dtype}>"

    @staticmethod
    def loadop(op: LoadOps, shape: Tuple[int, ...], dtype: DType, arg=None) -> 'TensorData':
        """Create a TensorData object based on a specific loading operation.

        Supported operations include creating random data, constant data, or empty data.

        Args:

            op (LoadOps): The operation type (e.g., RAND, CONST, EMPTY).
            shape (Tuple[int, ...]): The shape of the tensor to be created.
            dtype (DType): The data type of the tensor.
            arg (Optional): Additional argument needed for some operations, like the value for CONST.

        Returns:
            TensorData: The resulting TensorData object.

        Raises:
            NotImplementedError: If the operation is not supported.
        """
        if op == LoadOps.RAND:
            return TensorData(np.random.default_rng(arg).random(size=shape, dtype=dtype.np))
        elif op == LoadOps.CONST:
            return TensorData(np.full(shape, arg, dtype=dtype.np))
        elif op == LoadOps.EMPTY:
            return TensorData(np.empty(shape, dtype=dtype.np))
        else:
            raise NotImplementedError(f"Operation {op} not implemented")

    def cast(self, dtype: DType, bitcast: bool = False) -> 'TensorData':
        """Cast the TensorData to a different data type."""
        if bitcast:
            return TensorData(self.data.view(dtype.np))
        else:
            return TensorData(self.data.astype(dtype.np))

    def exec(self, op, *srcs: 'TensorData'):
        """Execute a unary, binary, or ternary operation on the data."""
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
        """Perform reduction operations on the data."""
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

    # ------------------------------------------------------------------------------------------------------------------
    # movement operations
    def reshape(self, arg):
        """Reshape the data to a new shape."""
        return TensorData(self.data.reshape(arg))

    def expand(self, arg):
        """Expand the data to a new shape by broadcasting."""
        return TensorData(np.broadcast_to(self.data, arg))

    def shrink(self, arg):
        """Shrink the data by slicing it."""
        return TensorData(self.data[tuple(slice(p[0], p[1], None) for p in arg)])

    def permute(self, arg):
        """Permute the axes of the data."""
        return TensorData(self.data.transpose(arg))

    def pad(self, arg):
        """Pad the data with specified padding."""
        return TensorData(np.pad(self.data, arg))

    def stride(self, arg):
        """Apply striding to the  data."""
        return TensorData(self.data[tuple(slice(None, None, i) for i in arg)])
    
    def is_unrealized_contiguous_const(self):
        """Checks if the data is an unrealized contiguous constant."""
        return False
    
    def const(self, x) -> 'TensorData':
        """Returns a new TensorData with a constant value."""
        return TensorData(np.full_like(self.data, x))
