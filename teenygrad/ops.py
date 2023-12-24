"""This module defines various Enums representing different types of tensor operations implemented in function.py.

These Enums categorize operations in the library, facilitating a structured approach to implementing and managing tensor
operations.
Each enum type corresponds to a specific category of tensor operations, ranging from unary to complex movement
operations.

"""
from enum import Enum, auto


class UnaryOps(Enum):
    """Enum for unary operations, which are operations taking a single operand (tensor).

    These operations apply a specific mathematical or functional transformation to every element of a tensor.
    """
    NOOP = auto()
    EXP2 = auto()
    LOG2 = auto()
    CAST = auto()
    SIN = auto()
    SQRT = auto()
    RECIP = auto()
    NEG = auto()


class BinaryOps(Enum):
    """Enum for binary operations, which involve two operands (tensors).

    These operations typically perform element-wise calculations between two tensors.
    """
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MAX = auto()
    MOD = auto()
    CMPLT = auto()


class TernaryOps(Enum):
    """Enum for ternary operations, involving three operands (tensors).

    These are more complex operations that usually combine aspects of multiple binary operations.
    """
    MULACC = auto()
    WHERE = auto()


class ReduceOps(Enum):
    """Enum for reduction operations, which aggregate values across specified dimensions of a tensor.

    These operations can reduce a tensor to fewer dimensions based on the operation.
    """
    SUM = auto()
    MAX = auto()


class MovementOps(Enum):
    """Enum for operations that modify the structure or shape of tensor data without altering the actual data values."""
    RESHAPE = auto()
    PERMUTE = auto()
    EXPAND = auto()
    PAD = auto()
    SHRINK = auto()
    STRIDE = auto()


class LoadOps(Enum):
    """Enum for operations related to loading or initializing tensor data.

    These operations are typically used for creating tensors from various sources or with specific initialization patterns.
    """
    EMPTY = auto()
    RAND = auto()
    CONST = auto()
    FROM = auto()
    CUSTOM = auto()
