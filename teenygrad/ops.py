"""This module defines various enums representing different types of operations.

These enums are utilized to categorize and manage various operations in the library, 
allowing for a structured and organized approach to implementing and handling these operations.

"""
from enum import Enum, auto


class UnaryOps(Enum):
    """Unary operations, taking a single operand."""
    NOOP = auto()
    EXP2 = auto()
    LOG2 = auto()
    CAST = auto()
    SIN = auto()
    SQRT = auto()
    RECIP = auto()
    NEG = auto()

class BinaryOps(Enum):
    """Binary operations, involving two operands."""
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MAX = auto()
    MOD = auto()
    CMPLT = auto()

class TernaryOps(Enum):
    """Ternary operations, involving three operands."""
    MULACC = auto()
    WHERE = auto()

class ReduceOps(Enum):
    """Reduction operations, aggregating values across dimensions."""
    SUM = auto()
    MAX = auto()

class MovementOps(Enum):
    """Operations that modify the structure or shape of data."""
    RESHAPE = auto()
    PERMUTE = auto()
    EXPAND = auto()
    PAD = auto()
    SHRINK = auto()
    STRIDE = auto()

class LoadOps(Enum):
    """Operations related to loading or initializing data."""
    EMPTY = auto()
    RAND = auto()
    CONST = auto()
    FROM = auto()
    CUSTOM = auto()
