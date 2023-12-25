"""This module defines various operation types for tensor operations implemented in function.py, using namedtuples.

These namedtuples categorize operations in the program, facilitating a structured approach to implementing and managing
tensor operations.

Operation Types:
- UnaryOps: Operations taking a single operand (tensor). They apply specific mathematical or functional transformations
    to every element of a tensor.
- BinaryOps: Involve two operands (tensors), typically performing element-wise calculations.
- TernaryOps: More complex operations involving three operands (tensors), combining multiple binary operations.
- ReduceOps: Aggregate values across specified dimensions of a tensor, reducing its dimensions.
- MovementOps: Modify the structure or shape of tensor data without altering actual data values.
- LoadOps: Related to loading or initializing tensor data, creating tensors from various sources or with specific
    initialization patterns.

"""
from collections import namedtuple

UnaryOps = namedtuple('UnaryOps', ['NOOP', 'EXP2', 'LOG2', 'CAST', 'SIN', 'SQRT', 'RECIP', 'NEG'])
BinaryOps = namedtuple('BinaryOps', ['ADD', 'SUB', 'MUL', 'DIV', 'MAX', 'MOD', 'CMPLT'])
TernaryOps = namedtuple('TernaryOps', ['MULACC', 'WHERE'])
ReduceOps = namedtuple('ReduceOps', ['SUM', 'MAX'])
MovementOps = namedtuple('MovementOps', ['RESHAPE', 'PERMUTE', 'EXPAND', 'PAD', 'SHRINK', 'STRIDE'])
LoadOps = namedtuple('LoadOps', ['EMPTY', 'RAND', 'CONST', 'FROM', 'CUSTOM'])
