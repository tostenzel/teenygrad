"""This module contains functions for automatic differentiation in a neural network library.

It includes 'deepwalk' for graph traversal and 'backward' for backpropagation. 
These functions are part of the autograd mechanism, enabling gradient calculations for tensors.

"""
from __future__ import annotations


def deepwalk(tensor: 'Tensor'):
    """Traverse the computational graph of a tensor in a depth-first manner.
    
    This function recursively explores the graph from the given tensor, moving backwards
    to its parents. It is used to gather all the tensors involved in the computation that led
    to the given tensor.

    Args:
        tensor: The tensor from which to start the graph traversal.

    Returns:
        List[Tensor]: A list of tensors in the order they were visited.

    """
    def _deepwalk(node, visited, nodes):
        visited.add(node)
        if getattr(node, "_ctx", None):
            for i in node._ctx.parents:
                if i not in visited: 
                    _deepwalk(i, visited, nodes)
            nodes.append(node)
        return nodes
    return _deepwalk(tensor, set(), [])


def backward(tensor: 'Tensor') -> 'Tensor':
    """Performs the backward pass of automatic differentiation.
    
    This function computes the gradients of all tensors that were involved in the 
    computation of the given tensor, using the chain rule. It assumes that the tensor 
    for which it is called is a scalar (i.e., its shape is ()).

    Args:
        tensor: The tensor for which to compute the gradients. Must be a scalar.

    Returns:
        Tensor: The tensor with updated gradients.

    Raises:
        AssertionError: If the tensor is not a scalar.
    """
    from teenygrad.tensor import Tensor
    assert tensor.shape == tuple(), "backward can only be called for scalar tensors, but it has shape {})".format(tensor.shape)

    # Initialize the gradient of the tensor to 1. This serves as the base case for chain rule.
    tensor.grad = Tensor(1, requires_grad=False)

    # Reverse traverse the computational graph starting from the given tensor.
    for t0 in reversed(deepwalk(tensor)):
        assert t0.grad is not None
        # Compute the gradients for each tensor in the graph.
        grads = t0._ctx.backward(t0.grad.data)
        grads = [Tensor(g, requires_grad=False) if g is not None else None
                for g in ([grads] if len(t0._ctx.parents) == 1 else grads)]
        for t, g in zip(t0._ctx.parents, grads):
            if g is not None and t.requires_grad:
                # Accumulate gradients in the tensors that require them.
                assert g.shape == t.shape, "grad shape must match tensor shape, {} != {}".format(g.shape, t.shape)
                t.grad = g if t.grad is None else (t.grad + g)
        del t0._ctx
    return tensor
