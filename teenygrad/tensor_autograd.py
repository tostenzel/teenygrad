"""This module contains functions for automatically computing the gradients from the cost/loss w.r.t. the model params.

Conceptual explanation: https://www.tobiasstenzel.com/blog/2023/dl-backprop/

"""
from __future__ import annotations

from typing import List, Set


def _collect_backward_graph(tensor: 'Tensor'):
    """Collects tensors involved in the computational graph of the given tensor in backward pass order.

    This function performs a depth-first search to traverse the graph from the provided tensor to its origins.
    It's crucial for backpropagation as it orders tensors in the sequence they were involved in computations.

    Example:
    Consider a computational graph for the function C(θ1, θ2) = θ1 * θ2 + tanh(θ1), with θ1 and θ2 as leaf nodes (inputs).
    This graph includes intermediate nodes z1 to z6. If this function is called on z6 (C(θ1, θ2)), it collects all tensors
    involved in the computation in the following order: [z6, z5, z4, z3, z2, z1, θ2, θ1]. This order represents the reversed
    sequence in which these tensors were computed during the forward pass.
    
    During the forward pass, for every resulting tensor, we first stored an instance of the forward function in tensor._ctx
    and a Tuple of input tensors from which the operation is derived ._ctx.parents.
    
    In backpropagation, gradients are calculated starting from the output (z6), representing the model cost/loss
    (loss.backward()) and propagated backward through intermediate nodes to the input parameters (θ1, θ2). The gradients
    are then used by the optimizer to update the parameter values before the next forward pass.

    Args:
        tensor: Starting point for graph traversal.

    Returns:
        List[Tensor]: Tensors in the order they were computed during the forward pass.
    """
    def depth_first_search(node: 'Tensor', visited: Set, nodes: List['Tensor']):
        visited.add(node)
        if getattr(node, "_ctx", None):
            # Visit each parent recursively. Parents are tensors that contributed to creating 'node' in forward pass.
            for parent in node._ctx.parents:
                if parent not in visited:
                    depth_first_search(parent, visited, nodes)

            # After visiting all parents, add the current node to the list.
            # This ensures that nodes are added after all their dependencies are added.
            nodes.append(node)
        return nodes

    # Begin traversal from the provided tensor.
    return reversed(depth_first_search(tensor, set(), []))


def backward(tensor: 'Tensor') -> 'Tensor':
    """Performs the backward pass of automatic differentiation.
    
    This function computes the gradients of all tensors that were involved in the 
    computation of the given tensor, using the chain rule. It assumes that the tensor 
    for which it is called is a scalar (i.e., its shape is ()). The function is to be
    applied to the model cost/loss.
    
    Consider the above toy example. The backward pass involves the following steps:
    
    1. Backpropagate from z6 (C) to z5:
       - Since z6 = z5, ∂z6/∂z5 = 1.
    
    2. Backpropagate from z5 to z3 and z4:
       - z5 is a sum of z3 and z4, so gradients are distributed: ∂z5/∂z3 = 1 and ∂z5/∂z4 = 1.
    
    3. Backpropagate from z4 to z1 (θ1):
       - With z4 = tanh(z1), ∂z4/∂z1 = 1 - tanh²(z1).
    
    4. Backpropagate from z3 to z1 and z2:
       - z3 = z1 * z2, thus ∂z3/∂z1 = z2 and ∂z3/∂z2 = z1.
    
    5. Accumulate gradients for z1 (θ1) and compute final gradient for z2 (θ2):
       - ∂C/∂θ1 = ∂z4/∂z1 + ∂z3/∂z1 = θ2 + (1 - tanh²(θ1)).
       - ∂C/∂θ2 = ∂z3/∂z2 = θ1.
    
    Args:
        tensor: The tensor for which to compute the gradients. Must be a scalar.
    
    Returns:
        Tensor: The tensor with updated gradients.
    
    Raises:
        AssertionError: If the tensor is not a scalar.

    """
    from teenygrad.tensor import Tensor

    assert tensor.shape == tuple(), "Backward only for scalar tensors."

    # Start with the gradient of the output tensor (child of all children from forward pass perspective) set to 1.
    tensor.grad = Tensor(1, requires_grad=False)

    # Traverse the backward graph
    for t0 in _collect_backward_graph(tensor):
        assert t0.grad is not None

        # Compute gradients for the current tensor.
        grads = t0._ctx.backward(t0.grad.data)
        grads = [Tensor(g, requires_grad=False) if g is not None else None
                 for g in ([grads] if len(t0._ctx.parents) == 1 else grads)]

        for parent, grad in zip(t0._ctx.parents, grads):
            if grad is not None and parent.requires_grad:
                """
                Updating gradients of parent tensors (future iterations) from forward pass perspective:

                For each tensor in the graph, its gradient is calculated based on the operation
                that created it. This gradient calculation uses the gradients of the 'child' tensor
                (the current tensor in the loop) to determine the gradients of its 'parent' tensors 
                (the tensors that contributed to its creation in the forward pass, next in the loop).

                If a parent tensor contributes to multiple operations (nodes in the graph), its 
                gradient is the sum of gradients from each of these contributions. This summation 
                is a key aspect of the chain rule and ensures that gradients
                accumulate correctly in cases of branched computations.
                """
                parent.grad = grad if parent.grad is None else (parent.grad + grad)

        # Clean up to prevent memory leaks.
        del t0._ctx

    """
    The process concludes when all tensors in the graph have had their gradients computed
    and updated. The final gradients on the input tensors (parameters) can now be used
    to perform optimization steps, such as gradient descent, to improve the model based
    on the computed gradients.
    """
    return tensor