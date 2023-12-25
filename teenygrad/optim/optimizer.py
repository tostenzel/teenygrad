"""Optimization algorithms for gradient-based learning methods.

This module provides a set of optimizer classes for updating the parameters of a neural network model
in response to the gradients computed during backpropagation.

"""
from typing import List
from teenygrad.helpers import dedup
from teenygrad.tensor import Tensor


class Optimizer:
    """Base class for all optimizers.

    Optimizers are algorithms or methods used to change the attributes of the neural network,
    such as weights and learning rate, in order to reduce the losses.

    Attributes:
        params (List[Tensor]): List of parameters to be optimized.
        learning_rate (Tensor): Learning rate for the optimizer.
        buffers (List[Tensor]): Tensors without gradient requirement, typically used for internal states of the optimizer.
    """
    def __init__(self, params: List[Tensor], lr: float):
        # Ensure all parameters are set to require gradients if not already specified
        for x in params:
            if x.requires_grad is None: 
                x.requires_grad = True

        # Deduplicate and filter out tensors that require gradients
        self.params = dedup([x for x in params if x.requires_grad])
        assert len(self.params) != 0, "optimizer must have at least one param"

        # Deduplicate and store buffers
        self.buffers = dedup([x for x in params if not x.requires_grad])

        # Set the learning rate as a tensor
        self.learning_rate = Tensor([lr], requires_grad=False)

    def zero_grad(self):
        """Resets the gradients of all optimized parameters to None.
        
        This should be called before each optimization step.
        """
        for param in self.params:
            param.grad = None


class SGD(Optimizer):
    """Stochastic Gradient Descent (SGD) optimizer.

    This optimizer updates model parameters by moving them in the opposite direction of their gradients,
    scaled by the learning rate. Optional momentum and weight decay can be applied.

    The update rule for a parameter \( p \) with gradient \( g \) is:
    \( p = p - lr \times (g + weight\_decay \times p) \)

    If momentum is used, the update rule becomes:
    \( v = momentum \times v + g \)
    \( p = p - lr \times v \)

    Attributes:
        momentum: Momentum factor.
        weight_decay: Weight decay.
        nesterov: Whether to use Nesterov momentum.
        buffer: Buffer storing the momentum values.

    """
    def __init__(self, params: List[Tensor], lr: float=0.001, momentum: int=0, weight_decay: float=0.0, nesterov: bool=False):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov

        # Initialize the momentum buffer if momentum is used
        self.buffer = [Tensor.zeros(*t.shape, requires_grad=False) for t in self.params] if self.momentum else []

    def step(self):
        """Performs a single optimization step.

        Updates the parameters based on the gradients, learning rate, momentum, and weight decay.
        """
        for i, t in enumerate(self.params):
            assert t.grad is not None
            grad = t.grad + self.weight_decay * t.detach()
            if self.momentum:
                self.buffer[i].assign(self.momentum * self.buffer[i] + grad)
                grad = (grad + self.momentum * self.buffer[i]) if self.nesterov else self.buffer[i]
            t.assign(t.detach() - grad * self.learning_rate)


def AdamW(params: List[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
    """Variant of the Adam optimizer that includes weight decay."""
    return LAMB(params, lr, b1, b2, eps, wd, adam=True)


def Adam(params: List[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
    """Standard Adam optimizer without weight decay."""
    return LAMB(params, lr, b1, b2, eps, 0.0, adam=True)


class LAMB(Optimizer):
    """LAMB (Layer-wise Adaptive Moments optimizer for Batch training) optimizer.

    This optimizer combines the LARS (Layer-wise Adaptive Rate Scaling) algorithm with Adam or AdamW.
    It adjusts the learning rate for each layer individually, making it effective for training large models.

    The update rule is:
    \( m = \beta_1 \times m + (1 - \beta_1) \times g \)
    \( v = \beta_2 \times v + (1 - \beta_2) \times g^2 \)
    \( m\_hat = m / (1 - \beta_1^t) \)
    \( v\_hat = v / (1 - \beta_2^t) \)
    \( update = (m\_hat / (\sqrt{v\_hat} + \epsilon)) + wd \times p \)
    \( trust\_ratio = r1 / r2 \) (if not Adam)
    \( p = p - lr \times trust\_ratio \times update \)

    Attributes:
        beta1: Exponential decay rate for the first moment estimates.
        beta2: Exponential decay rate for the second moment estimates.
        epsilon: Term added to the denominator to improve numerical stability.
        weight_decay: Weight decay coefficient.
        is_adam: Flag to use Adam instead of LAMB algorithm.
        time_step: Counter for the number of steps taken.
        moments: First moment vectors (moving averages of the gradients).
        velocities: Second moment vectors (moving averages of the squared gradients).

    """
    def __init__(self, params: List[Tensor], lr: float=0.001, b1: float=0.9, b2: float=0.999, eps: float=1e-6, wd: float=0.0, adam: bool=False):
        super().__init__(params, lr)
        self.beta1 = b1
        self.beta2 = b2
        self.epsilon = eps
        self.weight_decay = wd
        self.is_adam = adam

        # Initialize time step, moments, and velocities for the optimizer
        self.time_step = Tensor([0], requires_grad=False)
        self.moments = [Tensor.zeros(*t.shape, requires_grad=False) for t in self.params]
        self.velocities = [Tensor.zeros(*t.shape, requires_grad=False) for t in self.params]

    def step(self):
        """Performs a single optimization step.

        Updates the parameters based on the LAMB or Adam algorithm, depending on the `is_adam` flag.
        Applies layer-wise adaptive learning rates.

        """
        self.time_step.assign(self.time_step + 1)
        for i, t in enumerate(self.params):
            assert t.grad is not None
            # Compute moments and velocities
            grad = t.grad
            self.moments[i].assign(self.beta1 * self.moments[i] + (1.0 - self.beta1) * grad)
            self.velocities[i].assign(self.beta2 * self.velocities[i] + (1.0 - self.beta2) * (grad * grad))

            # Compute bias-corrected moments
            m_hat = self.moments[i] / (1.0 - self.beta1 ** self.time_step)
            v_hat = self.velocities[i] / (1.0 - self.beta2 ** self.time_step)

            # Compute the update with weight decay
            update = (m_hat / (v_hat.sqrt() + self.epsilon)) + self.weight_decay * t.detach()

            # Compute the trust ratio if not using Adam
            if not self.is_adam:
                r1 = t.detach().square().sum().sqrt()
                r2 = update.square().sum().sqrt()
                trust_ratio = Tensor.where(r1 > 0, Tensor.where(r2 > 0, r1 / r2, 1.0), 1.0)
            else:
                trust_ratio = 1.0

            # Update the parameter
            t.assign(t.detach() - self.learning_rate * trust_ratio * update)
