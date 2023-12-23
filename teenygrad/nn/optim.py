from typing import List
from teenygrad.helpers import dedup
from teenygrad.tensor import Tensor

class Optimizer:
    """Base class for all optimizers."""
    def __init__(self, params: List[Tensor], lr: float):
        # Enable gradient calculation if not already set
        for x in params:
            if x.requires_grad is None: 
                x.requires_grad = True

        self.params = dedup([x for x in params if x.requires_grad])
        assert len(self.params) != 0, "optimizer must have at least one param"
        # Buffers: Tensors without gradient requirement
        self.buffers = dedup([x for x in params if not x.requires_grad])
        self.learning_rate = Tensor([lr], requires_grad=False).contiguous()

    def zero_grad(self):
        """Resets the gradient of all parameters to None."""
        for param in self.params:
            param.grad = None

    def realize(self, extra=None):
        """Realizes the computation for the optimizer's parameters and any additional tensors."""
        tensors_to_realize = self.params + self.buffers
        if extra is not None:
            tensors_to_realize += extra
        Tensor.corealize(tensors_to_realize)

class SGD(Optimizer):
    """Stochastic Gradient Descent (SGD) optimizer with optional momentum and weight decay."""
    def __init__(self, params: List[Tensor], lr=0.001, momentum=0, weight_decay=0.0, nesterov=False):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.buffer = [Tensor.zeros(*t.shape, requires_grad=False) for t in self.params] if self.momentum else []

    def step(self):
        """Performs a single optimization step (parameter update)."""
        for i, t in enumerate(self.params):
            assert t.grad is not None
            grad = t.grad.realize() + self.weight_decay * t.detach()
            if self.momentum:
                self.buffer[i].assign(self.momentum * self.buffer[i] + grad).realize()
                grad = (grad + self.momentum * self.buffer[i]) if self.nesterov else self.buffer[i]
            t.assign(t.detach() - grad * self.learning_rate)
        self.realize(self.buffer)


def AdamW(params: List[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
    """Variant of the Adam optimizer that includes weight decay."""
    return LAMB(params, lr, b1, b2, eps, wd, adam=True)


def Adam(params: List[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
    """Standard Adam optimizer without weight decay."""
    return LAMB(params, lr, b1, b2, eps, 0.0, adam=True)


class LAMB(Optimizer):
    """
    LAMB (Layer-wise Adaptive Moments optimizer for Batch training) optimizer.
    This optimizer adjusts the learning rate for each layer individually, 
    making it effective for training large models or datasets. It's a variant 
    of LARS applied to Adam/W.
    """
    def __init__(self, params: List[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-6, wd=0.0, adam=False):
        super().__init__(params, lr)
        self.beta1 = b1
        self.beta2 = b2
        self.epsilon = eps
        self.weight_decay = wd
        self.is_adam = adam
        self.time_step = Tensor([0], requires_grad=False).realize()
        self.moments = [Tensor.zeros(*t.shape, requires_grad=False) for t in self.params]
        self.velocities = [Tensor.zeros(*t.shape, requires_grad=False) for t in self.params]

    def step(self):
        """
        Performs a single optimization step (parameter update).
        """
        self.time_step.assign(self.time_step + 1).realize()
        for i, t in enumerate(self.params):
            assert t.grad is not None
            grad = t.grad.realize()
            self.moments[i].assign(self.beta1 * self.moments[i] + (1.0 - self.beta1) * grad).realize()
            self.velocities[i].assign(self.beta2 * self.velocities[i] + (1.0 - self.beta2) * (grad * grad)).realize()
            m_hat = self.moments[i] / (1.0 - self.beta1 ** self.time_step)
            v_hat = self.velocities[i] / (1.0 - self.beta2 ** self.time_step)
            update = (m_hat / (v_hat.sqrt() + self.epsilon)) + self.weight_decay * t.detach()
            if not self.is_adam:
                r1 = t.detach().square().sum().sqrt()
                r2 = update.square().sum().sqrt()
                trust_ratio = Tensor.where(r1 > 0, Tensor.where(r2 > 0, r1 / r2, 1.0), 1.0)
            else:
                trust_ratio = 1.0
            t.assign(t.detach() - self.learning_rate * trust_ratio * update)
        self.realize([self.time_step] + self.moments + self.velocities)