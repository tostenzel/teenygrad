from __future__ import annotations
import math
from typing import Tuple, Optional, Union

from teenygrad.helpers import make_pair, flatten, dtypes, all_int, shape_int


# processing ops

def _pool(tensor: 'Tensor', k_:Tuple[shape_int, ...], stride:Union[Tuple[int, ...], int], dilation:Union[Tuple[int, ...], int]) -> 'Tensor':
    """Performs pooling operation on a tensor.

    This function performs pooling (either max, average, etc.) on a tensor based on the specified kernel size,
    stride, and dilation. It supports various complex scenarios like non-unit strides and dilations.

    Args:
        tensor (Tensor): The input tensor to be pooled.
        k_ (Tuple[shape_int, ...]): The size of the pooling kernel.
        stride (Union[Tuple[int, ...], int]): The stride of the pooling operation. Can be a single int or a tuple.
        dilation (Union[Tuple[int, ...], int]): The dilation of the pooling operation. Can be a single int or a tuple.

    Returns:
        Tensor: The pooled tensor.

    Raises:
        AssertionError: If the tensor's dimension is less than the kernel dimension.
        AssertionError: If the tensor or kernel dimensions are not all integers.
        AssertionError: If there's a mismatch in the dimensions of kernel, stride, and dilation.
    """

    # Ensure tensor dimensions are adequate for pooling
    assert len(tensor.shape) >= len(k_), f"Tensor's dimension {tensor.shape} is less than kernel's dimension {k_}"
    assert all_int(tensor.shape) and all_int(k_), "Tensor and kernel dimensions must be integers"

    # Prepare stride and dilation parameters
    s_, d_ = make_pair(stride, len(k_)), make_pair(dilation, len(k_))
    assert len(k_) == len(s_) and len(k_) == len(d_), "Mismatch in dimensions of kernel, stride, and dilation"

    # Prepare the tensor for pooling
    slc_prefix, prefix, i_ = [(0, x) for x in tensor.shape[:-len(k_)]], tensor.shape[:-len(k_)], tensor.shape[-len(k_):]
    if any(k > s for k,s in zip(k_, s_)) or any(d != 1 for d in d_):
        # Handle non-unit strides and dilations
        # Calculate output dimensions and expand tensor to fit the pooling operation
        o_ = [(i - d * (k-1) - 1)//s + 1 for i,d,k,s in zip(i_, d_, k_, s_)]
        e_ = [math.ceil(k*(i+d) / i) for k,i,d in zip(k_, i_, d_)]
        xup = tensor.reshape(*prefix, *flatten((1, i) for i in i_)).expand(*prefix, *flatten((e, i) for e, i in zip(e_, i_))).reshape(*prefix, *[e*i for e, i in zip(e_, i_)])

        # Slide by dilation and handle stride
        # Rearrange the tensor to prepare for pooling and permute for reduction
        xup = xup.slice(slc_prefix + [(0, k*(i+d)) for k, i, d in zip(k_, i_, d_)])
        xup = xup.reshape(*prefix, *flatten((k, i+d) for k, i, d in zip(k_, i_, d_)))
        xup = xup.slice(slc_prefix + flatten(((0, k), (0, o*s)) for k, o, s in zip(k_, o_, s_)))
        xup = xup.reshape(*prefix, *flatten((k, o, s) for k, o, s in zip(k_, o_, s_)))
        xup = xup.slice(slc_prefix + flatten(((0, k), (0, o), (0, 1)) for k, o in zip(k_, o_)))
        xup = xup.reshape(*prefix, *flatten((k, o) for k, o in zip(k_, o_)))
        return xup.permute(*range(len(prefix)), *[len(prefix)+i*2+1 for i in range(len(k_))], *[len(prefix)+i*2 for i in range(len(k_))])

    # Alternative implementation for simpler pooling scenarios
    o_ = [(i+(s-k))//s for i,s,k in zip(i_, s_, k_)]
    xup = tensor.slice(slc_prefix + [(0, o*s) for o, s in zip(o_, s_)])
    xup = xup.reshape(*prefix, *flatten(((o, s) for o, s in zip(o_, s_))))
    xup = xup.slice(slc_prefix + flatten(((0, o), (0, k)) for o, k in zip(o_, k_)))
    return xup.permute(*range(len(prefix)), *[len(prefix)+i*2 for i in range(len(k_))], *[len(prefix)+i*2+1 for i in range(len(k_))])


# NOTE: these work for more than 2D
def avg_pool2d(tensor: 'Tensor', kernel_size, stride, dilation):
    """Performs average pooling on a tensor.

    This function computes the average of elements within each kernel-sized window.
    It can handle tensors with more than two dimensions and supports various kernel sizes, strides, and dilations.

    Notes:
        - 'make_pair' ensures kernel_size, stride, and dilation are appropriately formatted for the pooling function.
        - The function first performs the pooling operation using `_pool`, then computes the mean over the last len(kernel_size) dimensions.
    """

    # Adjust stride and dilation if they are not provided
    stride = stride if stride is not None else kernel_size
    kernel_size = make_pair(kernel_size)
    
    # Perform pooling operation
    pooled_tensor = tensor._pool(kernel_size, stride, dilation)

    # Compute mean over each kernel-sized window
    # The axis argument specifies the dimensions to be reduced.
    # The range here computes the axes over which the mean should be taken.
    return pooled_tensor.mean(axis=tuple(range(0-len(kernel_size), 0)))


def max_pool2d(tensor: 'Tensor', kernel_size, stride, dilation):
    """Performs max pooling on a tensor.

    This function computes the maximum value of elements within each kernel-sized window.
    It can handle tensors with more than two dimensions and supports various kernel sizes, strides, and dilations.

    Notes:
        - 'make_pair' ensures kernel_size, stride, and dilation are appropriately formatted for the pooling function.
        - The function first performs the pooling operation using `_pool`, then computes the max over the last len(kernel_size) dimensions.
    """

    # Adjust stride and dilation if they are not provided
    stride = stride if stride is not None else kernel_size
    kernel_size = make_pair(kernel_size)
    
    # Perform pooling operation
    pooled_tensor = tensor._pool(kernel_size, stride, dilation)

    # Compute max over each kernel-sized window
    # The axis argument specifies the dimensions to be reduced.
    # The range here computes the axes over which the max should be taken.
    return pooled_tensor.max(axis=tuple(range(0-len(kernel_size), 0)))


#wino = int(getenv("WINO", "0"))
def conv2d(tensor: 'Tensor', weight: 'Tensor', bias: Optional['Tensor'] = None, groups=1, stride=1, dilation=1, padding=0) -> 'Tensor':
    """Performs a 2D convolution operation on the input tensor.

    This function convolves the input tensor with a set of learnable filters (weights),
    optionally adds a bias, and outputs the result. It supports adjustable stride, dilation,
    and padding parameters.

    Args:
        tensor (Tensor): The input tensor to be convolved.
        weight (Tensor): The convolution kernels (weights).
        bias (Optional[Tensor]): Optional bias to be added to the convolution result.
        groups (int): Number of blocked connections from input channels to output channels.
        stride (int): Stride of the convolution.
        dilation (int): Dilation factor of the kernel.
        padding (int or tuple/list): Padding added to both sides of the input.

    Returns:
        Tensor: The tensor resulting from the convolution operation.

    Notes:
        - Padding can be an integer or a tuple/list specifying the amount of padding.
        - The function adjusts the input tensor's shape using padding and then performs
          the convolution operation.
        - The convolution is implemented by pooling with padding, followed by reshaping and
          element-wise multiplication with the weight tensor, and a summation reduction.
        - If a bias tensor is provided, it is added to the result after convolution.
    """
    from teenygrad.tensor import Tensor

    # Extract batch size and input/output channels from tensor and weight shapes
    (bs, cin_), (cout, cin), HW = tensor.shape[:2], weight.shape[:2], weight.shape[2:]

    # Check compatibility of input tensor and weight shapes
    assert groups * cin == cin_ and len(tensor.shape) == len(weight.shape), \
        f"Input 'Tensor' shape {tensor.shape} does not match the shape of the weights {weight.shape}. ({groups * cin} vs. {cin_})"

    # Process padding parameter
    padding_ = [padding] * 2 * len(HW) if isinstance(padding, int) else \
               (padding if len(padding) == 2 * len(HW) else [p for p in padding for _ in range(2)][::-1])

    # Apply padding to the input tensor and perform pooling operation
    x = tensor.pad2d(padding_)._pool(HW, stride, dilation)  # (bs, groups * cin, oy, ox, H, W)

    # Calculate output tensor dimensions
    rcout, oyx = cout // groups, x.shape[2:-len(HW)]

    # Perform convolution operation
    if not all(x == 3 for x in HW) or stride != 1 or dilation != 1 or not Tensor.wino:
        # Reshape and expand input tensor for convolution
        x = x.reshape(bs, groups, cin, 1, *oyx, *HW).expand(bs, groups, cin, rcout, *oyx, *HW)
        x = x.permute(0, 1, 3, *[4 + i for i in range(len(oyx))], 2, *[4 + len(oyx) + i for i in range(len(HW))])

        # Perform convolution: element-wise multiplication and summation reduction
        ret = (x * weight.reshape(1, groups, rcout, *[1] * len(oyx), cin, *HW)).sum([-1 - i for i in range(1 + len(oyx))], keepdim=True).reshape(bs, cout, *oyx)

        # Add bias if provided
        return ret if bias is None else ret.add(bias.reshape(1, -1, *[1] * len(HW)))
    
    else:
        raise NotImplemented("Alternative or optimized convolution operations are not implemented.")


# ----------------------------------------------------------------------------------------------------------------------
# functional nn ops

def linear(tensor: 'Tensor', weight: 'Tensor', bias: Optional['Tensor'] = None):
    """Computes a linear transformation (y = xW + b) of the input tensor.

    If the weight tensor is one-dimensional, it performs element-wise multiplication
    between the input tensor and the weight tensor. Otherwise, it performs a dot product
    between the input tensor and the weight tensor.

    If a bias tensor is provided, it is added to the result of the multiplication/dot product.
    The bias tensor must match the dimensions required for broadcasting over the output tensor.
    """
    # Element-wise multiplication if weight is 1D, otherwise dot product
    x = tensor.mul(weight) if len(weight.shape) == 1 else tensor.dot(weight)
    
    # Add bias if provided
    return x.add(bias) if bias is not None else x


def binary_crossentropy(tensor: 'Tensor', y: 'Tensor') -> 'Tensor':
    """Computes the binary cross-entropy loss between the predicted tensor and the target tensor.

    The binary cross-entropy loss is a measure used in binary classification tasks.
    This function calculates the loss for each pair of predicted and target values,
    then takes the mean of these losses.
    """
    # Calculate loss for each element and then take the mean
    return (-y * tensor.log() - (1 - y) * (1 - tensor).log()).mean()


def binary_crossentropy_logits(tensor: 'Tensor', y: 'Tensor') -> 'Tensor':
    """Computes the binary cross-entropy loss between the predicted logits and the target tensor.

    This function is similar to binary_crossentropy but takes logits as input instead of
    probabilities. Logits are the raw outputs of the last linear layer and have not been
    passed through a sigmoid function.
    """
    # Compute loss using logits
    return (tensor.maximum(0) - y * tensor + (1 + tensor.abs().__neg__().exp()).log()).mean()


def sparse_categorical_crossentropy(tensor: 'Tensor', Y, ignore_index=-1) -> 'Tensor':
    """Computes the sparse categorical cross-entropy loss for classification tasks.

    This function is used when the classes are mutually exclusive (e.g., each sample belongs
    to exactly one class). The input tensor is expected to be logits, and the target Y should
    contain integer class indices. The function supports an 'ignore_index' to exclude certain
    targets from loss computation.

    The function converts the class indices into a one-hot encoded format internally and then
    computes the cross-entropy loss.
    """
    # Create a mask to ignore certain targets based on the ignore_index
    loss_mask = Y != ignore_index

    # Convert class indices to one-hot encoded format
    y_counter = tensor.arange(tensor.shape[-1], dtype=dtypes.int32, requires_grad=False).unsqueeze(0).expand(Y.numel(), tensor.shape[-1])
    y = ((y_counter == Y.flatten().reshape(-1, 1)).where(-1.0, 0) * loss_mask.reshape(-1, 1)).reshape(*Y.shape, tensor.shape[-1])

    # Compute cross-entropy loss
    return tensor.log_softmax().mul(y).sum() / loss_mask.sum()
