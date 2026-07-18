"""
Build a Trainable CNN from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - argmax_rows
import numpy as np

def argmax_rows(x):
    return np.argmax(x, axis=1)

# Step 2 - row_max
import numpy as np

def row_max(matrix):
    return np.max(matrix, axis=1, keepdims=True)

# Step 3 - row_sum
import numpy as np

def row_sum(matrix):
    return np.sum(matrix, axis=1, keepdims=True)

# Step 4 - exp_shifted
import numpy as np

def exp_shifted(logits):
    return np.exp(logits - row_max(logits))

# Step 5 - stable_softmax
import numpy as np

def stable_softmax(logits):
    exp_vals = exp_shifted(logits)
    return exp_vals / row_sum(exp_vals)

# Step 6 - one_hot
import numpy as np

def one_hot(labels, num_classes):
    result = np.zeros((len(labels), num_classes), dtype=float)
    result[np.arange(len(labels)), labels] = 1.0
    return result

# Step 7 - gather_true_class_probs
import numpy as np

def gather_true_class_probs(probs, labels):
    return probs[np.arange(len(labels)), labels]

# Step 8 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(probs, labels, eps=1e-12):
    true_probs = gather_true_class_probs(probs, labels)
    true_probs = np.clip(true_probs, eps, 1.0)
    return -np.mean(np.log(true_probs))

# Step 9 - accuracy
import numpy as np

def accuracy(predictions, labels):
    preds = argmax_rows(predictions)
    return np.mean(preds == labels)

# Step 10 - he_std
import numpy as np

def he_std(fan_in):
    return float(np.sqrt(2.0 / fan_in))

# Step 11 - he_init
import numpy as np

def he_init(shape, fan_in, seed):
    np.random.seed(seed)
    return np.random.normal(loc=0.0, scale=he_std(fan_in), size=shape)

# Step 12 - init_zero_bias
import numpy as np

def init_zero_bias(length):
    return np.zeros(length, dtype=np.float64)

# Step 13 - pad_2d
import numpy as np

def pad_2d(images, pad):
    return np.pad(
        images,
        pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode='constant'
    )

# Step 14 - output_spatial_size
def output_spatial_size(input_size, kernel_size, stride, padding):
    return (input_size - kernel_size + 2 * padding) // stride + 1

# Step 15 - im2col
import numpy as np

def im2col(images, kernel_h, kernel_w, stride, padding):
    images = pad_2d(images, padding)

    N, C, H, W = images.shape
    out_h = output_spatial_size(H - 2 * padding, kernel_h, stride, padding)
    out_w = output_spatial_size(W - 2 * padding, kernel_w, stride, padding)

    cols = np.empty((N * out_h * out_w, C * kernel_h * kernel_w), dtype=images.dtype)

    row = 0
    for n in range(N):
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                w_start = j * stride
                patch = images[n, :, h_start:h_start + kernel_h, w_start:w_start + kernel_w]
                cols[row] = patch.reshape(-1)
                row += 1

    return cols

# Step 16 - col2im
import numpy as np

def col2im(cols, input_shape, kernel_h, kernel_w, stride=1, padding=0):
    N, C, H, W = input_shape

    # Compute output spatial dimensions
    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)

    # Reshape columns back into patches
    cols = cols.reshape(N, out_h, out_w, C, kernel_h, kernel_w)
    cols = cols.transpose(0, 3, 4, 5, 1, 2)

    # Allocate padded image
    img = np.zeros(
        (N, C, H + 2 * padding, W + 2 * padding),
        dtype=cols.dtype
    )

    # Scatter-add patches back into the image
    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x in range(kernel_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += cols[:, :, y, x, :, :]

    # Remove padding if present
    if padding > 0:
        return img[:, :, padding:-padding, padding:-padding]
    return img

# Step 17 - conv2d_forward
import numpy as np

def conv2d_forward(x, W, b, stride=1, padding=0):
    """
    Perform 2D convolution using the im2col trick.

    Args:
        x: Input tensor of shape (N, C_in, H, W)
        W: Weights of shape (C_out, C_in, kernel_h, kernel_w)
        b: Bias of shape (C_out,)
        stride: Stride
        padding: Padding

    Returns:
        out: Output tensor of shape (N, C_out, out_h, out_w)
        cache: Values needed for backward pass
    """
    N, C_in, H, W_in = x.shape
    C_out, _, kernel_h, kernel_w = W.shape

    # Output spatial dimensions
    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W_in, kernel_w, stride, padding)

    # Convert input to columns
    cols = im2col(x, kernel_h, kernel_w, stride, padding)

    # Flatten filters
    W_col = W.reshape(C_out, -1)

    # Convolution
    out = cols @ W_col.T
    out += b

    # Reshape to (N, C_out, out_h, out_w)
    out = out.reshape(N, out_h, out_w, C_out)
    out = out.transpose(0, 3, 1, 2)

    # Cache for backward pass
    cache = {
        "x_shape": x.shape,
        "weights": W,
        "cols": cols,
        "stride": stride,
        "padding": padding,
        "kernel_h": kernel_h,
        "kernel_w": kernel_w,
    }

    return out, cache

# Step 18 - conv2d_grad_input
import numpy as np

def conv2d_grad_input(d_out, cache):
    """
    Backpropagate through the convolution with respect to the input.

    Args:
        d_out: Upstream gradient of shape (N, C_out, out_h, out_w)
        cache: Dictionary returned by conv2d_forward

    Returns:
        dx: Gradient with respect to the input x
    """
    x_shape = cache["x_shape"]
    W = cache["weights"]
    stride = cache["stride"]
    padding = cache["padding"]
    kernel_h = cache["kernel_h"]
    kernel_w = cache["kernel_w"]

    N, C_out, out_h, out_w = d_out.shape

    # (N*out_h*out_w, C_out)
    d_out_cols = d_out.transpose(0, 2, 3, 1).reshape(-1, C_out)

    # (C_out, C_in*kernel_h*kernel_w)
    W_cols = W.reshape(C_out, -1)

    # Gradient with respect to im2col output
    d_cols = d_out_cols @ W_cols

    # Fold columns back into image
    dx = col2im(
        d_cols,
        x_shape,
        kernel_h,
        kernel_w,
        stride,
        padding
    )

    return dx

# Step 19 - conv2d_grad_weights
import numpy as np

def conv2d_grad_weights(d_out, cache):
    """
    Compute gradient with respect to convolution weights.

    Args:
        d_out: Upstream gradient of shape (N, C_out, out_h, out_w)
        cache: Dictionary returned by conv2d_forward

    Returns:
        dW: Gradient with respect to weights, same shape as cache["weights"]
    """
    cols = cache["cols"]
    W = cache["weights"]
    kernel_h = cache["kernel_h"]
    kernel_w = cache["kernel_w"]

    C_out, C_in, _, _ = W.shape

    # (N*out_h*out_w, C_out)
    d_out_cols = d_out.transpose(0, 2, 3, 1).reshape(-1, C_out)

    # (C_out, C_in*kernel_h*kernel_w)
    dW = d_out_cols.T @ cols

    # Reshape back to original filter layout
    dW = dW.reshape(C_out, C_in, kernel_h, kernel_w)

    return dW

# Step 20 - conv2d_grad_bias
import numpy as np

def conv2d_grad_bias(d_out):
    """
    Compute gradient with respect to convolution bias.

    Args:
        d_out: Upstream gradient of shape (N, C_out, out_h, out_w)

    Returns:
        db: Gradient with respect to bias, shape (C_out,)
    """
    return np.sum(d_out, axis=(0, 2, 3))

# Step 21 - conv2d_backward
import numpy as np

def conv2d_backward(d_out, cache):
    """
    Compute gradients of a 2D convolution layer.

    Args:
        d_out: Upstream gradient of shape (N, C_out, out_h, out_w)
        cache: Cache returned by conv2d_forward

    Returns:
        (dx, dW, db)
    """
    dx = conv2d_grad_input(d_out, cache)
    dW = conv2d_grad_weights(d_out, cache)
    db = conv2d_grad_bias(d_out)

    return dx, dW, db

# Step 22 - maxpool2d_forward
import numpy as np

def maxpool2d_forward(x, kernel=2, stride=2):
    """
    Forward pass for 2D max pooling.

    Args:
        x: Input of shape (N, C, H, W)
        kernel: Pooling kernel size
        stride: Pooling stride

    Returns:
        out: Output of shape (N, C, out_h, out_w)
        cache: Dictionary containing values needed for backward pass
    """
    N, C, H, W = x.shape

    out_h = output_spatial_size(H, kernel, stride, 0)
    out_w = output_spatial_size(W, kernel, stride, 0)

    out = np.zeros((N, C, out_h, out_w), dtype=x.dtype)
    argmax = np.zeros((N, C, out_h, out_w), dtype=np.int64)

    for i in range(out_h):
        h_start = i * stride
        h_end = h_start + kernel

        for j in range(out_w):
            w_start = j * stride
            w_end = w_start + kernel

            window = x[:, :, h_start:h_end, w_start:w_end]
            window_flat = window.reshape(N, C, -1)

            argmax[:, :, i, j] = np.argmax(window_flat, axis=2)
            out[:, :, i, j] = np.max(window_flat, axis=2)

    cache = {
        "x_shape": x.shape,
        "argmax": argmax,
        "kernel": kernel,
        "stride": stride,
    }

    return out, cache

# Step 23 - scatter_grad_window
import numpy as np

def scatter_grad_window(grad, argmax_index, kernel):
    """
    Place the upstream gradient at the argmax position within a
    (kernel x kernel) window.

    Args:
        grad: Scalar upstream gradient.
        argmax_index: Flat index (row-major) of the max element.
        kernel: Pooling kernel size.

    Returns:
        A (kernel, kernel) array with grad at the argmax position.
    """
    window = np.zeros((kernel, kernel), dtype=float)

    row = argmax_index // kernel
    col = argmax_index % kernel

    window[row, col] = grad

    return window

# Step 24 - maxpool2d_backward
import numpy as np

def maxpool2d_backward(d_out, cache):
    """
    Backward pass for 2D max pooling.

    Args:
        d_out: Upstream gradient of shape (N, C, out_h, out_w)
        cache: Dictionary from maxpool2d_forward

    Returns:
        dx: Gradient with respect to the input, shape (N, C, H, W)
    """
    x_shape = cache["x_shape"]
    argmax = cache["argmax"]
    kernel = cache["kernel"]
    stride = cache["stride"]

    N, C, H, W = x_shape
    _, _, out_h, out_w = d_out.shape

    dx = np.zeros(x_shape, dtype=d_out.dtype)

    for n in range(N):
        for c in range(C):
            for i in range(out_h):
                h_start = i * stride
                h_end = h_start + kernel

                for j in range(out_w):
                    w_start = j * stride
                    w_end = w_start + kernel

                    grad_window = scatter_grad_window(
                        d_out[n, c, i, j],
                        argmax[n, c, i, j],
                        kernel
                    )

                    dx[n, c, h_start:h_end, w_start:w_end] += grad_window

    return dx

# Step 25 - relu_forward
import numpy as np

def relu_forward(x):
    """
    Forward pass for ReLU activation.

    Args:
        x: Input array of any shape.

    Returns:
        out: ReLU output with same shape as x.
        cache: Dictionary containing the original input.
    """
    out = np.maximum(0, x)

    cache = {
        "x": x
    }

    return out, cache

# Step 26 - relu_backward
import numpy as np

def relu_backward(d_out, cache):
    """
    Backward pass for ReLU.

    Args:
        d_out: Upstream gradient of any shape.
        cache: Dictionary from relu_forward containing 'x'.

    Returns:
        dx: Gradient with respect to the input.
    """
    x = cache["x"]

    dx = d_out * (x > 0)

    return dx

# Step 27 - flatten_forward
import numpy as np

def flatten_forward(x):
    """
    Forward pass for flatten layer.

    Args:
        x: Input of shape (N, C, H, W)

    Returns:
        out: Flattened output of shape (N, C*H*W)
        cache: Dictionary containing the original input shape.
    """
    out = x.reshape(x.shape[0], -1)

    cache = {
        "x_shape": x.shape
    }

    return out, cache

# Step 28 - flatten_backward
import numpy as np

def flatten_backward(d_out, cache):
    """
    Backward pass for flatten layer.

    Args:
        d_out: Upstream gradient of shape (N, C*H*W)
        cache: Dictionary from flatten_forward containing 'x_shape'

    Returns:
        dx: Gradient reshaped back to the original input shape.
    """
    x_shape = cache["x_shape"]
    dx = d_out.reshape(x_shape)
    return dx

# Step 29 - linear_forward
import numpy as np

def linear_forward(x, weights, bias):
    """
    Forward pass for a fully-connected (linear) layer.

    Args:
        x: Input of shape (N, D_in)
        weights: Weight matrix of shape (D_in, D_out)
        bias: Bias vector of shape (D_out,)

    Returns:
        out: Output of shape (N, D_out)
        cache: Dictionary containing x and weights.
    """
    out = x @ weights + bias

    cache = {
        "x": x,
        "weights": weights
    }

    return out, cache

# Step 30 - linear_grad_input
import numpy as np

def linear_grad_input(d_out, cache):
    """
    Compute gradient of the loss with respect to the input of a linear layer.

    Args:
        d_out: Upstream gradient of shape (N, D_out)
        cache: Dictionary from linear_forward containing 'weights'

    Returns:
        dx: Gradient with respect to the input, shape (N, D_in)
    """
    weights = cache["weights"]
    dx = d_out @ weights.T
    return dx

# Step 31 - linear_grad_weights
import numpy as np

def linear_grad_weights(x, d_out):
    """
    Compute the gradient of the loss with respect to the weight matrix.

    Args:
        x: Input to the linear layer of shape (N, D_in)
        d_out: Upstream gradient of shape (N, D_out)

    Returns:
        dW: Gradient with respect to weights of shape (D_in, D_out)
    """
    dW = x.T @ d_out
    return dW

# Step 32 - linear_grad_bias
import numpy as np

def linear_grad_bias(d_out):
    """
    Compute the gradient with respect to the bias vector.

    Args:
        d_out: Upstream gradient of shape (N, D_out)

    Returns:
        db: Gradient with respect to bias, shape (D_out,)
    """
    return np.sum(d_out, axis=0)

# Step 33 - linear_backward
import numpy as np

def linear_backward(d_out, cache):
    """
    Backward pass for a fully-connected (linear) layer.

    Args:
        d_out: Upstream gradient of shape (N, D_out)
        cache: Dictionary from linear_forward containing 'x', 'weights', and optionally 'bias'

    Returns:
        dx: Gradient with respect to input, shape (N, D_in)
        dW: Gradient with respect to weights, shape (D_in, D_out)
        db: Gradient with respect to bias, shape (D_out,)
    """
    x = cache["x"]

    dx = linear_grad_input(d_out, cache)
    dW = linear_grad_weights(x, d_out)
    db = linear_grad_bias(d_out)

    return dx, dW, db

# Step 34 - softmax_cross_entropy_forward
import numpy as np

def softmax_cross_entropy_forward(logits, y):
    probs = stable_softmax(logits)
    loss = cross_entropy_loss(probs, y)

    loss = float(loss)

    # Normalize -0.0 to +0.0
    if loss == 0.0:
        loss = 0.0

    return loss

# Step 35 - softmax_cross_entropy_backward
import numpy as np

def softmax_cross_entropy_backward(logits, y):
    """
    Backward pass for fused softmax + cross-entropy loss.

    Args:
        logits: Array of shape (N, C)
        y: Integer labels of shape (N,)

    Returns:
        dlogits: Gradient with respect to logits, shape (N, C)
    """
    N = logits.shape[0]

    probs = stable_softmax(logits)
    targets = one_hot(y, logits.shape[1])

    dlogits = (probs - targets) / N

    return dlogits

# Step 36 - sgd_step
import numpy as np

def sgd_step(param, grad, learning_rate):
    """
    Perform one SGD update.

    Args:
        param: Parameter array.
        grad: Gradient array (same shape as param).
        learning_rate: Learning rate.

    Returns:
        Updated parameter array.
    """
    return param - learning_rate * grad

# Step 37 - adam_update_m
import numpy as np

def adam_update_m(m, grad, beta_one):
    """
    Update Adam's first moment estimate.

    Args:
        m: Previous first moment.
        grad: Current gradient.
        beta_one: Exponential decay rate for the first moment.

    Returns:
        Updated first moment.
    """
    return beta_one * m + (1.0 - beta_one) * grad

# Step 38 - adam_update_v
import numpy as np

def adam_update_v(v, grad, beta_two):
    """
    Update Adam's second moment estimate.

    Args:
        v: Previous second moment.
        grad: Current gradient.
        beta_two: Exponential decay rate for the second moment.

    Returns:
        Updated second moment.
    """
    return beta_two * v + (1.0 - beta_two) * (grad ** 2)

# Step 39 - adam_bias_correct
import numpy as np

def adam_bias_correct(moment, beta, t):
    """
    Bias-correct an Adam moment estimate.

    Args:
        moment: First or second moment estimate.
        beta: Decay rate (beta_one or beta_two).
        t: Current timestep (starting from 1).

    Returns:
        Bias-corrected moment.
    """
    return moment / (1.0 - beta ** t)

# Step 40 - adam_param_step
import numpy as np

def adam_param_step(param, m_hat, v_hat, lr, eps):
    """
    Apply one Adam parameter update using bias-corrected moments.

    Args:
        param: Parameter array.
        m_hat: Bias-corrected first moment.
        v_hat: Bias-corrected second moment.
        lr: Learning rate.
        eps: Small constant for numerical stability.

    Returns:
        Updated parameter array.
    """
    return param - lr * m_hat / (np.sqrt(v_hat) + eps)

# Step 41 - adam_step
import numpy as np

def adam_step(param, grad, m, v, t, lr, beta_one, beta_two, eps):
    """
    Perform one complete Adam optimizer update.

    Args:
        param: Parameter array.
        grad: Gradient array.
        m: Running first moment estimate.
        v: Running second moment estimate.
        t: Current timestep (starting from 1).
        lr: Learning rate.
        beta_one: First moment decay rate.
        beta_two: Second moment decay rate.
        eps: Small constant for numerical stability.

    Returns:
        (new_param, new_m, new_v)
    """
    # Update running moments
    new_m = adam_update_m(m, grad, beta_one)
    new_v = adam_update_v(v, grad, beta_two)

    # Bias correction
    m_hat = adam_bias_correct(new_m, beta_one, t)
    v_hat = adam_bias_correct(new_v, beta_two, t)

    # Parameter update
    new_param = adam_param_step(param, m_hat, v_hat, lr, eps)

    return new_param, new_m, new_v

# Step 42 - init_conv_layer
import numpy as np

def init_conv_layer(out_channels, in_channels, kernel_size, seed=None):
    """
    Initialize a convolutional layer using He initialization.

    Args:
        out_channels: Number of output channels.
        in_channels: Number of input channels.
        kernel_size: Size of the (square) convolution kernel.
        seed: Optional random seed.

    Returns:
        Dictionary with keys:
            'W': Convolution weights of shape
                 (out_channels, in_channels, kernel_size, kernel_size)
            'b': Bias vector of shape (out_channels,)
    """
    fan_in = in_channels * kernel_size * kernel_size

    W = he_init(
        (out_channels, in_channels, kernel_size, kernel_size),
        fan_in,
        seed
    )

    b = init_zero_bias(out_channels)

    return {
        "W": W,
        "b": b
    }

# Step 43 - init_linear_layer
import numpy as np

def init_linear_layer(in_features, out_features, seed=None):
    """
    Initialize a fully-connected layer using He initialization.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        seed: Optional random seed.

    Returns:
        Dictionary containing:
            'W': Weight matrix of shape (in_features, out_features)
            'b': Bias vector of shape (out_features,)
    """
    W = he_init(
        (in_features, out_features),
        in_features,
        seed
    )

    b = init_zero_bias(out_features)

    return {
        "W": W,
        "b": b
    }

# Step 44 - init_lenet
import numpy as np

def init_lenet(in_channels, num_classes, seed=None):
    """
    Initialize a LeNet-style network.

    Architecture:
        Conv1: 6 filters, 5x5 kernel
        Conv2: 16 filters, 5x5 kernel
        FC1: 120 units
        FC2: num_classes units

    Args:
        in_channels: Number of input image channels.
        num_classes: Number of output classes.
        seed: Base random seed.

    Returns:
        Dictionary containing all layer parameters.
    """
    if seed is None:
        seeds = [None, None, None, None]
    else:
        seeds = [seed, seed + 1, seed + 2, seed + 3]

    params = {
        "conv1": init_conv_layer(
            out_channels=6,
            in_channels=in_channels,
            kernel_size=5,
            seed=seeds[0]
        ),

        "conv2": init_conv_layer(
            out_channels=16,
            in_channels=6,
            kernel_size=5,
            seed=seeds[1]
        ),

        "fc1": init_linear_layer(
            in_features=16 * 4 * 4,
            out_features=120,
            seed=seeds[2]
        ),

        "fc2": init_linear_layer(
            in_features=120,
            out_features=num_classes,
            seed=seeds[3]
        )
    }

    return params

# Step 45 - forward_conv_block
import numpy as np

def forward_conv_block(x, W, b, pool_size, stride, pad):
    """
    Forward pass for a LeNet-style convolution block:
    Conv -> ReLU -> MaxPool

    Args:
        x: Input tensor of shape (N, C, H, W)
        W: Convolution weights
        b: Convolution bias
        pool_size: Max pooling kernel size
        stride: Convolution stride
        pad: Convolution padding

    Returns:
        out: Final pooled activation
        cache: Dictionary containing caches for backward pass
    """
    conv_out, conv_cache = conv2d_forward(
        x,
        W,
        b,
        stride=stride,
        padding=pad
    )

    relu_out, relu_cache = relu_forward(conv_out)

    out, pool_cache = maxpool2d_forward(
        relu_out,
        kernel=pool_size,
        stride=pool_size
    )

    cache = {
        "conv_cache": conv_cache,
        "relu_cache": relu_cache,
        "pool_cache": pool_cache
    }

    return out, cache

# Step 46 - forward_classifier_block
import numpy as np

def forward_classifier_block(x, fc1, fc2):
    """
    Forward pass for the LeNet classifier head:
    Flatten -> Linear -> ReLU -> Linear

    Args:
        x: Input feature map (N, C, H, W)
        fc1: First fully-connected layer dict with 'W' and 'b'
        fc2: Second fully-connected layer dict with 'W' and 'b'

    Returns:
        logits: Final class scores
        cache: Dictionary containing all sublayer caches
    """
    flat, flatten_cache = flatten_forward(x)

    fc1_out, fc1_cache = linear_forward(
        flat,
        fc1["W"],
        fc1["b"]
    )

    relu_out, relu_cache = relu_forward(fc1_out)

    logits, fc2_cache = linear_forward(
        relu_out,
        fc2["W"],
        fc2["b"]
    )

    cache = {
        "flatten_cache": flatten_cache,
        "fc1_cache": fc1_cache,
        "relu_cache": relu_cache,
        "fc2_cache": fc2_cache
    }

    return logits, cache

# Step 47 - lenet_forward
import numpy as np

def lenet_forward(x, params):
    """
    Full LeNet forward pass:
    ConvBlock1 -> ConvBlock2 -> Classifier

    Args:
        x: Input tensor of shape (N, C, H, W)
        params: Dictionary containing:
            conv1, conv2, fc1, fc2

    Returns:
        logits: Class scores
        caches: Dictionary containing block caches
    """
    block1_out, block1_cache = forward_conv_block(
        x,
        params["conv1"]["W"],
        params["conv1"]["b"],
        pool_size=2,
        stride=1,
        pad=0
    )

    block2_out, block2_cache = forward_conv_block(
        block1_out,
        params["conv2"]["W"],
        params["conv2"]["b"],
        pool_size=2,
        stride=1,
        pad=0
    )

    logits, classifier_cache = forward_classifier_block(
        block2_out,
        params["fc1"],
        params["fc2"]
    )

    caches = {
        "block1": block1_cache,
        "block2": block2_cache,
        "classifier": classifier_cache
    }

    return logits, caches

# Step 48 - backward_conv_block
import numpy as np

def backward_conv_block(dout, cache):
    """
    Backward pass for a convolution block:
    Conv -> ReLU -> MaxPool

    Args:
        dout: Upstream gradient from the next layer.
        cache: Dictionary from forward_conv_block containing:
            conv_cache, relu_cache, pool_cache

    Returns:
        dx: Gradient with respect to block input
        dW: Gradient with respect to convolution weights
        db: Gradient with respect to convolution bias
    """

    # Backprop through max pooling
    d_pool = maxpool2d_backward(
        dout,
        cache["pool_cache"]
    )

    # Backprop through ReLU
    d_relu = relu_backward(
        d_pool,
        cache["relu_cache"]
    )

    # Backprop through convolution
    dx, dW, db = conv2d_backward(
        d_relu,
        cache["conv_cache"]
    )

    return dx, dW, db

# Step 49 - backward_classifier_block
import numpy as np

def backward_classifier_block(dout, cache):
    """
    Backward pass for classifier block:
    Flatten -> FC1 -> ReLU -> FC2

    Returns:
        {
            'dx': gradient into input feature map,
            'fc1': {'dW': ..., 'db': ...},
            'fc2': {'dW': ..., 'db': ...}
        }
    """

    # FC2 backward
    d_relu, dW2, db2 = linear_backward(
        dout,
        cache["fc2_cache"]
    )

    # ReLU backward
    d_fc1, = (relu_backward(
        d_relu,
        cache["relu_cache"]
    ),)

    # FC1 backward
    d_flat, dW1, db1 = linear_backward(
        d_fc1,
        cache["fc1_cache"]
    )

    # Flatten backward
    dx = flatten_backward(
        d_flat,
        cache["flatten_cache"]
    )

    return {
        "dx": dx,
        "fc1": {
            "dW": dW1,
            "db": db1
        },
        "fc2": {
            "dW": dW2,
            "db": db2
        }
    }

# Step 50 - lenet_backward
import numpy as np

def lenet_backward(dlogits, caches):
    """
    Full LeNet backward pass.

    Args:
        dlogits: Gradient with respect to logits.
        caches: Dictionary from lenet_forward containing:
            block1, block2, classifier

    Returns:
        grads: Dictionary containing gradients for all layers.
    """

    # Backward through classifier
    classifier_grads = backward_classifier_block(
        dlogits,
        caches["classifier"]
    )

    # Gradient flowing into second conv block
    d_block2 = classifier_grads["dx"]

    # Backward through conv block 2
    _, dW2, db2 = backward_conv_block(
        d_block2,
        caches["block2"]
    )

    # Gradient flowing into first conv block
    d_block1 = caches["block2"]["conv_cache"]  # not needed directly
    d_block1 = backward_conv_block(
        d_block2,
        caches["block2"]
    )[0]

    # Backward through conv block 1
    _, dW1, db1 = backward_conv_block(
        d_block1,
        caches["block1"]
    )

    return {
        "conv1": {
            "dW": dW1,
            "db": db1
        },
        "conv2": {
            "dW": dW2,
            "db": db2
        },
        "fc1": {
            "dW": classifier_grads["fc1"]["dW"],
            "db": classifier_grads["fc1"]["db"]
        },
        "fc2": {
            "dW": classifier_grads["fc2"]["dW"],
            "db": classifier_grads["fc2"]["db"]
        }
    }

# Step 51 - lenet_predict (not yet solved)
# TODO: implement

# Step 52 - build_synthetic_image_dataset (not yet solved)
# TODO: implement

# Step 53 - shuffle_indices (not yet solved)
# TODO: implement

# Step 54 - train_test_split (not yet solved)
# TODO: implement

# Step 55 - iterate_minibatches (not yet solved)
# TODO: implement

# Step 56 - train_step (not yet solved)
# TODO: implement

# Step 57 - train_one_epoch (not yet solved)
# TODO: implement

# Step 58 - train_loop (not yet solved)
# TODO: implement

# Step 59 - evaluate (not yet solved)
# TODO: implement

