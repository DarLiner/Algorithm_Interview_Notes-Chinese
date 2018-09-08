"""高速网络 Highway Network

注意 x 经过 Highway 之后维度应该保持不变

References:
    https://github.com/fomorians/highway-fcn
    https://github.com/fomorians/highway-cnn
"""
import tensorflow as tf

from ..utils import get_wb
from ..activations import relu, sigmoid


def highway_dense(x, act_fn=relu, carry_bias=-1.0, name=None):
    """用于全连接层的 highway
    Input shape:  [batch_size, n_input]
    Output shape: [batch_size, n_input]

    公式
        `o = H(x, W)T(x, W) + x(1 - T(x, W))`
    其中
        H, T = dense
    """
    n_input = int(x.get_shape()[-1])
    with tf.variable_scope(name or "highway_dense"):
        W, b = get_wb([n_input, n_input])

        with tf.variable_scope("transform"):
            W_T, b_T = get_wb([n_input, n_input], b_initializer=tf.initializers.constant(carry_bias))

        H = act_fn(tf.matmul(x, W) + b)
        T = sigmoid(tf.matmul(x, W_T) + b_T)
        o = tf.multiply(H, T) + tf.multiply(x, (1. - T))

    return o


def multi_highway_dense(x, n_layer, act_fn=relu, carry_bias=-1.0, name=None):
    """多层 highway_dense
    Input shape:  [batch_size, n_input]
    Output shape: [batch_size, n_input]
    """
    name = name or "highway_dense"
    for i in range(n_layer):
        x = highway_dense(x, act_fn=act_fn, carry_bias=carry_bias, name="{}-{}".format(name, i))

    return x


def highway_conv2d(x, kernel_size,
                   act_fn=relu,
                   strides=1,
                   padding="SAME",
                   carry_bias=-1.0,
                   name=None):
    """用于 conv2d 的 highway
    Input shape:  [batch_size, in_h, in_w, in_channels]
    Output shape: [batch_size, in_h, in_w, in_channels]

    公式
        `o = H(x, W)T(x, W) + x(1 - T(x, W))`
    其中
        H, T = conv2d
    """
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * 2
    if isinstance(strides, int):
        strides = [strides] * 4

    assert len(kernel_size) == 2, "len(kernel_size) == 2"
    assert len(strides) == 4, "len(strides) == 4"

    in_channels = int(x.get_shape()[-1])
    kernel_shape = list(kernel_size) + [in_channels, in_channels]

    with tf.variable_scope(name or "highway_conv2d"):
        W, b = get_wb(kernel_shape, b_initializer=tf.initializers.constant(carry_bias))

        with tf.variable_scope("transform"):
            W_T, b_T = get_wb(kernel_shape)

        H = act_fn(tf.nn.conv2d(x, W, strides=strides, padding=padding) + b)
        T = sigmoid(tf.nn.conv2d(x, W_T, strides=strides, padding=padding) + b_T)
        o = tf.multiply(H, T) + tf.multiply(x, (1. - T))
    return o


def multi_highway_conv2d(x, kernel_size, n_layer,
                         act_fn=relu,
                         strides=1,
                         padding="SAME",
                         carry_bias=-1.0,
                         name=None):
    """多层 highway_conv2d"""
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * n_layer

    assert len(kernel_size) == n_layer, "len(kernel_size) == n_layer"

    name = name or "highway_conv2d"
    for i, kz in enumerate(kernel_size):
        x = highway_conv2d(x, kz, act_fn, strides, padding, carry_bias, name="{}-{}".format(name, i))

    return x
