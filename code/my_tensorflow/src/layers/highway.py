"""高速网络

References:
    https://github.com/fomorians/highway-fcn
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
    """
    n_input = int(x.get_shape()[-1])
    with tf.variable_scope(name or "highway"):
        W, b = get_wb([n_input, n_input])

        with tf.variable_scope("transform"):
            W_T, b_T = get_wb([n_input, n_input], b_initializer=tf.initializers.constant(carry_bias))

        H = act_fn(tf.matmul(x, W) + b)
        T = sigmoid(tf.matmul(x, W_T) + b_T)

        o = tf.multiply(H, T) + tf.multiply(x, (1. - T))

    return o


def multi_highway_dense(x, n_layer, act_fn=relu, carry_bias=-1.0, name=None):
    """多层 highway
    Input shape:  [batch_size, n_input]
    Output shape: [batch_size, n_input]
    """
    name = name or "highway"
    for i in range(n_layer):
        x = highway_dense(x, act_fn=act_fn, carry_bias=carry_bias, name="{}-{}".format(name, i))

    return x
