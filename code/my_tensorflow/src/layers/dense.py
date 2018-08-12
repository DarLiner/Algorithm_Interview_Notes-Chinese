"""
全连接层

References:
    tensorlayer.layers.DenseLayer
"""

import tensorflow as tf

from ..utils import get_wb
from ..activations import relu


def dense(x, n_unit, act_fn=relu, name=None):
    """
    Args:
        x: need `tf.Tensor` to use `x.get_shape()`
        n_unit(int): 
        act_fn:
        name(str):
    """
    # n_in = tf.shape(x)[-1]  # err: need int but tensor
    n_in = int(x.get_shape()[-1])
    with tf.variable_scope(name or "dense"):
        W, b = get_wb([n_in, n_unit])
        o = act_fn(tf.matmul(x, W) + b)
    return o

