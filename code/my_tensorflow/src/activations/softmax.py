"""SoftMax"""

import tensorflow as tf


def softmax(x, axis=-1):
    """
    Args:
        x(tf.Tensor):
        axis:
    """
    n_dim = x.get_shape().ndims
    assert n_dim >= 2, "Cannot apply softmax to a tensor that is 1D"

    if n_dim == 2:
        return tf.nn.softmax(x)
    else:
        e = tf.exp(x - tf.reduce_max(x, axis=axis, keepdims=True))
        s = tf.reduce_sum(e, axis=axis, keepdims=True)
        return e / s
