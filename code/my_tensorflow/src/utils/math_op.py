"""Math

References:
    https://www.tensorflow.org/api_guides/python/math_ops
"""

import tensorflow as tf


def dot(x, y):
    """Multiplies 2 tensors (and/or variables) and returns a *tensor*.
    """
    x_shape = list(x.get_shape())
    y_shape = list(y.get_shape())
    x_ndim = len(x_shape)
    y_ndim = len(y_shape)

    if x_ndim is not None and (x_ndim > 2 or y_ndim > 2):
        y_permute_dim = list(range(y_ndim))
        y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
        xt = tf.reshape(x, [-1, x_shape[-1]])
        yt = tf.reshape(tf.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
        return tf.reshape(tf.matmul(xt, yt),
                          x_shape[:-1] + y_shape[:-2] + y_shape[-1:])
    else:
        return tf.matmul(x, y)
