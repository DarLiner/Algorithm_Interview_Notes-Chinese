"""Tensor Transformations

References:
    https://www.tensorflow.org/api_guides/python/array_ops
    keras.backend
"""

import tensorflow as tf
import keras.backend as K


def permute(x, perm):
    """
    Examples:
        x.shape == [128, 32, 1]
        x = permute(x, [0, 2, 1])
        x.shape == [128, 1, 32]

        y.shape == [128, 64, 32]
        y = permute(x, [2, 1, 0])
        y.shape == [32, 64, 128]

    References:
        K.permute_dimensions()
    """
    return tf.transpose(x, perm)


def repeat(x, n):
    """
    Examples:
        x.shape == [batch_size, n_input]
        x = repeat(x, n_step)
        x.shape == [batch_size, n_step, n_input]

    References:
        K.repeat()
        tf.tile()
    """
    assert x.get_shape().ndims == 2
    x = tf.expand_dims(x, axis=1)  # -> [batch_size, 1, n_input]
    return tf.tile(x, [1, n, 1])  # -> [batch_size, n, n_input]

