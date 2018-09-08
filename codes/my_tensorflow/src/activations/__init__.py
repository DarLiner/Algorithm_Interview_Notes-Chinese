"""激活函数"""

from .relu import *


def linear(x):
    """"""
    return x


def identity(x):
    """"""
    return tf.identity(x)


def sigmoid(x):
    """"""
    return tf.nn.sigmoid(x)


def hard_sigmoid(x):
    """
    x = 0.                  x < -2.5
      = 1.                  x > 2.5
      = 0.2 * x + 0.5       otherwise
    """
    x = (0.2 * x) + 0.5
    x = tf.clip_by_value(x, 0., 1.)
    return x


def tanh(x):
    """"""
    return tf.nn.tanh(x)


def softplus(x):
    """"""
    return tf.nn.softplus(x)


def softsign(x):
    """
    o = x / (1 + abs(x))
    """
    return tf.nn.softsign(x)


def softmax(x, axis=-1):
    """
    Examples:
        n_dim = x.get_shape().ndims
        assert n_dim >= 2

        if n_dim == 2:
            return tf.nn.softmax(x)
        else:
            e = tf.exp(x - tf.reduce_max(x, axis=axis, keepdims=True))
            s = tf.reduce_sum(e, axis=axis, keepdims=True)
            return e / s
    """
    return tf.nn.softmax(x, axis=axis)


def elu(x):
    """指数线性单元"""
    return tf.nn.elu(x)


def selu(x):
    """缩放型指数线性单元"""
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    o = tf.nn.elu(x)
    return scale * tf.where(x > 0, o, alpha * o)
