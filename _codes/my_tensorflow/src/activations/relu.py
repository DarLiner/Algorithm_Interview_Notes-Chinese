"""ReLU 系列激活函数"""
import tensorflow as tf

from ..utils import get_w, get_shape
from ..initializers import constant


def relu(x):
    """ReLU
    `o = max(0., x)`
    """
    return tf.nn.relu(x)


def relu6(x):
    """
    `o = min(max(x, 0), 6)`
    """
    return tf.nn.relu6(x)


def crelu(x, axis=-1):
    """Concatenated ReLU
    """
    return tf.nn.crelu(x, axis=axis)


def leaky_relu(x, alpha=0.1):
    """渗透 ReLU
    `o = max(alpha * x, x)`
    """
    return tf.nn.leaky_relu(x, alpha)


def clip_relu(x, max_value):
    """截断 ReLU
    `o = min(max(0., x), max_value)`
    """
    o = tf.nn.relu(x)
    o = tf.minimum(o, max_value)
    return o


def parametric_relu(x, channel_shared=False, alpha_init=constant(0.), name="parametric_relu", reuse=None):
    """参数化 ReLU

    References:
        tflearn.prelu
    """
    if channel_shared:
        alpha_shape = get_shape(x)[-1:]
    else:
        alpha_shape = [1]

    with tf.variable_scope(name, reuse=reuse):
        alpha = get_w(alpha_shape, w_initializer=alpha_init, name="alpha")
        # o = relu(x) + 0.5 * tf.multiply(alpha, x - tf.abs(x))  # TFLearn
        o = leaky_relu(x, alpha)  # TensorLayer / <Deep Learning>

    return o
