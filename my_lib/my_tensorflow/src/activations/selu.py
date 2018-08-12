"""指数线性单元"""

import tensorflow as tf


def elu(x):
    """指数线性单元"""
    return tf.nn.elu(x)


def selu(x):
    """缩放型指数线性单元"""
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    o = tf.nn.elu(x)
    return scale * tf.where(x > 0, o, alpha * o)
