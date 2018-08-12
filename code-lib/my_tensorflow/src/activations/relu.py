"""ReLU 系列激活函数"""
import tensorflow as tf


def relu(x):
    """"""
    return tf.nn.relu(x)


def leaky_relu(x, alpha=0.2):
    """渗透 ReLU"""
    return tf.nn.leaky_relu(x, alpha)


def relu_clip(x, max_value):
    """"""
    o = tf.nn.relu(x)
    o = tf.minimum(o, max_value)
    return o
