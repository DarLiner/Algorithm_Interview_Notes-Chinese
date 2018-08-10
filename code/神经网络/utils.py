"""工具函数
函数列表：
    sigmoid
    sigmoid_prime
    one_hot

References:
    https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network2.py
"""
import numpy as np


def sigmoid(z):
    """The sigmoid function.
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function.
    """
    return sigmoid(z) * (1 - sigmoid(z))


def one_hot(i, n_class):
    """Return a n-dim unit vector with 1.0 in the i'th position.
    """
    o = np.zeros((n_class, 1))
    o[i] = 1.0
    return o


def cross_entropy(o, y):
    """
    Cross Entropy loss
        `-sum(p_xi * log(q_xi))`

    在使用交叉熵之前，应该先过一个 softmax 层，
    tensorflow 中将 softmax 和 cross_entropy 一起实现
        `tf.nn.softmax_cross_entropy_with_logits`

    References:
        `np.clip()`: 截断函数，防止 log0

    Args:
        o(list): 网络输出
        y(list):
    """
    o = np.clip(np.array(o), 1e-10, 1.0)  # 截断，防止 log0
    y = np.array(y)
    return np.sum(-y * np.log(o))


def binary_cross_entropy(o, y):
    """Binary Cross Entropy loss

    Args:
        o(int):
        y(int):
    """
    o = np.clip(o, 1e-10, 1.0)
    return -o * np.log(y) - (1 - o) * np.log(1 - y)


def softmax(y_):
    """SoftMax
    """
    return np.exp(y_) / np.sum(np.exp(y_), axis=0)
