"""工具函数
函数列表：
    sigmoid
    sigmoid_prime
    one_hot

References:
    https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network2.py
"""
import numpy as np
from keras.layers import Dense


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


def cross_entropy(y_, y):
    """Cross Entropy loss
    """
    return np.sum(np.clip(-y * np.log(y_), 1e-10, 1.0))


def softmax(y_):
    """SoftMax
    """
    return np.exp(y_) / np.sum(np.exp(y_), axis=0)
