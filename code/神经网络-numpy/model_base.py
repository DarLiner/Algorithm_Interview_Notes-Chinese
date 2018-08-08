"""使用纯 numpy 编写的神经网络
实现模块：
    前向传播
    反向传播
    随机梯度下降

References:
    https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py
"""
import random
import numpy as np

from .utils import *


class NN_base():
    """"""

    def __init__(self, n_hidden_list):
        """

        Args:
            n_hidden_list(list of int):
                隐藏层单元数列表，第一层为输入层，最后一层为输出层
        """

        self.n_layers = len(n_hidden_list)
        self.sizes = n_hidden_list

        self._build()

    def _build(self):
        """"""
        # 上一层的输出就是下一层的输入
        self.weights = [np.random.randn(input_dim, n_unit) for input_dim, n_unit in
                        zip(self.sizes[:-1], self.sizes[1:])]
        self.biases = [np.random.randn(n_unit, 1) for n_unit in self.sizes[1:]]

    def _forward(self, a):
        """"""
        for w, b in zip(self.weights, self.biases):
            z = np.dot(np.transpose(w), a) + b
            a = sigmoid(z)
        return a

    def _backprop(self, x, y):
        """"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self._loss_delta(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.n_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def _update_batch(self):
        """"""

    @staticmethod
    def _loss_delta(o, y):
        """"""
        return o - y

    def train(self, data, ):
        """"""

    def evaluate(self, data):
        """"""
        test_results = [(np.argmax(self._forward(x)), y) for x, y in data]
        return sum(int(x == y) for (x, y) in test_results)
