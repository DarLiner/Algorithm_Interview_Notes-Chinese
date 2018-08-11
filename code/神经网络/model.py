"""使用纯 numpy 编写的神经网络
实现模块：
    前向传播
    反向传播
    随机梯度下降

References:
    https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py
"""
import numpy as np
import random
from utils import sigmoid, sigmoid_prime


class Network(object):

    def __init__(self, n_units):
        """
        初始化定义网络的单元数，包括输入层和输出层
        比如：[784, 30, 10]
        Args:
            n_units(list):
        """
        self.n_layer = len(n_units)
        self.sizes = n_units

        self.lr = 1.0  # 默认值
        self._build()

    def _build(self):
        self.biases = [np.random.randn(n_unit, 1)
                       for n_unit in self.sizes[1:]]

        # self.weights = [np.random.randn(n_unit, input_dim)
        #                 for input_dim, n_unit in zip(self.sizes[:-1], self.sizes[1:])]

        # 优化后可以更快收敛
        self.weights = [np.random.randn(n_unit, input_dim) / np.sqrt(input_dim)
                        for input_dim, n_unit in zip(self.sizes[:-1], self.sizes[1:])]

    def _forward(self, a):
        """前向传播"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        print(a.shape)
        return a

    def _backprop(self, x, y):
        """反向传播，计算单个样本的梯度"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # forward
        activations = [x]  # 按顺序保存每层的激活值
        zs = []  # 按层保存 z，z = wa + b

        a = x  # 保存每层的激活值
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            a = sigmoid(z)  # 使用 sigmoid 激活函数
            zs.append(z)
            activations.append(a)

        # backward
        delta = self._loss_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # 从导数第二层开始反向计算梯度
        for l in range(2, self.n_layer):
            z = zs[-l]
            sp = sigmoid_prime(z)  # sigmoid 的导数
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla_b, nabla_w

    def train(self, train_data, batch_size, learning_rate=None):
        """batch SGD 训练"""
        if learning_rate is not None:
            self.lr = learning_rate

        train_data = list(train_data)
        n = len(train_data)

        random.shuffle(train_data)

        batches = [train_data[k:k + batch_size]
                   for k in range(0, n, batch_size)]

        for batch in batches:
            self._update_theta_by_batch(batch)

    def _update_theta_by_batch(self, batch):
        """按批更新参数"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 计算批梯度
        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self._backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # 更新参数
        self.weights = [w - (self.lr / len(batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (self.lr / len(batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def evaluate(self, test_data):
        """"""
        ret = [(np.argmax(self._forward(x)), y) for (x, y) in test_data]

        return sum(int(x == y) for (x, y) in ret)

    @staticmethod
    def _loss_derivative(o, y):
        """交叉熵损失的导数"""
        return o - y

    def run(self, train_data, dev_data, test_data=None, n_epoch=20, batch_size=20, learning_rate=None):
        """训练，验证与测试
        如果没有验证集，则使用测试集作为验证集
        """
        if learning_rate is not None:
            self.lr = learning_rate

        train_data = list(train_data)
        dev_data = list(dev_data)
        n_dev = len(dev_data)
        test_data = list(test_data)
        for e in range(n_epoch):
            self.train(train_data, batch_size, self.lr)
            print("Epoch {} : {} / {}".format(e + 1, self.evaluate(dev_data), n_dev))

        if test_data is not None:
            n_test = len(test_data)
            print("Test : {} / {}".format(self.evaluate(test_data), n_test))

    def save(self):
        """"""
        # Todo(huay)
