"""
全连接层

References:
    tensorlayer.layers.DenseLayer
    keras.layers.Dense
"""
import tensorflow as tf

from ..utils import get_wb
from ..activations import relu
from ..activations import linear


def dense(x, n_unit, act_fn=relu, name=None, reuse=None):
    """全连接层
    Input shape:  [batch_size, n_input]
    Output shape: [batch_size, n_unit]

    如果需要 reuse 推荐使用类实现的 `Dense`

    Args:
        x(tf.Tensor):
        n_unit(int): 
        act_fn:
        name(str):
        reuse(bool):
    """
    # n_input = tf.shape(x)[-1]  # err: need int but tensor
    n_input = int(x.get_shape()[-1])
    with tf.variable_scope(name or "dense", reuse=reuse):
        W, b = get_wb([n_input, n_unit])
        o = act_fn(tf.matmul(x, W) + b)
    return o


def multi_dense(x, n_unit_ls, act_fn=relu, name=None):
    """多层全连接
    Input shape:  [batch_size, n_input]
    Output shape: [batch_size, n_unit_list[-1]]

    Args:
        x(tf.Tensor):
        n_unit_ls(list of int):
        act_fn:
        name(str):
    """
    # n_layer = len(n_unit_list)
    name = name or "dense"
    for i, n_unit in enumerate(n_unit_ls):
        x = dense(x, n_unit, act_fn=act_fn, name="{}-{}".format(name, i))

    return x


def linear_dense(x, n_unit, name=None, reuse=None):
    """线性全连接层
    Input shape:  [batch_size, n_input]
    Output shape: [batch_size, n_unit]

    Args:
        x(tf.Tensor):
        n_unit(int):
        name(str):
        reuse(bool)
    """
    return dense(x, n_unit, act_fn=linear, name=(name or "linear_dense"), reuse=reuse)


class Dense(object):
    """全连接层的类实现，方便 reuse"""

    def __init__(self, n_unit, act_fn=relu, name=None):
        """"""
        self.n_unit = n_unit
        self.act_fn = act_fn
        self.name = name

        self._built = False

    def _build(self, n_input):
        """"""
        with tf.variable_scope(self.name or "dense"):
            self.W, self.b = get_wb([n_input, self.n_unit])

    def _call(self, x):
        """"""
        o = self.act_fn(tf.matmul(x, self.W) + self.b)
        return o

    def __call__(self, x):
        """"""
        n_input = int(x.get_shape()[-1])
        if not self._built:
            self._build(n_input)
            self._built = True

        return self._call(x)
