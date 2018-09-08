"""Attention 层

作用与 highway 类似
经过 Attention 层后维度应该保持一致

References:
    https://github.com/philipperemy/keras-attention-mechanism
"""

import tensorflow as tf

from ..utils import get_wb, permute
from ..activations import softmax


def attention_for_dense(x, name=None, reuse=None):
    """
    Input shape:  [batch_size, n_input]
    Output shape: [batch_size, n_input]

    公式
        `o = x * softmax(Wx + b)`

    一般用法
        ```
        x = attention1d(x)
        o = dense(x, n_unit)
        ```
    """
    n_input = int(x.get_shape()[-1])

    with tf.variable_scope(name or "attention_for_dense", reuse=reuse):
        W, b = get_wb([n_input, n_input])

        a = softmax(tf.matmul(x, W) + b)  # attention
        o = tf.multiply(x, a)

    return o


def attention_for_rnn(x, n_step=None, name=None, reuse=None, use_mean_attention=False):
    """
    Input shape:  [batch_size, n_step, n_input]
    Output shape: [batch_size, n_step, n_input]

    Examples:
        以下示例使用了 TensorLayer 库来快速构建模型；本来想用 Keras，但是无法完整获取中间输出的 shape

        Use attention **after** lstm:
            ```
            tf.reset_default_graph()

            # Input shape: [128, 5, 32]
            x = tf.constant(np.arange(10240, dtype=np.float32).reshape([128, 16, 5]))
            x = InputLayer(x)

            # Use attention after lstm
            x = RNNLayer(x, tf.nn.rnn_cell.LSTMCell, n_hidden=32)
            x = attention_for_rnn(x.outputs)

            x = InputLayer(x)
            x = FlattenLayer(x)
            x = DenseLayer(x, n_units=1, act=tf.nn.sigmoid)
            o = x.outputs

            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                o_val = o.eval()
            ```

        Use attention **before** lstm:
            ```
            tf.reset_default_graph()

            # Input shape: [128, 5, 32]
            x = tf.constant(np.arange(10240, dtype=np.float32).reshape([128, 16, 5]))

            # Use attention before lstm
            x = attention_for_rnn(x)
            x = InputLayer(x)
            x = RNNLayer(x, tf.nn.rnn_cell.LSTMCell, n_hidden=32, return_last=True)

            # x = FlattenLayer(x)
            x = DenseLayer(x, n_units=1, act=tf.nn.sigmoid)
            o = x.outputs

            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                o_val = o.eval()
            ```
    """
    n_input = int(x.get_shape()[-1])
    if n_step is None:
        n_step = int(x.get_shape()[-2])  # 这种写法，不兼容 keras.layers.LSTM，此时需要手工传入 n_step

    with tf.variable_scope(name or "attention_for_rnn", reuse=reuse):
        a = permute(x, [0, 2, 1])  # [batch_size, n_input, n_step]
        a = tf.reshape(a, [-1, n_step])  # [batch_size*n_input, n_step]

        W, b = get_wb([n_step, n_step])
        a = softmax(tf.matmul(a, W) + b)
        a = tf.reshape(a, [-1, n_input, n_step])  # [batch_size, n_input, n_step]

        if use_mean_attention:
            a = tf.reduce_mean(a, axis=1)  # [batch_size, n_step]
            a = tf.expand_dims(a, axis=1)  # [batch_size, 1, n_step]
            a = tf.tile(a, [1, n_input, 1])  # [batch_size, n_input, n_step]

        a = permute(a, [0, 2, 1])  # [batch_size, n_step, n_input]
        o = tf.multiply(x, a)  # # [batch_size, n_step, n_input]

    return o






