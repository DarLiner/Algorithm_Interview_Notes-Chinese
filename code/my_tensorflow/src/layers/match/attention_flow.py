"""Attention Flow Match Layer

Attention flow layer is responsible for linking and fusing information from the context and the query words.

References:
    [paper] Bidirectional Attention Flow for Machine Comprehension (https://arxiv.org/abs/1611.01603)
    [DuReader] https://github.com/baidu/DuReader/blob/master/tensorflow/layers/match_layer.py
"""

import tensorflow as tf

from ...activations import softmax
from ...utils import get_w


def attention_flow_self(c, q, T=None, name=None, reuse=None):
    """Attention Flow Match Layer
    Input shape:
        c: [batch_size, T, n_feature]
        q: [batch_size, J, n_feature]
    Output shape:
        [batch_size, T, n_feature*4]

    这里 self 的意思是直接使用 c 和 q 的 cosine 相似度作为相似度矩阵，因此该层没有新的参数—— DuReader的实现方式
    原文使用了可训练的相似度矩阵

    Args:
        c: context encoding
            shape: [batch_size, T, n_feature]
        q: question encoding
            shape: [batch_size, J, n_feature]
        T(int): context length
        J(int): question length
        name(str):
        reuse(bool):
    """
    T = T or int(c.get_shape()[-2])

    with tf.variable_scope(name or "attention_flow", reuse=reuse):
        # cosine similarity matrix
        S = tf.matmul(c, q, transpose_b=True)  # [batch_size, T, J]

        # attention weights on the question words for context2question_attention(c2q_a)
        c2q_a = tf.matmul(softmax(S), q)  # [batch_size, T, n_feature]

        # attention weights on the context words for question2context_attention(q2c_a)
        b = tf.reduce_max(S, axis=2)  # [batch_size, T]
        b = softmax(b, axis=-1)  # [batch_size, T]
        b = tf.expand_dims(b, axis=1)  # [batch_size, 1, T]
        q2c_a = tf.matmul(b, c)  # [batch_size, 1, n_feature]
        q2c_a = tf.tile(q2c_a, [1, T, 1])  # [batch_size, T, n_feature]

        g = tf.concat([c, c2q_a, c * c2q_a, c * q2c_a], -1)  # [batch_size, T, n_feature*4]

    return g


# TODO(huay): 原版带参数的attention_flow
#   ref: (PyTorch) https://github.com/jojonki/BiDAF/blob/master/layers/bidaf.py
def attention_flow(c, q, T=None, name=None, reuse=None):
    """"""
