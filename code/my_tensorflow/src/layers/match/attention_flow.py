"""Attention Flow 匹配层

Attention flow layer is responsible for linking and fusing information from the context and the query words.

References:
    [paper] Bidirectional Attention Flow for Machine Comprehension (https://arxiv.org/abs/1611.01603)
    [github/DuReader] https://github.com/baidu/DuReader/blob/master/tensorflow/layers/match_layer.py
    [github/BiDAF(PyTorch)] https://github.com/jojonki/BiDAF/blob/master/layers/bidaf.py
"""

import tensorflow as tf

from ...activations import softmax
from ...utils import get_w


def attention_flow_self(h, u, T=None, J=None, d=None, name=None, reuse=None):
    """Attention Flow Self Match Layer
    Input shape:
        h: [N, T, d]  # 原文中的 shape 为 [N, T, 2d], 因为经过了 bi-LSTM, 维度扩了一倍
        u: [N, J, d]
    Output shape:
        [N, T, 4d]

    这里用到的变量名严格按照论文中的定义

    后缀 self 的意思是直接使用 h 和 u 的 cosine 相似度作为相似度矩阵，因此该层没有新的参数—— DuReader的实现方式
    原文使用了可训练的相似度矩阵，带参数的版本参考 `attention_flow()`

    这里实际上没有用到 J 和 d 这个参数，保留是为了与 `attention_flow()` 的参数兼容

    Args:
        h: context encoding     shape: [N, T, d]
        u: question encoding    shape: [N, J, d]
        T(int): context length
        J(int): question length
        d(int): features size
        name(str):
        reuse(bool):
    """
    T = T or int(h.get_shape()[-2])

    with tf.variable_scope(name or "attention_flow_self", reuse=reuse):
        # similarity matrix
        S = tf.matmul(h, u, transpose_b=True)  # [N, T, J]

        # u_tilde(u~): context to question attended query vectors
        u_tilde = tf.matmul(softmax(S), u)  # [N, T, d]

        # h_tilde(h~): question to context attended query vectors
        b = tf.reduce_max(S, axis=2)  # [N, T]
        b = softmax(b, axis=-1)  # [N, T]
        b = tf.expand_dims(b, axis=1)  # [N, 1, T]
        h_tilde = tf.matmul(b, h)  # [N, 1, d]
        h_tilde = tf.tile(h_tilde, [1, T, 1])  # [N, T, d]

        g = tf.concat([h, u_tilde, h * u_tilde, h * h_tilde], axis=-1)  # [N, T, 4d]

    return g


def attention_flow(h, u, T=None, J=None, d=None, name=None, reuse=None):
    """Attention Flow Match Layer
    Input shape:
        h: [N, T, d]  # 原文中的 shape 为 [N, T, 2d], 因为经过了 bi-LSTM, 维度扩了一倍
        u: [N, J, d]
    Output shape:
        [N, T, 4d]

    Args:
        h: context encoding     shape: [N, T, d]
        u: question encoding    shape: [N, J, d]
        T(int): context length
        J(int): question length
        d(int): features size
        name(str):
        reuse(bool):

    Returns:

    """
    d = d or int(h.get_shape()[-1])
    T = T or int(h.get_shape()[-2])
    J = J or int(u.get_shape()[-2])

    with tf.variable_scope(name or "attention_flow", reuse=reuse):
        h_expand = tf.tile(tf.expand_dims(h, axis=2), [1, 1, J, 1])  # [N, T, J, d]
        u_expand = tf.tile(tf.expand_dims(u, axis=1), [1, T, 1, 1])  # [N, T, J, d]
        hu = tf.multiply(h_expand, u_expand)  # [N, T, J, d]
        h_u_hu = tf.concat([h_expand, u_expand, hu], axis=-1)  # [N, T, J, 3d]
        W_s = get_w([3 * d, 1])  # [3d, 1]

        # similarity matrix
        S = tf.reshape(tf.einsum("ntjd,do->ntjo", h_u_hu, W_s), [-1, T, J])  # [N, T, J]
        # 以上操作等价于
        # S = tf.reshape(tf.matmul(tf.reshape(h_u_hu, [-1, 3*d]), W_s), [-1, T, J])

        # 得到 S 后，下面的操作就与 `attention_flow_self` 一样了

        # u_tilde(u~): context to question attended query vectors
        u_tilde = tf.matmul(softmax(S), u)  # [N, T, d]

        # h_tilde(h~): question to context attended query vectors
        b = tf.reduce_max(S, axis=2)  # [N, T]
        b = softmax(b, axis=-1)  # [N, T]
        b = tf.expand_dims(b, axis=1)  # [N, 1, T]
        h_tilde = tf.matmul(b, h)  # [N, 1, d]
        h_tilde = tf.tile(h_tilde, [1, T, 1])  # [N, T, d]

        g = tf.concat([h, u_tilde, h * u_tilde, h * h_tilde], axis=-1)  # [N, T, 4d]

    return g


# # Test
# def attention_flow_2(h, u, T=None, J=None, d=None, name=None, reuse=None):
#     """Attention Flow Match Layer
#     Input shape:
#         h: [N, T, d]  # 原文中的 shape 为 [N, T, 2d], 因为经过了 bi-LSTM, 维度扩了一倍
#         u: [N, J, d]
#     Output shape:
#         [N, T, 4d]
# 
#     Args:
#         h: context encoding     shape: [N, T, d]
#         u: question encoding    shape: [N, J, d]
#         T(int): context length
#         J(int): question length
#         d(int): features size
#         name(str):
#         reuse(bool):
# 
#     Returns:
# 
#     """
#     print("Test")
#     d = d or int(h.get_shape()[-1])
#     T = T or int(h.get_shape()[-2])
#     J = J or int(u.get_shape()[-2])
# 
#     with tf.variable_scope(name or "attention_flow", reuse=reuse):
#         h_expand = tf.tile(tf.expand_dims(h, axis=2), [1, 1, J, 1])
#         u_expand = tf.tile(tf.expand_dims(u, axis=1), [1, T, 1, 1])
#         hu = tf.multiply(h_expand, u_expand)  # [N, T, J, d]
#         h_u_hu = tf.concat([h_expand, u_expand, hu], axis=-1)  # [N, T, J, 3d]
#         W_s = get_w([3 * d, 1])  # [3d, 1]
# 
#         # similarity matrix
#         # S = tf.reshape(tf.einsum("ntjd,do->ntjo", h_u_hu, W_s), [-1, T, J])
#         # 以上操作等价于
#         S = tf.reshape(tf.matmul(tf.reshape(h_u_hu, [-1, 3 * d]), W_s), [-1, T, J])
# 
#         # 得到 S 后，下面的操作就与 `attention_flow_self` 一样了
# 
#         # u_tilde(u~): context to question attended query vectors
#         u_tilde = tf.matmul(softmax(S), u)  # [N, T, d]
# 
#         # h_tilde(h~): question to context attended query vectors
#         b = tf.reduce_max(S, axis=2)  # [N, T]
#         b = softmax(b, axis=-1)  # [N, T]
#         b = tf.expand_dims(b, axis=1)  # [N, 1, T]
#         h_tilde = tf.matmul(b, h)  # [N, 1, d]
#         h_tilde = tf.tile(h_tilde, [1, T, 1])  # [N, T, d]
# 
#         g = tf.concat([h, u_tilde, h * u_tilde, h * h_tilde], axis=-1)  # [N, T, 4d]
# 
#     return g
