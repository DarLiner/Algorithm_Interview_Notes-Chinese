"""
正则化函数
    `Tensor -> Tensor or None`

Examples:
    l2_regularizer = l2(0.01)
    tf.get_variable(..., regularizer=l2_regularizer, ...)
"""
from .L1L2 import *
