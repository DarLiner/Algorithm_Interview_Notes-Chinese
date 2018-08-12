"""
工具函数
"""
import tensorflow as tf
from ..regularizers import l2_regularizer

tf_dtype = tf.float32
zeros = tf.initializers.zeros
truncated_normal = tf.initializers.truncated_normal


def foo():
    print(1)


def get_wb(shape,
           w_initializer=truncated_normal,
           b_initializer=zeros,
           w_regularizer=l2_regularizer,
           b_regularizer=l2_regularizer):
    """"""
    n_in, n_unit = shape
    W = tf.get_variable('W', shape=[n_in, n_unit],
                        dtype=tf_dtype, initializer=truncated_normal, regularizer=l2_regularizer)
    b = tf.get_variable('b', shape=[n_unit], 
                        dtype=tf_dtype, initializer=zeros, regularizer=l2_regularizer)
    return W, b


def get_w(shape,
          w_initializer=truncated_normal,
          w_regularizer=l2_regularizer):
    n_in, n_unit = shape
    W = tf.get_variable('W', [n_in, n_unit], dtype=tf_dtype, initializer=w_initializer, regularizer=w_regularizer)
    return W
