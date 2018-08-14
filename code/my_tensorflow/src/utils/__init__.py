"""
工具函数
"""
from pprint import pprint
from functools import reduce
from operator import mul

from .array_op import *
from .math_op import *

import tensorflow as tf
from ..regularizers import l2_regularizer


tf_dtype = tf.float32
zeros = tf.initializers.zeros
truncated_normal = tf.initializers.truncated_normal


def foo():
    print("foo")


def get_wb(shape,
           w_initializer=truncated_normal,
           b_initializer=zeros,
           w_regularizer=l2_regularizer,
           b_regularizer=l2_regularizer):
    """"""
    W = tf.get_variable('W', shape=shape,
                        dtype=tf_dtype, initializer=w_initializer, regularizer=w_regularizer)
    b = tf.get_variable('b', shape=shape[-1:],
                        dtype=tf_dtype, initializer=b_initializer, regularizer=b_regularizer)
    return W, b


def get_w(shape,
          w_initializer=truncated_normal,
          w_regularizer=l2_regularizer):
    n_in, n_unit = shape
    W = tf.get_variable('W', [n_in, n_unit], dtype=tf_dtype, initializer=w_initializer, regularizer=w_regularizer)
    return W


def get_params_dict():
    """以字典形式获取所有 trainable 参数"""
    param_dict = dict()
    for var in tf.trainable_variables():
        param_dict[var.name] = {"shape": list(map(int, var.shape)),
                                "number": int(reduce(mul, var.shape, 1))}
    return param_dict


def print_params_dict():
    """"""
    param_dict = get_params_dict()
    # pprint(param_dict, indent=2)
    for k, v in param_dict.items():
        print(k, '\t', end='')
        pprint(v, indent=2)
        # for vk, vv in v.items():
        #     print(vk, '-', vv, '\t', end='')
        # print()


def get_params_number():
    """获取参数总量"""
    param_dict = get_params_dict()
    return sum((item["number"] for item in param_dict.values()))


def print_params_number():
    """"""
    print(get_params_number())
