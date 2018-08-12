"""
L1 和 L2 正则化

References:
    keras.regularizers
"""
import tensorflow as tf
import numpy as np


class L1L2Regularizer(object):
    """L1 L2 正则化

    Examples:
        l2_regularizer = l2(0.01)
        tf.get_variable(..., regularizer=l2_regularizer, ...)
    """

    def __init__(self, l1=0., l2=0.):
        """
        Args:
            l1(float): L1 正则化的系数
            l2(float): L2 正则化的系数
        """
        self.l1 = np.asarray(l1, dtype=np.float32)
        self.l2 = np.asarray(l2, dtype=np.float32)

    def __call__(self, x):
        """
        Args:
            x: 注意 x.dtype == float32
        """
        # x = tf.cast(x, dtype=tf.float32)  # 交给外部处理
        loss_regularization = 0.
        if self.l1:
            loss_regularization += tf.reduce_sum(self.l1 * tf.abs(x))
        if self.l2:
            loss_regularization += tf.reduce_sum(self.l2 * tf.square(x))
        return loss_regularization


"""预定义好的正则化器
"""
l1_regularizer = L1L2Regularizer(l1=0.01)

l2_regularizer = L1L2Regularizer(l2=0.01)

l1_l2_regularizer = L1L2Regularizer(l1=0.01, l2=0.01)
