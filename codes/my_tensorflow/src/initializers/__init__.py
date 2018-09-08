"""变量初始化器"""

import tensorflow as tf


'''常用的 Tensorflow 基本初始化器'''
# 零
zeros = tf.initializers.zeros
# 常量
constant = tf.initializers.constant
# 标准正太分布
normal = tf.initializers.random_normal
# 截断正态分布
truncated_normal = tf.initializers.truncated_normal
# 均匀分布
uniform = tf.initializers.random_uniform
#
