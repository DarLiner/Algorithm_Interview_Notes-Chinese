"""
Sigmoid 示例
"""
import tensorflow as tf
import numpy as np

# Using placeholder as Input
g = tf.Graph()
with g.as_default():
    # Input shape: [1, 10]
    x = tf.placeholder(shape=[1, 10], dtype=tf.float32, name='x')

    # Variable
    W = tf.Variable(
        tf.random_uniform(shape=[10, 5], minval=-0.1, maxval=0.1, dtype=tf.float32), name='W')
    b = tf.Variable(tf.zeros(shape=[5], dtype=tf.float32), name='b')

    # Output shape: [1, 5]
    #   [1, 10] * [10, 5] + [5] = [1, 5]
    h = tf.nn.sigmoid(tf.matmul(x, W) + b)

with tf.Session(graph=g) as sess:
    """"""
    # Init_op
    tf.global_variables_initializer().run()

    # Run
    h_eval = sess.run(h, feed_dict={x: np.random.rand(1, 10)})
    print(h_eval)
