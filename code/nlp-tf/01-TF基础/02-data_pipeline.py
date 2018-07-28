"""
Data Pipeline
"""
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

g = tf.get_default_graph()
with tf.Session(graph=g) as sess:
    """"""
    # Create file queue
    filenames = ['./data/test%d.txt' % i for i in range(1, 4)]
    filename_queue = tf.train.string_input_producer(filenames, capacity=3, shuffle=True, name='string_input_producer')

    # Check file exist
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
        else:
            print('File %s found.' % f)

    # Create reader
    reader = tf.TextLineReader(name="reader")
    key, value = reader.read(filename_queue, name='text_read_op')

    # Decode as csv
    cols = tf.decode_csv(value, record_defaults=[[-1.0]] * 10)
    features = tf.stack(cols)

    x = tf.train.shuffle_batch([features], batch_size=3,
                               capacity=5, name='data_batch',
                               min_after_dequeue=1, num_threads=1)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    W = tf.Variable(tf.random_uniform(shape=[10, 5], minval=-0.1, maxval=0.1, dtype=tf.float32), name='W')
    b = tf.Variable(tf.zeros(shape=[5], dtype=tf.float32), name='b')
    h = tf.nn.sigmoid(tf.matmul(x, W) + b)

    tf.global_variables_initializer().run()

    for step in range(5):
        x_eval, h_eval = sess.run([x, h])
        print('========== Step %d ==========' % step)
        print('Evaluated data (x)')
        print(x_eval)
        print('Evaluated data (h)')
        print(h_eval)
        print('')