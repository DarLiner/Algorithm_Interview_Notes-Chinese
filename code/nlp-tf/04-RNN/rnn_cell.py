import tensorflow as tf


def basic_rnn_cell(x, h_minus_1, scope):
    """"""
    with tf.variable_scope(scope, reuse=True):
        W_xh = tf.get_variable('W_xh')
        W_hh = tf.get_variable('W_hh')

        h = tf.nn.tanh(tf.matmul(tf.concat([x, h_minus_1], 1), tf.concat([W_xh, W_hh], 0)))
        return h


def lstm_cell(i, o, state):
    """Create an LSTM cell"""
    input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
    forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
    update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
    state = forget_gate * state + input_gate * tf.tanh(update)
    output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
    return output_gate * tf.tanh(state), state


def lstm_cell_with_peephole(i, o, state):
    """
    LSTM with peephole connections
    Our implementation for peepholes is based on
    https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43905.pdf
    """
    input_gate = tf.sigmoid(tf.matmul(i, ix) + state*ic + tf.matmul(o, im) + ib)
    forget_gate = tf.sigmoid(tf.matmul(i, fx) + state*fc + tf.matmul(o, fm) + fb)
    update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
    state = forget_gate * state + input_gate * tf.tanh(update)
    output_gate = tf.sigmoid(tf.matmul(i, ox) + state*oc + tf.matmul(o, om) + ob)

    return output_gate * tf.tanh(state), state


def gru_cell(i, o):
    """Create a GRU cell."""
    reset_gate = tf.sigmoid(tf.matmul(i, rx) + tf.matmul(o, rh) + rb)
    h_tilde = tf.tanh(tf.matmul(i, hx) + tf.matmul(reset_gate * o, hh) + hb)
    z = tf.sigmoid(tf.matmul(i, zx) + tf.matmul(o, zh) + zb)
    h = (1 - z) * o + z * h_tilde

    return h


from math import log2