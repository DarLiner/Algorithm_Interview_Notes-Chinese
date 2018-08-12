"""Highway network

References:
    fomorians/highway-fcn https://github.com/fomorians/highway-fcn/blob/master/main.py
"""

import tensorflow as tf
from keras.layers import *
import keras.backend as K
from keras.layers.convolutional import _Conv


def highway_layer():
    """"""
