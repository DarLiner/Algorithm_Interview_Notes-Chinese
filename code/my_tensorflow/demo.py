from typing import (
    TypeVar, Iterator, Iterable, overload, Container,
    Sequence, MutableSequence, Mapping, MutableMapping, Tuple, List, Any, Dict, Callable, Generic,
    Set, AbstractSet, FrozenSet, MutableSet, Sized, Reversible, SupportsInt, SupportsFloat,
    SupportsBytes, SupportsAbs, SupportsRound, IO, Union, ItemsView, KeysView, ValuesView,
    ByteString, Optional, AnyStr, Type,
)
from src.utils import tf_dtype

print(tf_dtype)

import tensorflow as tf
import numpy as np

from src.layers import multi_dense, Dense

from tensorlayer.layers import DenseLayer, Conv2dLayer

from keras.layers import Dense, Conv2D

import keras.initializers

import keras.utils.layer_utils

x = tf.constant(np.arange(16).reshape([8, 2]), dtype=tf.float32)

o = multi_dense(x, [3, 4, 5], name="DenSe")

# sess = tf.Session(graph=tf.get_default_graph())
tf.reset_default_graph()
sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())

sess.run(o)


def f(x):
    """

    Args:
        x(function):

    Returns:

    """


def b():
    """"""


f(1)

sorted()
