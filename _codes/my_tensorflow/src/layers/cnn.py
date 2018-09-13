"""卷积层

References:
    tensorlayer.layers.Conv2dLayer
"""
import tensorflow as tf

from ..activations import relu
from ..utils import get_wb, get_shape


# TODO(huay)
def conv1d():
    """"""


def conv2d(x, kernel_size, out_channels,
           act_fn=relu,
           strides=1,
           padding="SAME",
           name=None,
           reuse=None):
    """2-D 卷积层
    Input shape:  [batch_size, in_h, in_w, in_channels]
    Output shape: [batch_size, out_h, out_w, out_channels]

    Args:
        x(tf.Tensor):
        kernel_size(int or list of int):
        out_channels(int):
        act_fn(function):
        strides(int or list of int):
        padding(str):
        name(str):
        reuse(bool):

    Returns:

    """
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * 2
    if isinstance(strides, int):
        strides = [strides] * 4

    assert len(kernel_size) == 2
    assert len(strides) == 4

    in_channels = get_shape(x)[-1]
    kernel_shape = list(kernel_size) + [in_channels, out_channels]  # [kernel_h, kernel_w, in_channels, out_channels]

    with tf.variable_scope(name or "conv2d", reuse=reuse):
        W, b = get_wb(kernel_shape)

        o = tf.nn.conv2d(x, W, strides=strides, padding=padding) + b
        o = act_fn(o)

    return o
