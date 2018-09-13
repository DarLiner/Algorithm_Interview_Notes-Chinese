"""CharCNN Embedding Layer

References：
    [1509.01626] Character-level Convolutional Networks for Text Classification https://arxiv.org/abs/1509.01626
"""

import tensorflow as tf

from ...utils import get_shape, get_w


# TODO(huay): char_cnn_embedding
def char_cnn_embedding(x, c_embed_size=8, share_cnn_weights=True, name="char_cnn_embedding", reuse=None):
    """
    In:  [N, max_n_word, max_n_char]
    Out: [N, max_n_word, c_embed_size]

    max_n_word: 句子的最大长度
    max_n_char: 单词的最大长度

    Args:
        x:
        c_embed_size:
        share_cnn_weights:
        name:
        reuse:

    Returns:

    """
    max_sentence_len, max_word_len, char_vocab_size = get_shape(x)[1:]

    with tf.variable_scope(name, reuse=reuse):
        char_embed_mat = get_w([char_vocab_size, c_embed_size], name="char_embed_matrix")








