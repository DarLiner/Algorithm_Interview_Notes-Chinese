""""""
import random
import numpy as np
import itertools
from collections import deque
from gensim.models.word2vec import Word2Vec


def gen_batch_sg(sentences, batch_size=8, skip_window=1, num_skips=None, pad_str=("<PAD>", "<PAD>"), use_pad=False):
    """
    Args:
        sentences(list of tokens):
            A list of lists of tokens.
            If the source is "text8" type, just pass like this `[[text8]]`
        batch_size(int):
        skip_window(int):
        num_skips(int):
            num_skips <= skip_window*2
        pad_str(tuple):
            (START_PAD_STR, END_PAD_STR)
        use_pad(bool):
            If False, the pad_str just help deal data but no use.

    Examples:
        ss = ["a b c d e f".split(), "1 2 3 4 5".split()]
        for inputs, labels in itertools.islice(gen_batch_sg(ss), 10):
            print(inputs, labels)

    Returns:
        Batch data for skip gram
    """
    if num_skips is None:
        num_skips = 2 * skip_window
    else:
        assert num_skips <= 2 * skip_window
    assert batch_size % num_skips == 0

    inputs, labels = [], []
    pad_sta, pad_end = pad_str[0], pad_str[1]

    for data in itertools.cycle(sentences):
        data = [pad_sta] * skip_window + data + [pad_end] * skip_window

        center_index = skip_window
        while center_index < len(data) - skip_window:
            center_word = data[center_index]
            l_context_words = data[center_index - skip_window: center_index]
            r_context_words = data[center_index + 1: center_index + skip_window + 1]
            context_words = l_context_words + r_context_words
            context_words = random.sample(context_words, num_skips)

            for context_word in context_words:
                if use_pad:
                    inputs.append(center_word)
                    labels.append(context_word)
                else:
                    if context_word != pad_sta and context_word != pad_end:
                        inputs.append(center_word)
                        labels.append(context_word)

                if len(inputs) == batch_size:
                    yield inputs, labels
                    inputs, labels = [], []

            center_index += 1
