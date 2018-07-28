import os
import bz2
import nltk
import random
import numpy as np
from math import ceil
from collections import Counter, deque
from six.moves.urllib.request import urlretrieve


def maybe_download(filename, expected_bytes, dir="."):
    full_path = os.path.join(dir, filename)
    if not os.path.exists(full_path):
        print('Downloading file...')
        full_path, _ = urlretrieve(url + filename, full_path)
    statinfo = os.stat(full_path)
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return full_path


def read_data(filename):
    with bz2.BZ2File(filename) as f:
        data = []
        file_string = f.read().decode('utf-8').lower()
        file_string = nltk.word_tokenize(file_string)  # # nltk 提供的分词器
        data.extend(file_string)
    return data


def read_data_small(filename):
    """
    Extract a part of the data
    """
    with bz2.BZ2File(filename) as f:
        data = []
        file_size = os.stat(filename).st_size
        chunk_size = 1024 * 1024  # 限制读取的数据
        print('Reading data...')
        for i in range(int(ceil(file_size // chunk_size) + 1)):
            bytes_to_read = min(chunk_size, file_size - (i * chunk_size))
            file_string = f.read(bytes_to_read).decode('utf-8')
            file_string = file_string.lower()
            file_string = nltk.word_tokenize(file_string)  # nltk 提供的分词器
            data.extend(file_string)
    return data


def build_dataset(words, vocabulary_size=50000):
    word_cnt = [['UNK', -1]]

    # Use `Counter` to get the most common words
    word_cnt.extend(Counter(words).most_common(vocabulary_size - 1))
    word2id = dict()

    for word, _ in word_cnt:
        word2id[word] = len(word2id)  # ID start from 0

    data = list()
    unk_cnt = 0
    for word in words:
        if word in word2id:
            index = word2id[word]
        else:
            index = 0  # word2id['UNK']
            unk_cnt += 1
        data.append(index)

    # word_cnt[0] = ['UNK', unk_cnt]
    word_cnt[0][1] = unk_cnt

    id2word = dict(zip(word2id.values(), word2id.keys()))
    assert len(word2id) == vocabulary_size

    return data, word_cnt, word2id, id2word


data_index = 0
"""指定整个滑动窗口 [context center context] 的起始位置"""


def generate_batch_sg(data, batch_size=8, skip_window=1, num_skips=None):
    """Function to generate **one** training batch for the skip-gram model.

    每次完成 batch_size // num_skips 个中心词的构建，每个中心词取 num_skips 个上下文词

    Args:
        num_skips: 随机选择`num_skips`个窗口中的 context words
            在CS224n的课程里没有提到这个参数，也就是默认使用所有的上下文词，即 `num_skips=2 * skip_window`
        skip_window: 一侧的窗口长度，完整的窗口大小为 1+2*skip_window

    Returns:
        inputs: center_words
            形如 ndarray([1,1,33,33,55,55，67,67]) 每个中心词的重复次数等于`num_skips`
        labels: context_words
            形如 ndarray([23,243,543,65,7658,342，8567,3123])
            长度与 center_words 相同，分别是对应中心词的上下文词，即 (1,23),(1,243),(33,543),...
    """
    global data_index

    if num_skips is None:
        num_skips = 2 * skip_window
    else:
        assert num_skips <= 2 * skip_window
    assert batch_size % num_skips == 0

    # center_words
    inputs = np.ndarray(shape=(batch_size,), dtype=np.int32)  # (batch_size,)
    # context_words
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)  # (batch_size, 1)

    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = deque(maxlen=span)  # 双端队列，支持自动弹出

    if data_index + span > len(data):
        data_index = 0
    # buffer.extend(data[data_index:data_index + span])
    # data_index += span
    for _ in range(span):
        buffer.append(data[data_index])
        data_index += 1

    context_words = [w for w in range(span) if w != skip_window]
    for i in range(batch_size // num_skips):
        context_words = random.sample(context_words, num_skips)

        for j, context_word in enumerate(context_words):
            inputs[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]

        if data_index >= len(data):
            buffer = data[:span]
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1

    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index - span + len(data)) % len(data)

    return inputs, labels


if __name__ == '__main__':
    """"""
    url = 'http://www.evanjones.ca3360286/software/'
    filename = maybe_download('wikipedia2text-extracted.txt.bz2', 18377035, dir="./data")

    # filename = "./02-词向量/data/wikipedia2text-extracted.txt.bz2"  # for Ipython
    # words = read_data(filename)
    words = read_data_small(filename)
    print('Data size %d' % len(words))
    print('Example words (start): ', words[:10])
    print('Example words (end): ', words[-10:])

    data, count, word2id, id2word = build_dataset(words)
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10])
    del words

    print('data:', [id2word[di] for di in data[:8]])

    for skip_window in [1, 2]:
        data_index = 0
        batch, labels = generate_batch_sg(data, batch_size=8, skip_window=skip_window)
        print('\nwith window_size = %d:' % skip_window)
        print('    batch:', [id2word[bi] for bi in batch])
        print('    labels:', [id2word[li] for li in labels.reshape(8)])
