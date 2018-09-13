"""
倒排索引

没有任何优化，只是展示一下什么是“倒排索引”
"""

import os

from itertools import combinations

from nltk.tokenize import RegexpTokenizer
from nltk import PorterStemmer

tokenizer = RegexpTokenizer(r'\w+')
""" 分词器：匹配所有连续的字母串
比如 "abc!@#def$%^gh, ijk" -> 'abc' 'def' 'gh' 'ijk'
"""
stemmer = PorterStemmer()
""" 词干提取器
"""


def word_clean(word):
    """单词清洗"""
    word = word.lower()  # 小写化
    word = stemmer.stem(word)  # 提取词干
    return word


def create_inverse_index(files_list):
    """对给定文件列表构建倒排索引"""

    ''' 倒排索引字典 '''
    index = dict()
    ''' 频率字典：统计每个词在多少篇文档中出现 '''
    word_freq = dict()

    # 读取文件
    for f in files_list:
        txt = open(f).read()
        # words = word_tokenize(txt)
        words = tokenizer.tokenize(txt)
        # creating inverted index data structure
        for word in words:
            word = word_clean(word)  # 单词清洗
            if word not in index:
                index[word] = {f}
            else:
                index[word].add(f)

    for word in index.keys():
        word_freq[word] = len(index[word])

    # print(index)
    # print(word_freq)
    return index, word_freq


# def get_all_subset(tokens: list):
#     """获取 tokens 的所有非空子集"""
#     tokens = set(tokens)
#
#     ret = []
#     for i in range(1, len(tokens) + 1):
#         ret.extend(list(combinations(tokens, i)))
#
#     return ret


def search_tokens(tokens, inverse_index, word_freq=None):
    """"""
    ret = dict()
    for term in tokens:
        if term in inverse_index:
            ret[frozenset([term])] = inverse_index[term]
        else:
            ret[frozenset([term])] = set()
    return ret


def search(txt, inverse_index, word_freq=None):
    """

    Args:
        txt(str):
        inverse_index(dict):
        word_freq(dict):

    Returns:
        dict
    """
    tokens = tokenizer.tokenize(txt)
    tokens = [word_clean(token) for token in tokens]

    ret = search_tokens(tokens, inverse_index, word_freq)

    for i in range(2, len(tokens) + 1):
        for ts in combinations(tokens, i):
            ret[frozenset(ts)] = set.intersection(*[ret[frozenset([t])] for t in ts])

    # tokens_list = get_all_subset(tokens)
    # for tokens in tokens_list:
    #     ret[tokens] = search_tokens(tokens, inverse_index, word_freq)

    return ret


def printy(ret):
    """

    Args:
        ret(dict):

    Returns:

    """
    for k, v in ret.items():
        tokens = list(k)
        print(tokens)
        if v:
            for t in v:
                print('\t', t)
        else:
            print('\t', '---')


if __name__ == '__main__':
    dir_path = './data'
    files = ['a.txt', 'b.txt', 'c.txt']
    files = [os.path.join(dir_path, file) for file in files]

    inverse_index, word_freq = create_inverse_index(files)
    # print(inverse_index)

    search_txt = 'html a z'
    ret = search(search_txt, inverse_index, word_freq)
    # print(ret)
    printy(ret)
    r""" 
    ['html']
         ./data\b.txt
         ./data\c.txt
    ['a']
         ./data\b.txt
         ./data\a.txt
    ['z']
         -
    ['a', 'html']
         ./data\b.txt
    ['html', 'z']
         -
    ['a', 'z']
         -
    ['a', 'html', 'z']
         -
    """
