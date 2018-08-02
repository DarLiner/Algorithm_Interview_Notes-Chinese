"""
倒排索引
"""

import os
import sys
import nltk
import glob
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer

tokenizer = RegexpTokenizer(r'\w+')
""" 分词器：匹配所有连续的字母串
比如 "abc!@#def$%^gh, ijk" -> 'abc' 'def' 'gh' 'ijk'
"""
stemmer = PorterStemmer()
""" 词干提取器
"""


def combine_indexes(tokens, files):
    index, freq_word = create_inverse_index(files)

    ret = dict()
    # sum_freq = 0
    # index_list = []
    # print(tokens)
    ret_set = []
    for term in tokens:
        if term in index.keys():
            ret[term] = index[term]
            # print("Term is " + str(term))
            # print("Index term " + str(index[term]))
            # index_list.append(index[term])
            # sum_freq = sum_freq + freq_word[term]
        else:
            ret[term] = set()

    ret_sets = list(ret.values())
    ret["__∩__"] = set.intersection(*ret_sets)

    # print("Index list " + str(index_list))

    # if sum_freq:
    #     index_result = list(set.intersection(*index_list))  # 求交集
    #     print("Index result is " + str(index_result))
    #     return index_result, sum_freq
    # else:
    #     return ["No results found"], 0
    return ret


def input_parse(word):
    word = word.strip()
    if ',' in word:
        words_list = word.split(',')
    elif ' ' in word:
        words_list = word.split(' ')
    elif ';' in word:
        words_list = word.split(';')
    elif ':' in word:
        words_list = word.split(':')
    else:
        words_list = [word]

    return words_list


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


def search(txt, files_list):
    # tokens = input_parse(term)
    tokens = tokenizer.tokenize(txt)
    tokens = [word_clean(token) for token in tokens]
    index_result, sum_freq = combine_indexes(tokens, files_list)
    return index_result, sum_freq


def get_all_subset(tokens: list):
    """获取 tokens 的所有非空子集"""
    tokens = list(set(tokens))

    ret = []
    for i in range(len(tokens))
        for j in range(i, )




if __name__ == '__main__':
    dir_path = './data'
    files = ['a.txt', 'b.txt', 'c.txt']
    files = [os.path.join(dir_path, file) for file in files]

    index, freq_word = create_inverse_index(files)
    search_txt = 'html a z'
    tokens = tokenizer.tokenize(search_txt)
    tokens = [word_clean(token) for token in tokens]

    # index_result, sum_freq = search(search_txt, files)

    ret = combine_indexes(tokens, files)

    print(ret)
    """ 
    
    """
