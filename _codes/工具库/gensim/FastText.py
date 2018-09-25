"""
`gensim.models.FastText` 使用示例
"""
# gensim 示例
import gensim
import numpy as np
from gensim.test.utils import common_texts
from gensim.models.keyedvectors import FastTextKeyedVectors
from gensim.models._utils_any2vec import compute_ngrams, ft_hash
from gensim.models import FastText

# 构建 FastText 模型
sentences = [["Hello", "World", "!"], ["I", "am", "huay", "."]]
min_ngrams, max_ngrams = 2, 4  # ngrams 范围
model = FastText(sentences, size=5, min_count=1, min_n=min_ngrams, max_n=max_ngrams)

# 可以通过相同的方式获取每个单词以及任一个 n-gram 的向量
print(model.wv['hello'])
print(model.wv['<h'])
"""
[-0.03481839  0.00606661  0.02581969  0.00188777  0.0325358 ]
[ 0.04481247 -0.1784363  -0.03192253  0.07162753  0.16744071]
"""
print()

# 词向量和 n-gram 向量是分开存储的
print(len(model.wv.vectors))  # 7
print(len(model.wv.vectors_ngrams))  # 57
# gensim 好像没有提供直接获取所有 ngrams tokens 的方法

print(model.wv.vocab.keys())
"""
['Hello', 'World', '!', 'I', 'am', 'huay', '.']
"""
print()

sum_ngrams = 0
for s in sentences:
    for w in s:
        w = w.lower()
        # from gensim.models._utils_any2vec import compute_ngrams
        ret = compute_ngrams(w, min_ngrams, max_ngrams)
        print(ret)
        sum_ngrams += len(ret)
"""
['<h', 'he', 'el', 'll', 'lo', 'o>', '<he', 'hel', 'ell', 'llo', 'lo>', '<hel', 'hell', 'ello', 'llo>']
['<w', 'wo', 'or', 'rl', 'ld', 'd>', '<wo', 'wor', 'orl', 'rld', 'ld>', '<wor', 'worl', 'orld', 'rld>']
['<!', '!>', '<!>']
['<i', 'i>', '<i>']
['<a', 'am', 'm>', '<am', 'am>', '<am>']
['<h', 'hu', 'ua', 'ay', 'y>', '<hu', 'hua', 'uay', 'ay>', '<hua', 'huay', 'uay>']
['<.', '.>', '<.>']
"""
assert sum_ngrams == len(model.wv.vectors_ngrams)
print(sum_ngrams)  # 57
print()

# 因为 "a", "aa", "aaa" 中都只含有 "<a" ，所以它们直积上都只是 "<a"
print(model.wv["a"])
print(model.wv["aa"])
print(model.wv["aaa"])
print(model.wv["<a"])
"""
[ 0.00226487 -0.19139008  0.17918809  0.13084619 -0.1939924 ]
[ 0.00226487 -0.19139008  0.17918809  0.13084619 -0.1939924 ]
[ 0.00226487 -0.19139008  0.17918809  0.13084619 -0.1939924 ]
[ 0.00226487 -0.19139008  0.17918809  0.13084619 -0.1939924 ]
"""
print()

word_unk = "aam"
ngrams = compute_ngrams(word_unk, min_ngrams, max_ngrams)  # min_ngrams, max_ngrams = 2, 4
word_vec = np.zeros(model.vector_size, dtype=np.float32)
ngrams_found = 0
for ngram in ngrams:
    ngram_hash = ft_hash(ngram) % model.bucket
    if ngram_hash in model.wv.hash2index:
        word_vec += model.wv.vectors_ngrams[model.wv.hash2index[ngram_hash]]
        ngrams_found += 1

if word_vec.any():  #
    word_vec = word_vec / max(1, ngrams_found)
else:  # 如果一个 ngram 都没找到，gensim 会报错；个人认为把 0 向量传出来也可以
    raise KeyError('all ngrams for word %s absent from model' % word_unk)

print(word_vec)
print(model.wv["aam"])
"""
[ 0.02210762 -0.10488641  0.05512805  0.09150169  0.00725085]
[ 0.02210762 -0.10488641  0.05512805  0.09150169  0.00725085]
"""

# 如果一个 ngram 都没找到，gensim 会报错
#   其实可以返回一个 0 向量的，它内部实际上是从一个 0 向量开始累加的；
#   但返回时做了一个判断——如果依然是 0 向量，则报错
# print(model.wv['z'])
r"""
Traceback (most recent call last):
  File "D:/OneDrive/workspace/github/DL-Notes-for-Interview/code/工具库/gensim/FastText.py", line 53, in <module>
    print(model.wv['z'])
  File "D:\program\work\Python\Anaconda3\envs\tf\lib\site-packages\gensim\models\keyedvectors.py", line 336, in __getitem__
    return self.get_vector(entities)
  File "D:\program\work\Python\Anaconda3\envs\tf\lib\site-packages\gensim\models\keyedvectors.py", line 454, in get_vector
    return self.word_vec(word)
  File "D:\program\work\Python\Anaconda3\envs\tf\lib\site-packages\gensim\models\keyedvectors.py", line 1989, in word_vec
    raise KeyError('all ngrams for word %s absent from model' % word)
KeyError: 'all ngrams for word z absent from model'
"""

