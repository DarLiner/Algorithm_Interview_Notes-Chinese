""""""

import re
import csv
import numpy as np

data_source = r"D:\OneDrive\workspace\data\nlp\文本分类\ag_news_csv\train.csv".replace("\\", "/")


class Data(object):
    # 定义一些全局变量、超参数
    def __init__(self,
                 data_source,
                 alphabet="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
                 l0=1014,
                 batch_size=128,
                 no_of_classes=4):

        self.alphabet = alphabet
        self.alphabet_size = len(self.alphabet)
        self.dict = {}
        self.no_of_classes = no_of_classes
        for i, c in enumerate(self.alphabet):
            self.dict[c] = i + 1

        self.length = l0
        self.batch_size = batch_size
        self.data_source = data_source

    def loadData(self):
        data = []
        with open(self.data_source, 'r') as f:
            rdr = csv.reader(f, delimiter=',', quotechar='"')
            # 将每行数据的二三项进行处理拼接得到文本
            for row in rdr:
                txt = ""
                for s in row[1:]:
                    txt = txt + " " + re.sub("^\s*(.-)\s*$", "%1", s).replace("\\n", "\n")
                # 第一项为标签，构造训练数据
                data.append((int(row[0]), txt))

        self.data = np.array(data)
        self.shuffled_data = self.data

    def shuffleData(self):
        # shufflter数据。为每次epoch打乱数据顺序
        data_size = len(self.data)

        shuffle_indices = np.random.permutation(np.arange(data_size))
        self.shuffled_data = self.data[shuffle_indices]

    def getBatchToIndices(self, batch_num=0):
        # 将data分成batch，并将字符转化为one-hot编码
        data_size = len(self.data)
        start_index = batch_num * self.batch_size
        end_index = data_size if self.batch_size == 0 else min((batch_num + 1) * self.batch_size, data_size)
        # 获取第batch_num批数据（根据开始和结束索引）
        batch_texts = self.shuffled_data[start_index:end_index]
        # 将单词转化为索引存储在batch_indices中
        batch_indices = []
        # 类别的one-hot编码，一共4类
        one_hot = np.eye(self.no_of_classes, dtype='int64')
        classes = []
        for c, s in batch_texts:
            batch_indices.append(self.strToIndexs(s))
            c = int(c) - 1
            classes.append(one_hot[c])

        return np.asarray(batch_indices, dtype='int64'), classes

    def strToIndexs(self, s):
        # 将一个字符串进行padding并转化为索引
        s = s.lower()
        m = len(s)
        n = min(m, self.length)
        # padding
        str2idx = np.zeros(self.length, dtype='int64')
        for i in range(1, n):
            c = s[i]
            if c in self.dict:
                str2idx[i] = self.dict[c]
        return str2idx

    # 注意，上面的函数使用的是正序读入数据，也可以使用论文中提到的倒序读取数据如下所示：
    def strToIndexs2(self, s):
        s = s.lower()
        m = len(s)
        n = min(m, self.length)
        str2idx = np.zeros(self.length, dtype='int64')
        k = 0
        for i in range(1, n + 1):
            c = s[-i]
            if c in self.dict:
                str2idx[i - 1] = self.dict[c]
        return str2idx

    def getLength(self):
        return len(self.data)


if __name__ == '__main__':
    """"""
    d = Data(data_source)
    d.loadData()