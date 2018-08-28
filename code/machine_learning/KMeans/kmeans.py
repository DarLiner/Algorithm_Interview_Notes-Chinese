"""
K-Means
"""
import logging as log
import numpy as np
import random

log.basicConfig(format="%(message)s", level=log.INFO)


def load_data(file_path):
    """加载数据
        源数据格式为多行，每行为两个浮点数，分别表示 (x,y)
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as fr:
        for line in fr.read().splitlines():
            line_float = list(map(float, line.split('\t')))
            data.append(line_float)
    data = np.array(data)
    return data


def score_euclidean(a, b):
    """计算两个点之间的欧式距离"""
    s = np.sqrt(np.sum(np.power(a - b, 2)))
    return s


def rand_center(data, k):
    """随机采样 k 个样本作为聚类中心"""
    centers = np.array(random.sample(list(data), k))
    return centers


def k_means(data, k, max_iter=100, score=score_euclidean, e=1e-6):
    """
    K-Means 算法

    一般 K-Mean 算法的终止条件有如下几个：
        1. 所有样本的类别不再改变
        2. 达到最大迭代次数
        3. 精度达到要求（？）

    返回聚类中心及聚类结果
    """
    # 样本数
    n = len(data)

    # 保存结果
    # 每个结果为一个二元组 [label, score] 分别保存每个样本所在的簇及距离质心的距离
    ret = np.array([[-1, np.inf]] * n)

    # 选取聚类中心
    centers = rand_center(data, k)

    changed = True  # 标记样本类别是否改变
    n_iter = 0  # 记录迭代次数
    while changed and n_iter < max_iter:
        changed = False
        n_iter += 1

        for i in range(n):  # 对每个数据
            i_score = np.inf
            i_label = -1
            for j in range(k):  # 与每个质心比较
                s_ij = score(data[i], centers[j])
                if s_ij < i_score:
                    i_score = s_ij
                    i_label = j

            if ret[i, 0] != i_label:  # 样本的类别发生了改变
                changed = True

            ret[i, :] = i_label, i_score

        # 更新聚类中心
        log.info(centers)
        for i in range(k):
            data_i = data[ret[:, 0] == i]  # 标签为 i 的样本
            centers[i, :] = np.mean(data_i, axis=0)  # 按类别过滤样本

    log.info(n_iter)  # 迭代次数
    return centers, ret


def _test():
    """"""
    file_path = r"./data.txt"

    data = load_data(file_path)
    print(data)
    print(np.shape(data)[1])

    s = score_euclidean(data[0], data[1])
    print(s)

    centers = rand_center(data, 3)
    print(centers)


if __name__ == '__main__':
    """"""
    # _test()

    file_path = "./data.txt"
    data = load_data(file_path)

    centers, ret = k_means(data, 3)
    # print(ret)
