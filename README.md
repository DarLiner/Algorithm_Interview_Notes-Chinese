算法/NLP/深度学习/机器学习面试笔记
===
**GitHub** 地址：https://github.com/imhuay/CS_Interview_Notes-Chinese

深度学习/机器学习面试问题整理，想法来源于这个[仓库](https://github.com/elviswf/DeepLearningBookQA_cn).
- 该仓库整理了“花书”《深度学习》中的一些常见问题，其中部分偏理论的问题没有收录，如有需要可以浏览原仓库。

此外，还包括我看到的所有机器学习/深度学习面经中的问题。除了其中 DL/ML 相关的，其他与算法岗相关的计算机知识也会记录。

但是不会包括如前端/测试/JAVA/Android等岗位中有关的问题。

<!-- 替换地址 -->
<!-- https://github.com/imhuay/CS_Interview_Notes-Chinese/blob/master/ -->
## RoadMap
- [数学](./数学)
  - [微积分的本质](./数学/微积分的本质.md)
  - [深度学习的核心](./数学/深度学习的核心.md)
- [机器学习-深度学习-NLP](./机器学习-深度学习-NLP)
  - 深度学习
    - [深度学习基础](./机器学习-深度学习-NLP/DL-深度学习基础.md)
    - [《深度学习》整理](./机器学习-深度学习-NLP/DL-《深度学习》整理.md)
    - [CNN专题](./机器学习-深度学习-NLP/DL-CNN.md)
  - 机器学习
    - [机器学习算法](./机器学习-深度学习-NLP/ML-机器学习算法.md)
    - [机器学习实践](./机器学习-深度学习-NLP/ML-机器学习实践.md)
  - 自然语言处理
    - [序列建模](./机器学习-深度学习-NLP/NLP-序列建模.md) TODO
    - [词向量](./机器学习-深度学习-NLP/NLP-词向量.md)
      - [Word2Vec](./机器学习-深度学习-NLP/NLP-词向量.md#word2vec)
      - [GloVe](./机器学习-深度学习-NLP/NLP-词向量.md#glove)
      - [FastText](./机器学习-深度学习-NLP/NLP-词向量.md#fasttext)
      - WordRank TODO
  - 公共基础
    - 背景知识
    - 损失函数 TODO
    - [工具库](./机器学习-深度学习-NLP/专题-工具库.md) TODO
- [算法](./算法)
  - [题解-剑指Offer](./算法/题解-剑指Offer.md)
  - [题解-LeetCode](./算法/题解-剑指Offer.md) TODO
- [编程语言](./编程语言)
  - [Cpp专题-基础知识](./编程语言/Cpp-基础知识.md)
  - [Cpp专题-左值与右值](./编程语言/Cpp-左值与右值.md)
  - [Cpp专题-面向对象编程](./编程语言/Cpp-面向对象编程.md) TODO
- [笔试面经](./笔试面经)
- [project](./project)
- [code](./code)
  - [工具库](./code/工具库)
    - [gensim.FastText 的使用](./机器学习-深度学习-NLP/NLP-词向量.md#gensimmodelsfasttext-使用示例)
  - [倒排索引](./code/倒排索引)
  - [Tensorflow 基础](./code/tf-基础) TODO
- [招聘要求](./招聘要求.md)

## 必备清单 TODO
- [深度学习](./机器学习-深度学习-NLP/DL-深度学习基础.md)
  - [反向传播算法](./机器学习-深度学习-NLP/DL-深度学习基础.md#反向传播算法)
  - [梯度下降法](./机器学习-深度学习-NLP/DL-深度学习基础.md#梯度下降法)
  - 相关代码 TODO
- [机器学习算法](./机器学习-深度学习-NLP/ML-机器学习算法.md)
  - [逻辑斯蒂回归](./机器学习-深度学习-NLP/ML-机器学习算法.md#逻辑斯蒂回归)
  - [支持向量机](./机器学习-深度学习-NLP/ML-机器学习算法.md#支持向量机)
  - [AdaBoost 算法](./机器学习-深度学习-NLP/ML-机器学习算法.md#adaboost-算法)
  - [GBDT 梯度提升决策树](./机器学习-深度学习-NLP/ML-机器学习算法.md#梯度提升决策树-gbdt)
  - 相关代码 TODO
- 计算机基础
  - [必背算法](./算法/备忘-必备算法.md)
  - [编程语言](./编程语言)


**欢迎分享你在深度学习/机器学习面试过程中遇见的问题！**
---
你可以直接以你遇到的问题作为 issue 标题，然后分享你的回答或者其他参考资料。

当然，你也可以直接创建 PR，分享问题的同时改正我的错误！

> 我会经常修改文档的结构（特别是代码的链接）。如果文中有链接失效，请告诉我！
> 文档中大部分链接都是指向仓库内的文件或标记；涉及编程代码的链接会指向我的另一个仓库（[Algorithm_for_Interview](https://github.com/imhuay/Algorithm_for_Interview-Chinese)）

### Reference

- exacity/[deeplearningbook-chinese](https://github.com/exacity/deeplearningbook-chinese): 深度学习中文版 
- elviswf/[DeepLearningBookQA_cn](https://github.com/elviswf/DeepLearningBookQA_cn): 深度学习面试问题 回答对应的DeepLearning中文版页码
- huihut/[interview: C/C++面试知识总结](https://github.com/huihut/interview) 
- 七月在线：[结构之法 算法之道](https://blog.csdn.net/v_july_v) - CSDN博客
- 在线 LaTeX 公式编辑器 http://www.codecogs.com/latex/eqneditor.php
- GitHub 搜索：[Deep Learning Interview](https://github.com/search?q=deep+learning+interview)
- GitHub 搜索：[Machine Learning Interview](https://github.com/search?q=machine+learning+interview)
    - geekcircle/[machine-learning-interview-qa](https://github.com/geekcircle/machine-learning-interview-qa): 人工智能-机器学习笔试面试题解析 
- [牛客网-讨论区](https://www.nowcoder.com/discuss?type=2&order=0)

### 发布站点
- [算法/NLP/深度学习/机器学习面试笔记](https://zhuanlan.zhihu.com/p/41515995) - 知乎
- [算法/NLP/深度学习/机器学习面试笔记](https://www.jianshu.com/p/55b0703aa1ad) - 简书 
- [算法/NLP/深度学习/机器学习面试笔记](https://blog.csdn.net/imhuay/article/details/81490564) - CSDN博客 
- [GitHub 上整理的深度学习/机器学习面试笔记](https://www.v2ex.com/t/473047) - V2EX 