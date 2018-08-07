算法/NLP/深度学习/机器学习面试笔记
===

深度学习/机器学习面试问题整理，想法来源于这个[仓库](https://github.com/elviswf/DeepLearningBookQA_cn).
- 该仓库整理了“花书”《深度学习》中的一些常见问题，其中部分偏理论的问题没有收录，如有需要可以浏览原仓库。

此外，还包括我看到的所有机器学习/深度学习面经中的问题。除了其中 DL/ML 相关的，其他与算法岗相关的计算机知识也会记录。

但是不会包括如前端/测试/JAVA/Android等岗位中有关的问题。

## RoadMap
- [数学](./数学)
  - [微积分的本质](./数学/微积分的本质.md)
  - [深度学习的核心](./数学/深度学习的核心.md)
- [自然语言处理](./自然语言处理)
  - [词向量](./自然语言处理/专题-词向量.md)
  - [序列建模](./自然语言处理/专题-序列建模.md) TODO
  - [工具库](./自然语言处理/专题-工具库.md) TODO
- [机器学习-深度学习](./机器学习-深度学习)
  - 公共基础
    - 背景知识
    - 损失函数
  - 深度学习
    - [深度学习基础](./机器学习-深度学习/DL专题-深度学习基础.md)
    - [《深度学习》整理](./机器学习-深度学习/DL专题-《深度学习》整理.md)
    - [CNN专题](./机器学习-深度学习/DL专题-CNN.md)
  - 机器学习
    - [机器学习算法](./机器学习-深度学习/ML专题-机器学习算法.md)
    - [机器学习实践](./机器学习-深度学习/ML专题-机器学习实践.md)
- [算法](./算法)
  - [题解-剑指Offer](./算法/题解-剑指Offer.md)
- [编程语言](./编程语言)
  - [Cpp专题-基础知识](./编程语言/Cpp专题-基础知识.md)
  - [Cpp专题-左值与右值](./编程语言/Cpp专题-左值与右值.md)
- [笔试面经](./笔试面经)
- [项目经验](./项目经验)
- 各公司[招聘要求](./招聘要求.md)
<!-- 
- [深度学习](./深度学习)
  - [深度学习实践](./项目经验/README.md)（项目经验）
  - [“花书”《深度学习》整理](./深度学习/“花书”《深度学习》整理.md)
    > 顺序比较乱，很多相关问题也没放在一起，目前正在做进一步整理
- [机器学习](./机器学习)
- [OJ 记录](https://github.com/imhuay/Algorithm_for_Interview-Chinese)
  - [必背算法](https://github.com/imhuay/Algorithm_for_Interview-Chinese/tree/master/Algorithm_for_Interview/_必背算法)
  - [C++ 回顾](https://github.com/imhuay/Algorithm_for_Interview-Chinese/tree/master/Algorithm_for_Interview/_cpp回顾)
    - [传统OJ IO模板](https://github.com/imhuay/Algorithm_for_Interview-Chinese/blob/master/Algorithm_for_Interview/_cpp回顾/IO模板.hpp)
    - [STL 容器速览](https://github.com/imhuay/Algorithm_for_Interview-Chinese/tree/master/Algorithm_for_Interview/_Cpp回顾/STL容器)
- [算法](./算法)
- [数学](./数学)
  - [微积分的本质](./数学/微积分的本质.md)
  - [深度学习的核心](./数学/深度学习的核心.md)
- [笔试面经](./笔试面经) 
-->


## 必备清单
- [深度学习](./深度学习/README.md)
  - [反向传播算法](./深度学习/README.md#反向传播算法)
  - [梯度下降法](#梯度下降法)
  - [深度学习实践](./项目经验/README.md)（项目经验）
  - 相关代码 TODO
- [机器学习算法](./机器学习/README.md)
  - [逻辑斯蒂回归](./机器学习/README.md#逻辑斯蒂回归)
  - [支持向量机](./机器学习/README.md#支持向量机)
  - [AdaBoost 算法](./机器学习/README.md#adaboost-算法)
  - [GBDT 梯度提升决策树](./机器学习/README.md#梯度提升决策树-gbdt)
  - 相关代码 TODO
- 计算机基础
  - [必背算法](https://github.com/imhuay/Algorithm_for_Interview-Chinese/tree/master/Algorithm_for_Interview/_必背算法)
  - Python 常识 TODO
  - C++ 常识 TODO


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