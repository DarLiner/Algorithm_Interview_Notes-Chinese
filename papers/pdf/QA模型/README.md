QA模型-论文收录说明
===

Index
---
<!-- TOC -->

- [[2018]-QuAC数据集](#2018-quac数据集)
- [](#)

<!-- /TOC -->

## [2018]-QuAC数据集
> [PaperWeekly](https://mp.weixin.qq.com/s/iGkUVi7pKSiSZhaFnCWBHg) 

- 本文提出了一个基于上下文的机器阅读理解数据集 QuAC，该数据集存在两类人群：Student 和 Teacher。Student 依次提出一系列自由式的问题，而 Teacher 进行回答，该回答是基于文章内部的片段产生的。不同于以往的机器阅读理解数据集，该数据集存在以下特点： 

  1. 问题是开放式的，也就是说问题的答案不一定存在于文章的片段中。因此 Student 在提问前不知道是否能够被回答；

  2. Teacher 的回答必需基于文章内部的片段，不存在自由片段（游离于文章内容的片段）；

  3. 对话终止的条件包括：从开始对话到现在，(a). 已经有 12 个问题被回答了；(b). Student 和 Teacher 中的某一位主动提出结束对话；(c). 有两个问题不能够被回答。 

- 论文采用了 Pretrained InferSent，Feature-rich logistic regression，BiDAF++ 以及 BiDAF++ w/ x-ctx 作为基准算法，选用 HEQQ，HEQD 和 F1 等作为效果度量指标，进行了一系列实验。实验结果表明，目前的基准算法得到的最好结果，相较于人工判断的效果还存在很大提升空间。

- 论文地址：https://arxiv.org/abs/1808.07036
- 数据集：http://quac.ai/


## 