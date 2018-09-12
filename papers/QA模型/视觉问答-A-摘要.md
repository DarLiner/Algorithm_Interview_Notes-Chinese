VQA论文笔记
===

Index
---
<!-- TOC -->

- [[2016].Making the V in VQA Matter（CVPR2017）](#2016making-the-v-in-vqa-mattercvpr2017)

<!-- /TOC -->


### [2016].Making the V in VQA Matter（CVPR2017）
> [1612.00837] [Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering](https://arxiv.org/abs/1612.00837) 
- 前人的一些工作表明：**模型在同时处理图像与文本信息时存在偏见**（Bias）。因为文本具有较强的**先验**（Prior），导致模型往往做出决策时偏向于文本提供的信息，而忽略了视觉信息。
  > 相关论文：
  >> 1.Analyzing the Behavior of Visual Question Answering Models;<br/> 
  >> 2.Simple Baseline for Visual Question Answering——本文仅使用简单的词袋模型就得到了接近复杂深度模型的性能

- 本文构造了一个平衡数据集：通过加入**互补图像**，使**每个问题**对应于一对**不同答案**的图像；
- 数据集地址：http://visualqa.org/；
- 本文在平衡数据集上对一些最先进的 **VQA 模型**做了基准测试；
- 利用新的数据集，可以构建一种新的可解释模型。该模型除了能对给定图像返回答案，还提供了一个基于**反例**的解释。具体来说，即模型对一副与原始图像相似的图像，给出了不同的答案。