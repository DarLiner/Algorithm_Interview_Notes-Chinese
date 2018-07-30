局部敏感哈希
===

Index
---
<!-- TOC -->

- [什么是局部敏感哈希](#什么是局部敏感哈希)
  - [LSH 的特点](#lsh-的特点)
  - [LSH 的应用场景](#lsh-的应用场景)

<!-- /TOC -->


## 什么是局部敏感哈希
> [Locality-sensitive hashing](https://en.wikipedia.org/wiki/Locality-sensitive_hashing) - Wikipedia 
- **局部敏感哈希**（Locality-Sensitive Hashing, LSH）指的是一类特殊的 Hash 函数——它们能使相似的高维数据在经过**降维**后依然保持一定的相似性。
- LSH 本质上是一种利用 Hash 进行快速**相似度计算**的算法。
- 具体来说，相似的高维数据经过 LSH 处理后，会以较高的**概率**进入相同的“**桶(bucket)**”中，从而达到分类相似数据的目的。

### LSH 的特点
- **速度快**——利用 Hash 特点，LSH 特别用于处理**海量数据**的相似性计算；
  <!-- - **机械相似性**；相似性算法通常分为“机械相似性”与“语义相似性” -->

### LSH 的应用场景