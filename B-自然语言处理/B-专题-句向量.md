专题-句向量（Sentence Embedding）
===

Reference
---
- [The Current Best of Universal Word Embeddings and Sentence Embeddings](https://medium.com/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a) 

Index
---
<!-- TOC -->

- [基线模型](#基线模型)
  - [词袋模型（BoW）](#词袋模型bow)
  - [基于词向量的词袋模型](#基于词向量的词袋模型)
  - [基于 RNN](#基于-rnn)
  - [基于 CNN](#基于-cnn)
- [无监督模型](#无监督模型)
  - [[2018] Concatenated Power Mean 模型](#2018-concatenated-power-mean-模型)
  - [[2017] SIF 加权模型](#2017-sif-加权模型)
  - [[2015] Skip-Thought Vector](#2015-skip-thought-vector)
- [有监督模型](#有监督模型)
  - [Self-Attention](#self-attention)
- [参考文献](#参考文献)

<!-- /TOC -->


## 基线模型

### 词袋模型（BoW）
- 单个词的 One-Hot 表示
- 基于频数的词袋模型
- 基于 TF-IDF 的词袋模型

### 基于词向量的词袋模型
- **均值模型**
  <div align="center"><a href="http://www.codecogs.com/eqnedit.php?latex=\fn_jvn&space;s=\frac{1}{N}\sum_{i=1}^N&space;v_i"><img src="../_assets/公式_2018091402442.png" height="" /></a></div>

  > 其中 `v_i` 表示维度为 `d` 的词向量，均值指的是对所有词向量**按位求和**后计算每一维的均值，最后 `s` 的维度与 `v` 相同。

- **加权模型**
  <div align="center"><a href="http://www.codecogs.com/eqnedit.php?latex=\fn_jvn&space;s=\sum_{i=1}^N&space;\alpha_i\cdot&space;v_i"><img src="../_assets/公式_2018091402658.png" height="" /></a></div>

  > 其中 `α` 可以有不同的选择，但一般应该遵循这样一个准则：**越常见的词权重越小**
  >> [[2017] SIF 加权模型](#2017-sif-加权模型)

### 基于 RNN
- 以最后一个隐状态作为整个句子的 Embedding
  <div align="center"><img src="../_assets/TIM截图20180914013219.png" height="" /></div>

- 基于 RNN 的 Sentence Embedding 往往用于特定的有监督任务中，**缺乏可迁移性**，在新的任务中需要重新训练；
- 此外，由于 RNN 难以并行训练的缺陷，导致开销较大。


### 基于 CNN
- 卷积的优势在于提取**局部特征**，利用 CNN 可以提取句子中类似 n-gram 的局部信息；
- 通过整合不同大小的 n-gram 特征作为整个句子的表示。

  <div align="center"><img src="../_assets/TIM截图20180914013449.png" height="" /></div>


## 无监督模型

### [2018] Concatenated Power Mean 模型
> [4]
- 本文是均值模型的一种推广；通过引入“幂均值”（Power Mean）来捕捉序列中的其他信息；
- 记句子 `s=(w_1, w_2, ..., w_n)`
  <div align="center"><a href="http://www.codecogs.com/eqnedit.php?latex=\fn_cm&space;\large&space;s_p=\left&space;(&space;\frac{x_1^p&plus;\cdots&plus;x_n^p}{n}&space;\right&space;)^{\frac{1}{p}},\quad&space;p\in\mathbb{R}\cup\{\pm\infty\}"><img src="../_assets/公式_20180914232209.png" height="" /></a></div>

  > 普通的均值模型即 `p=1` 时的特例；特别的，本文使用 `±∞` 表示 `max` 和 `min`

- 本文通过**拼接**的方式来保留不同 `p` 的信息
  <div align="center"><a href="http://www.codecogs.com/eqnedit.php?latex=\fn_jvn&space;\large&space;s=[s_{p_1};s_{p_2};...;s_{p_k}]"><img src="../_assets/公式_20180914232558.png" height="" /></a></div>

- 特别的，本文在实验时，加入了 `max`、`min` 等


### [2017] SIF 加权模型
- **文献 [1]** 提出了一个简单但有效的**加权词袋模型 SIF** (**Smooth Inverse Frequency**)，其性能超过了简单的 RNN/CNN 模型

- **SIF** 的计算分为两步：<br/>
  **1）** 对句子中的每个词向量，乘以一个权重 `a/(a+p_w)`，其中 `a` 是一个常数（原文取 `0.0001`），`p_w` 为该词的词频；对于出现频率越高的词，其权重越小；<br/>
  **2）** 计算**句向量矩阵**的第一个主成分 `u`，让每个句向量减去它在 `u` 上的投影（类似 PCA）；
    
- **完整算法描述**
  <div align="center"><img src="../_assets/TIM截图20180914010334.png" height="" /></div>

<!--  

  <details><summary><b>Numpy 示例（点击展开）</b></summary>

  ```python
  
  ```
  
  </details>

-->


### [2015] Skip-Thought Vector
> [2]
- 给定一个三元组 `s_{i-1}, s_i, s_{i+1}` 表示 3 个连续的句子。
- 模型使用 Encoder-Decoder 框架；
- 训练时，由 Encoder 对 `s_i` 进行编码；然后分别使用两个 Decoder 生成前一句 `s_{i-1}` 和下一句 `s_{i+1}`
  <div align="center"><img src="../_assets/TIM截图20180914133101.png" height="" /></div>

  - **Encoder**
    <div align="center"><img src="../_assets/TIM截图20180914151435.png" height="" /></div>
  
  - **Decoder**
    <div align="center"><img src="../_assets/TIM截图20180914151535.png" height="" /></div>
    
    > 其中 `h_i` 为 Encoder 的输出，即表示 `s_i` 的 Sentence Embedding
  - **Decoder** 可以看作是以 **Encoder** 输出为条件的**神经语言模型**
    <div align="center"><img src="../_assets/TIM截图20180914151641.png" height="" /></div>

    > 语言模型，`v` 表示词向量

  - **目标函数**
    <div align="center"><img src="../_assets/TIM截图20180914152110.png" height="" /></div>
    





## 有监督模型

### Self-Attention
> [3]

- 本文提出使用**二维矩阵**作为句子表征，矩阵的行表示在句子不同位置的关注度，以解决句子被压缩成一维向量时的信息损失。
  <div align="center"><img src="../_assets/TIM截图20180914153455.png" height="" /></div>



## 参考文献
- [1] A Simple but Tough-to-Beat Baseline for Sentence Embeddings, ICLR 2016.
- [2] Skip-Thought Vectors, NIPS 2015.
- [3] A Structured Self-attentive Sentence Embedding, ICLR 2017.
- [4] 