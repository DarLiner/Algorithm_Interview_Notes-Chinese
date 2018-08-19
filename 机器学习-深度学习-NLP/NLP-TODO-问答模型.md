NLP-问答模型
===

问答模型简述
---
- 问答模型可以概括为 Seq2Seq 模型，但通常会比较复杂
  > [SQuAD文本理解挑战赛十大模型解读](https://www.sohu.com/a/216468593_500659)_搜狐科技_搜狐网 

Index
---
<!-- TOC -->

- [QANet 模型](#qanet-模型)
- [BiDAF 模型](#bidaf-模型)
  - [模型结构](#模型结构)
  - [参数规模 TODO](#参数规模-todo)
  - [论文/代码/参考](#论文代码参考)
- [decaNLP 模型](#decanlp-模型)
  - [论文/代码](#论文代码)

<!-- /TOC -->

## QANet 模型

## BiDAF 模型

### 模型结构
- 模型共 6 层
  <div align="center"><img src="../assets/TIM截图20180808232139.png" height="" /></div>

- **嵌入层**（char + word）
  - 记**材料**（context）和**查询**（query）分别为
    <div align="center"><a href="http://www.codecogs.com/eqnedit.php?latex=\{x_1,x_2,...,x_T\}\quad\text{and}\quad\{q_1,q_2,...,q_J\}"><img src="../assets/公式_20180810211954.png" height="" /></a></div>

  - 词向量使用预训练的 GloVe，OOV 可训练
  - 字向量由 CharCNN 训练
    > NLP-词向量#[CharCNN](./NLP-词向量.md#charcnn-字向量)
    - CharCNN 负责将**每个单词**映射到一个**向量**
    - 实际为单词中每个**字符向量**的叠加平均，具体由一个**卷积层和池化层**实现
    - 字符嵌入层实际上生成的也是一个词向量，但是使用的字符信息，而不是上下文信息
  - 每个单词的表示由词向量和字向量拼接而成，然后经过两层 highway 得到 centext vector 
- **Word Embedding Layer**
  - 使用预训练好的 GloVe 词向量
    > NLP-词向量#[GloVe](./NLP-词向量.md#glove)
  - 默认词向量和字符向量的维度相同
  - **输入 Embedding** 由 Character Embedding 和 Word Embedding **拼接**而成
- **Contextual Embedding Layer**
  - 利用周围词语的语境线索（contextual cues）来**改善**（refine）词的嵌入。
  - 双向 LSTM
- **Attention Flow Layer**
  - 耦合（couple）材料与问题，并为上下文中的**每个词**生成一组**查询感知特征向量**（query-aware feature vectors）
- **Modeling Layer**
  - 使用 RNN 扫描（scan）上下文
- **Output Layer**
  - 产生问题的答案

### 参数规模 TODO


### 论文/代码/参考
- [1611.01603] [Bidirectional Attention Flow for Machine Comprehension](https://arxiv.org/abs/1611.01603) 
- [allenai/bi-att-flow](https://github.com/allenai/bi-att-flow/tree/dev) (TF1.0+) - GitHub
- BiDAF https://allenai.github.io/bi-att-flow/
- [论文笔记 - Bi-Directional Attention Flow for Machine Comprehension](http://www.shuang0420.com/2018/04/01/论文笔记%20-%20Bi-Directional%20Attention%20Flow%20for%20Machine%20Comprehension/) | 徐阿衡 

## decaNLP 模型
> [一个模型搞定十大自然语言任务](https://www.toutiao.com/i6569393480089469454)｜论文+代码 

### 论文/代码
- [1806.08730] [The Natural Language Decathlon: Multitask Learning as Question Answering](https://arxiv.org/abs/1806.08730) 
- [salesforce/decaNLP](https://github.com/salesforce/decaNLP) - GitHub
