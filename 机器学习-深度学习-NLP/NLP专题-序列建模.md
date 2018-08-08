专题-序列建模
===

Reference
---
- 自然语言处理之序列模型 - 小象学院
- [从循环到卷积，探索序列建模的奥秘](https://mp.weixin.qq.com/s/f0sv7c-H5o5L_wy2sUonUQ) - 机器之心

Index
---
<!-- TOC -->

- [NLP 概述](#nlp-概述)
  - [解决 NLP 问题的一般思路](#解决-nlp-问题的一般思路)
  - [NLP 的历史进程](#nlp-的历史进程)
  - [Seq2Seq 模型](#seq2seq-模型)
- [序列建模](#序列建模)
  - [RNN](#rnn)
    - [RNN 的设计模式以及相应的递推公式（3） ToCompletion](#rnn-的设计模式以及相应的递推公式3-tocompletion)
    - [完整的 RNN 递推公式](#完整的-rnn-递推公式)
  - [LSTM](#lstm)
  - [RNN](#rnn-1)

<!-- /TOC -->

# NLP 概述

## 解决 NLP 问题的一般思路
```tex
这个问题人类可以做好么？
  - 可以 -> 记录自己的思路 -> 设计流程让机器完成你的思路
  - 很难 -> 尝试从计算机的角度来思考问题
```

## NLP 的历史进程
- **规则系统**
  - 正则表达式/自动机
  - 规则是固定的
  - **搜索引擎**
    ```tex
    “豆瓣酱用英语怎么说？”
    规则：“xx用英语怎么说？” => translate(XX, English)
    
    “我饿了”
    规则：“我饿（死）了” => recommend(饭店，地点)
    ```
- **概率系统**
  - 规则从数据中**抽取**
  - 规则是有**概率**的
  - 概率系统的一般**工作方式**
    ```tex
    流程设计
      收集训练数据
        预处理
          特征工程
            分类器（机器学习算法）
              预测
                评价
    ```
    <div align="center"><img src="../assets/TIM截图20180807203305.png" height="200" /></div>
    
    - 最重要的部分：数据收集、预处理、特征工程
  - 示例
    ```tex
    任务：
      “豆瓣酱用英语怎么说” => translate(豆瓣酱，Eng)

    流程设计（序列标注）：
      子任务1： 找出目标语言 “豆瓣酱用 **英语** 怎么说”
      子任务2： 找出翻译目标 “ **豆瓣酱** 用英语怎么说”

    收集训练数据：
      （子任务1）
      “豆瓣酱用英语怎么说”
      “茄子用英语怎么说”
      “黄瓜怎么翻译成英语”
    
    预处理：
      分词：“豆瓣酱 用 英语 怎么说”

    抽取特征：
      （前后各一个词）
      0 茄子：    < _ 用
      0 用：      豆瓣酱 _ 英语
      1 英语：    用 _ 怎么说
      0 怎么说：  英语 _ >

    分类器：
      SVM/CRF/HMM/RNN

    预测：
      0.1 茄子：    < _ 用
      0.1 用：      豆瓣酱 _ 英语
      0.7 英语：    用 _ 怎么说
      0.1 怎么说：  英语 _ >

    评价：
      准确率
    ```
- 概率系统的优/缺点
  - `+` 规则更加贴近于真实事件中的规则，因而效果往往比较好
  - `-` 特征是由专家/人指定的；
  - `-` 流程是由专家/人设计的；
  - `-` 存在独立的**子任务**

- **深度学习**
  - 深度学习相对概率模型的优势
    - 特征是由专家指定的 `->` 特征是由深度学习自己提取的
    - 流程是由专家设计的 `->` 模型结构是由专家设计的
    - 存在独立的子任务 `->` End-to-End Training

## Seq2Seq 模型
- 大部分自然语言问题都可以使用 Seq2Seq 模型解决
  <div align="center"><img src="../assets/TIM截图20180807210029.png" height="200" /></div>

- **“万物”皆 Seq2Seq**
  <div align="center"><img src="../assets/TIM截图20180807210133.png" height="300" /></div>


# 序列建模
> [从循环到卷积，探索序列建模的奥秘](https://mp.weixin.qq.com/s/f0sv7c-H5o5L_wy2sUonUQ) - 机器之心
- 序列建模即将一个**输入/观测**序列映射到一个**输出/标记**序列
  > 《统计学习方法》中称之为标注问题
- 在**传统机器学习**方法中，常用的模型有：隐马尔可夫模型（HMM），条件随机场（CRF）等
  > 机器学习专题 TODO
- 在**深度学习领域**的很长一段时间里，RNN/LSTM 都是序列建模的默认配置。
  > 《深度学习》中直接使用“序列建模：循环和递归网络”作为章节名
- 最近，CNN 开始在序列建模领域流行，一个关键想法是——在一维时间序列上使用**一维卷积**运算
  <div align="center"><img src="../assets/TIM截图20180808105242.png" height="" /></div>

  > [CNN for Sentence Classification](https://arxiv.org/abs/1408.5882) (Kim, 2014)

## RNN
- 循环神经网络本质上是一个递推函数
  <div align="center"><a href="http://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;s^{(t)}=f(s^{(t-1)};\theta)"><img src="../assets/公式_20180808110452.png" height="" /></a></div>

- 考虑隐藏状态和输入
  <div align="center"><a href="http://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;h^{(t)}=f({\color{Red}h^{(t-1)},x^{(t)}};\theta)"><img src="../assets/公式_20180808110741.png" height="" /></a></div>

- RNN 的计算图（无输出单元）
  <div align="center"><img src="../assets/TIM截图20180808110903.png" height="120" /></div>

### RNN 的设计模式以及相应的递推公式（3） ToCompletion
> 《深度学习》 10.2 循环神经网络 - RNN 几种不同的设计模型（3）
- 根据

### 完整的 RNN 递推公式
- 加入输出单元
  <div align="center"><a href="http://www.codecogs.com/eqnedit.php?latex=\large&space;\begin{aligned}&space;\textbf{\emph{a}}^{(t)}&=\textbf{\emph{b}}&plus;\textbf{\emph{Wh}}^{(t-1)}&plus;\textbf{\emph{Ux}}^{(t)}\\&space;\textbf{\emph{h}}^{(t)}&=\tanh(\textbf{\emph{a}}^{(t)})\\&space;\textbf{\emph{o}}^{(t)}&=\textbf{\emph{c}}&plus;\textbf{\emph{Vh}}^{(t)}\\&space;\hat{\textbf{\emph{y}}}^{(t)}&=\mathrm{softmax}(\textbf{\emph{o}}^{(t)})&space;\end{aligned}"><img src="../assets/公式_20180808114308.png" height="" /></a></div>

- 完整的计算图
  <div align="center"><img src="../assets/TIM截图20180808111835.png" height="" /></div>
  
  > 一般来说，有两种 RNN 的基本结构：Elman network 和 Jordan network；目前深度学习领域通常所说的 RNN 指的是前者
  > <div align="center"><img src="../assets/TIM截图20180808114753.png" height="" /></div>
  >
  >> [Recurrent neural network](https://en.wikipedia.org/wiki/Recurrent_neural_network#Elman_networks_and_Jordan_networks) - Wikipedia 
  

## LSTM

## RNN