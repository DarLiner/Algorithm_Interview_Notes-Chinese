**RodeMap**
---
<!-- TOC -->

- [评价机制](#评价机制)
  - [困惑度 (Perplexity, PPX)](#困惑度-perplexity-ppx)
    - [语言模型中的 PPX](#语言模型中的-ppx)
  - [BLEU](#bleu)
  - [ROUGE](#rouge)
- [词向量](#词向量)
- [文本匹配](#文本匹配)
- [阅读理解/问答](#阅读理解问答)
  - [self-attention 实现机制](#self-attention-实现机制)
  - [Reference](#reference)
- [NLP 实践](#nlp-实践)
  - [seq2seq 中 scheduled sampling 怎么做](#seq2seq-中-scheduled-sampling-怎么做)
  - [RL 中的 reward 机制](#rl-中的-reward-机制)
  - [Action 怎么实现的](#action-怎么实现的)
  - [NLP 怎么做数据增强](#nlp-怎么做数据增强)

<!-- /TOC -->

# 评价机制

## 困惑度 (Perplexity, PPX)
> [Perplexity](https://en.wikipedia.org/wiki/Perplexity) - Wikipedia
- 在信息论中，perplexity 用于度量一个**概率分布**或**概率模型**预测样本的好坏程度
  > 机器学习/[信息论](../机器学习/README.md#信息论) 

<h3>基本公式</h3>

- **概率分布**（离散）的困惑度
  <div align="center"><a href="http://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;{\displaystyle&space;2^{H(p)}=2^{-\sum&space;_{x}p(x)\log&space;_{2}p(x)}}"><img src="../assets/公式_20180728195601.png" /></a></div>
  
  > 其中 `H(p)` 即**信息熵**

- **概率模型**的困惑度
  <div align="center"><a href="http://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;{\displaystyle&space;b^{-{\frac&space;{1}{N}}\sum&space;_{i=1}^{N}\log&space;_{b}q(x_{i})}}"><img src="../assets/公式_20180728201614.png" /></a></div>

  > 通常 `b=2`
  
- **指数部分**也可以是**交叉熵**的形式，此时困惑度相当于交叉熵的指数形式
  <div align="center"><a href="http://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;2^{H(\tilde{p},q)}&space;=&space;2^{-\sum_x\tilde{p}(x)\log_{2}q(x)}"><img src="../assets/公式_20180728202629.png" /></a></div>

  > 其中 `p~` 为**测试集**中的经验分布——`p~(x) = n/N`，其中 `n` 为 x 的出现次数，N 为测试集的大小

### 语言模型中的 PPX
- 在 **NLP** 中，困惑度常作为**语言模型**的评价指标
  <div align="center"><a href="http://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\begin{aligned}&space;\mathrm{PPX}(W_{test})&space;&=2^{-\sum_{i=1}^{|V|}\tilde{p}(w_i)\log_{2}q(w_i)}\\&space;&=2^{-\sum_{i=1}^{|V|}\frac{\mathrm{cnt}(w_i)}{N}\log_{2}q(w_i)}&space;\end{aligned}"><img src="../assets/公式_20180728205525.png" /></a></div>

- 直观来说，就是下一个**候选词数目**的期望值——

  如果不使用任何模型，那么下一个候选词的数量就是整个词表的数量；通过使用 `bi-gram`语言模型，可以将整个数量限制到 `200` 左右

## BLEU
> [一种机器翻译的评价准则——BLEU](https://blog.csdn.net/qq_21190081/article/details/53115580) - CSDN博客 
- 机器翻译评价准则
- 计算公式
  <div style="position:relative;left:25%"><img src="../assets/TIM截图20180728212444.png" height="" /></div>
  <div style="position:relative;left:25%"><img src="../assets/TIM截图20180728212554.png" height="" /></div>

  > 其中 `c` 为生成句子的长度；`r` 为参考句子的长度

- 为了计算方便，会加一层 `log` 
  <div align="center"><img src="../assets/TIM截图20180728213300.png" height="" /></div>
  
  > 通常 `N=4, w_n=1/4` 

## ROUGE
> [自动文摘评测方法：Rouge-1、Rouge-2、Rouge-L、Rouge-S](https://blog.csdn.net/qq_25222361/article/details/78694617) - CSDN博客 
- 一种机器翻译/自动摘要的评价准则

> [BLEU，ROUGE，METEOR，ROUGE-浅述自然语言处理机器翻译常用评价度量](https://blog.csdn.net/joshuaxx316/article/details/58696552) - CSDN博客 

# 词向量

# 文本匹配

# 阅读理解/问答

## self-attention 实现机制


## Reference
- [近期有哪些值得读的QA论文？](https://www.jiqizhixin.com/articles/2018-06-11-14)| 专题论文解读 | 机器之心 

# NLP 实践
## seq2seq 中 scheduled sampling 怎么做

## RL 中的 reward 机制

## Action 怎么实现的

## NLP 怎么做数据增强
- 利用 NMT 做双向翻译——将语言A 翻译到其他语言，再翻译回语言 A

  这个过程相当于对样本进行了改写，使得训练样本的数量大大增加
- QANet 中的做法：
  <div align="center"><img src="../assets/TIM截图20180724200255.png" height="" /></div>

  - 对材料中每个句子通过翻译引擎得到`k`句法语候选，然后将每句法语转回英语，得到`k^2`个改写的句子，从中随机选择一句作为

  - 改写后答案的位置也可能改变，如何寻找**新答案的位置**？

    具体到 SQuAD 任务就是 (d,q,a) -> (d’, q, a’)，问题不变，对文档 d 翻译改写，由于改写后原始答案 a 现在可能已经不在改写后的段落 d’ 里了，所以需要从改写后的段落 d’ 里抽取新的答案 a’，采用的方法是计算 d’ 里每个单词和原始答案里 start/end words 之间的 **character-level 2-gram score**，分数最高的单词就被选择为新答案 a’ 的 start/end word。
    > 中文没有里面没有 character-level 2-gram，可以考虑词向量之间的相似度

    