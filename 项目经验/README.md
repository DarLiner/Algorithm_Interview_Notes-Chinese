**RoadMap**
---
<!-- TOC -->

- [微博用户画像](#微博用户画像)
  - [任务描述](#任务描述)
  - [思路](#思路)
  - [数据预处理](#数据预处理)
  - [特征工程](#特征工程)
  - [模型](#模型)
- [词向量 TODO](#词向量-todo)
  - [embedding 维度的选择](#embedding-维度的选择)
- [讯飞中文阅读理解评测](#讯飞中文阅读理解评测)
  - [任务说明](#任务说明)
  - [思路](#思路-1)
  - [模型](#模型-1)
    - [Encoder 部分](#encoder-部分)
    - [Decoder 部分](#decoder-部分)
- [CIPS-SOGOU 事实类问答评测](#cips-sogou-事实类问答评测)
  - [任务描述](#任务描述-1)
  - [思路](#思路-2)
  - [模型](#模型-2)
    - [为什么使用 CNN 代替 RNN](#为什么使用-cnn-代替-rnn)
    - [常见的卷积结构](#常见的卷积结构)
      - [基本卷积](#基本卷积)
      - [转置卷积](#转置卷积)
      - [空洞卷积](#空洞卷积)
      - [可分离卷积](#可分离卷积)
    - [门卷积](#门卷积)
      - [门卷积是如何防止梯度消失的](#门卷积是如何防止梯度消失的)
    - [如何构造位置向量](#如何构造位置向量)

<!-- /TOC -->

# 微博用户画像
> [2017.4-5]; 自然语言处理课程项目
>
> 本来是上届师兄参加的比赛，这次作为我们 NLP 课程的项目

## 任务描述
- 利用给定的新浪微博数据（包括用户个人信息、用户微博文本以及用户粉丝列表，详见数据描述部分），进行微博用户画像，具体包括以下三个任务：
  - 任务1：推断用户的年龄（共3个标签：-1979/1980-1989/1990+）
  - 任务2：推断用户的性别（共2个标签：男/女）
  - 任务3：推断用户的地域（共8个标签：东北/华北/华中/华东/西北/西南/华南/境外）

## 思路
- 形式上是一个多分类问题，**特征工程**是关键
> 当时刚刚接触机器学习，没有这个意识，大部分时间都用来熟悉模型了

## 数据预处理
- **停用词**处理
  - 对词频排序，手动筛选停用词
  <!-- - 只处理最基础的标点和特殊符号，尽量保留词信息 -->
- 对未分词数据用 jieba **分词**
  - 关闭“新词发现”
  - 扩充词典（网络收集，微博语料新词发现+人工筛选）
- 提取**表情符**
  - 默认系列：`[微笑]`等
- **缺失值**处理
  - 根据缺失数据的比例决定填充还是丢弃
- 构建**地域映射表**（任务3）

## 特征工程
- 基本特征
  - 粉丝数量，微博数量，发布平台（来源），发布时间段（离散化），...
- 统计特征
  - 平均每条微博的评论/转发人数，
  - 最大/最小的评论/转发数，
  - 博文中出现的地区数量，
  - 每个时间段的发帖量（每小时，早中晚），...
- **交叉特征**/特征组合
  - 特征组合的种类
    - `[A x A]` —— 对单个特征的值求平方形成的组合特征。
    - `[A X B]` —— 将两个特征的值相乘形成的组合特征。
    - `[A x B x C x ...]` —— 将多个特征的值相乘形成的组合特征。
  - 组合 Ont-Hot 编码
    - 组合两个 `1*3` 的编码可以得到一个 `1*9` 的编码
    - 比如组合**发帖时间**和**发布平台**
    
      `[早, 中, 晚]` x `[手机, 电脑, 平板]` -> `[早+手机, 中+手机, 晚+手机, 早+电脑, ...]`
- 词向量 word2vec
  - [词向量维数的选择](#embedding-维度的选择)

**特征选择**
- 使用 XGBoost 进行特征选择
  ```Python
  params = {...}
  xgtrain = xgb.DMatrix(X, label=y)
  bst = xgb.train(params, xgtrain)
  importance = bst.get_fscore()
  importance = sorted(importance.items(), key=lambda x: x[1])
  importance_df = pd.DataFrame(importance, columns=['feature', 'fscore'])
  importance_df.plot(kind='barh', x='feature', y='fscore')
  ```

## 模型
- **SVM** (baseline)
  > ../机器学习#[支持向量机](../机器学习/README.md#支持向量机)
- **GBDT**(XGBoost库)
  > ../机器学习#[GBDT](../机器学习/README.md#梯度提升决策树-gbdt) & #[XGBoost](../机器学习/README.md#xgboost-算法)


# 词向量 TODO
## embedding 维度的选择
- 经验公式 `embedding_size = n_categories ** 0.25`
- 在大型语料上训练的词向量维度通常会设置的更大一些，比如 `100~300`
  > 如果根据经验公式，是不需要这么大的，比如 200W 词表的词向量维度只需要 `200W ** 0.25 ≈ 37`


# 讯飞中文阅读理解评测
> [2017.7-9]; [第一届“讯飞杯”中文机器阅读理解评测 - CMRC 2017](http://www.hfl-tek.com/cmrc2017)
- 数据集和任务格式是哈工大去年提出的。
  工大去年的论文[`[1607.02250]`](https://arxiv.org/abs/1607.02250)研究了另一个相同形式的数据集。<!-- 是用通用的阅读理解模型做的。 -->

## 任务说明
> [任务介绍 - CMRC 2017](http://www.hfl-tek.com/cmrc2017/task/)
- 以**单个词**为答案的填空类问题、用户提问类问题。答案会在上下文中出现。

- 数据格式
  ```
  篇章：由若干个连续的句子组成的一个文本段落，但文中缺少某一个词，标记为 XXXXX
  问题：缺少的词 XXXXX 所在的句子
  答案：缺少的词 XXXXX
  ```

- 示例
  - 填空类问题
    ```
    1 ||| 工商 协进会 报告 ， 12月 消费者 信心 上升 到 78.1 ， 明显 高于 11月 的 72 。
    2 ||| 另 据 《 华尔街 日报 》 报道 ， 2013年 是 1995年 以来 美国 股市 表现 最 好 的 一 年 。
    3 ||| 这 一 年 里 ， 投资 美国 股市 的 明智 做法 是 追 着 “ 傻钱 ” 跑 。
    4 ||| 所谓 的 “ 傻钱 ” XXXXX ， 其实 就 是 买 入 并 持有 美国 股票 这样 的 普通 组合 。
    5 ||| 这个 策略 要 比 对冲 基金 和 其它 专业 投资者 使用 的 更为 复杂 的 投资 方法 效果 好 得 多 。
    <qid_1> ||| 所谓 的 “ 傻钱 ” XXXXX ， 其实 就 是 买 入 并 持有 美国 股票 这样 的 普通 组合 。
    ```
    答案
    ```
    <qid_1> ||| 策略
    ```
  - 提问类问题
    ```
    1 ||| 工商 协进会 报告 ， 12月 消费者 信心 上升 到 78.1 ， 明显 高于 11月 的 72 。
    2 ||| 另 据 《 华尔街 日报 》 报道 ， 2013年 是 1995年 以来 美国 股市 表现 最 好 的 一 年 。
    3 ||| 这 一 年 里 ， 投资 美国 股市 的 明智 做法 是 追 着 “ 傻钱 ” 跑 。
    4 ||| 所谓 的 “ 傻钱 ” 策略 ， 其实 就 是 买 入 并 持有 美国 股票 这样 的 普通 组合 。
    5 ||| 这个 策略 要 比 对冲 基金 和 其它 专业 投资者 使用 的 更为 复杂 的 投资 方法 效果 好 得 多 。
    <qid_1> ||| 哪 一 年 是 美国 股市 表现 最 好 的 一 年 ？
    ```
    答案
    ```
    <qid_1> ||| 2013年
    ```

## 思路
- 类似语言模型的思路：
  - 语言模型是根据前 n 个词从整个词表中预测下一个词；
  - 这里是根据上下文并从中挑选正确的答案
- 可以采用 **encoder-decoder 框架**，这里相当于 "seq2word"
- 对语言建模来说，LSTM 的效果更好；
  - 为了更好的获取全局的语义信息，可以使用多层 bi-LSTM 对上下文进行编码；

## 模型

### Encoder 部分
- Encoder 部分是一个**多层 bi-LSTM**
  - 因为语料不大，所以层数不多，实际使用的是 3 层 bi-LSTM
- 以答案断开材料，分为上文和下文；两者共享同一个 Encoder，即**参数共享**
  > 为什么共享参数？——模拟人在做完形填空的过程，上下文会共同影响结果。
- 拼接 bi-LSTM 的两段输出向量得到上下文各自的特征向量；然后对上下文的特征向量求平均作为**全局特征**。

### Decoder 部分
- 为了实现在上下文中搜索目标词，而不是在整个词表中：

  控制 Encoder 的输出向量，即全局特征与词向量的**维度相同**，然后计算其与段落中的每个词的 **cos 距离**（即内积），然后通过 softmax 得到所有词的**概率分布**。
- 除了与词向量做内积，还可以使用 LSTM 的状态向量，实验中也有类似的效果，甚至更好（原因未知）

<!-- TODO: 模型图示 -->

<h3>实现细节</h3>

- 原材料已经经过分句和分词处理，实际只是用了**分词**信息。
- 因为每个词有可能多次出现，所以需要对所有相同词的概率求和作为该词的概率。
- 未登录词的处理：所有未登录词会映射到同一个词向量，并开放训练
- 其他词向量在前 n 轮不参与训练，在 n 轮后会加入微调


# CIPS-SOGOU 事实类问答评测
> [2017.10-2018.1]; [CIPS-SOGOU问答比赛](http://task.www.sogou.com/cips-sogou_qa/)
- CIPS-SOGOU问答比赛是由中国中文信息学会（CIPS）和搜狗搜索（SOGOU SEARCH）联合主办的一项开放域的智能问答评测比赛。

## 任务描述
- 针对每个问题 q，给定与之对应的若干候选答案篇章 a1，a2，…，an，要求设计算法从候选篇章中**抽取合适的词语、短语或句子**，形成一段正确、完整、简洁的文本，作为预测答案 apred，目标是 apred 能够正确、完整、简洁地回答问题 q。
- 示例
  ```
  问题: 中国最大的内陆盆地是哪个
  答案：塔里木盆地
  材料：
    1. 中国新疆的塔里木盆地，是世界上最大的内陆盆地，东西长约1500公里，南北最宽处约600公里。盆地底部海拔1000米左右，面积53万平方公里。
    2. 中国最大的固定、半固定沙漠天山与昆仑山之间又有塔里木盆地，面积53万平方公里，是世界最大的内陆盆地。盆地中部是塔克拉玛干大沙漠，面积33.7万平方公里，为世界第二大流动性沙漠。
  ```

## 思路
- 参考论文：[[1607.06275]](https://arxiv.org/abs/1607.06275)——百度利用百度知道等资源构建一个 WebQA 数据集，数据形式与本次评测基本一致；

  论文使用的模型如下：
  <div align="center"><img src="../assets/TIM截图20180719220242.png" height="300" /></div>

<h3>结构说明</h3>

- 首先，**对问题编码**，得到问题的**特征向量** `rq`（LSTM + Attention）；
- 然后使用**问题的特征向量**和**材料的词向量**以及其他可选的**人工特征**（trick）**对材料编码**部分，得到材料的特征向量，最后送给 CRF（条件随机场）层得到答案的位置——`B, I, O` 分别表示 Begin、Inside、Outside 答案。
- 值得一提的是，论文用到了两个**共现特征**—— `Question-Evidence common word feature` 和 `Evidence-Evidence common word feature`
  - 前者表示每个在材料中的词是否在问题中出现，后者表示在该材料中出现的词是否在其他材料中出现。
  - 前者出于这一**直觉**——在问题中出现的词往往不是答案的一部分；后者则相反，在不同材料中多次出现的词很可能是答案的一部分。

## 模型
- 模型的结构与 WebQA 基本相同：
- 首先对问题编码，然后将问题编码拼接到材料中，最后当做序列标注问题处理，但做了如下改动：
  - 对问题进行编码得到**问题向量**，但是模型中 LSTM 替换为 CNN，具体用到了“膨胀卷积”与“[门卷积](#什么是门卷积)”（卷积+**门限单元**）
  - 将**问题向量**拼接到材料的**每一个词向量**中；  
    参考论文 “**Attention is all you need**” 中的 Transformer 模型，加入 **Position Embedding**（位置向量）；最后加入人工提取的**共现特征**。
  - 最后的预测任务转化为一个序列标注任务，但是不使用 CRF 而是简单的 "0/1" 标注（CRF不熟悉）

<!-- TODO: 加入模型图示 -->
<!-- 其实我们当时用的其实还是 LSTM，虽然效果还行，但是整体结构很大，加了好多来自不同模型的结构，而且不是一个人做的，所以最后整个模型很乱；这里介绍的模型实际上来自当时评测的第一名：[基于CNN的阅读理解式问答模型：DGCNN](https://kexue.fm/archives/5409) - 科学空间|Scientific Spaces  -->

### 为什么使用 CNN 代替 RNN
> [关于序列建模，是时候抛弃RNN和LSTM了](https://www.jiqizhixin.com/articles/041503) | 机器之心 [[英文原文]](https://towardsdatascience.com/the-fall-of-rnn-lstm-2d1594c74ce0)

**RNN/LSTM 本身的问题(3)**
1. RNN 需要更多的资源来训练，它和 硬件加速不匹配
    > 训练 RNN 和 LSTM 非常困难，因为计算能力受到内存和带宽等的约束。简单来说，每个 LSTM 单元需要四个仿射变换，且每一个时间步都需要运行一次，这样的仿射变换会要求非常多的内存带宽。**添加更多的计算单元很容易，但添加更多的内存带宽却很难**——这与目前的硬件加速技术不匹配，一个可能的解决方案就是让计算在存储器设备中完成。
1. RNN 容易发生**梯度消失**，即使是 LSTM
    > 在长期信息访问当前处理单元之前，需要按顺序地通过所有之前的单元。这意味着它很容易遭遇梯度消失问题；LSTM 一定程度上解决了这个问题，但 LSTM 网络中依然存在顺序访问的序列路径——直观来说，LSTM 能跳过一段信息中不太重要的部分，但如果整段信息都很重要，它依然需要完整的顺序访问，此时就跟 RNN 没有区别了。
1. **注意力机制模块**（记忆模块）的应用
    - 注意力机制模块可以同时**前向预测**和**后向回顾**。
    - **分层注意力编码器**（Hierarchical attention encoder）
    <div align="center"><img src="../assets/TIM截图20180720101423.png" height="250" /></div>

    - 分层注意力模块通过一个**层次结构**将过去编码向量**汇总**到一个**上下文向量**`C_t` ——这是一种更好的**观察过去信息**的方式（观点）
    - **分层结构**可以看做是一棵**树**，其路径长度为 `logN`，而 RNN/LSTM 则相当于一个**链表**，其路径长度为 `N`，如果序列足够长，那么可能 `N >> logN`
    > [放弃 RNN/LSTM 吧，因为真的不好用！望周知~](https://blog.csdn.net/heyc861221/article/details/80174475) - CSDN博客 

**任务角度(1)**
1. 从任务本身考虑，我认为也是 CNN 更有利，LSTM 因为能记忆比较长的信息，所以在推断方面有不错的表现（直觉）；但是在事实类问答中，并不需要复杂的推断，答案往往藏在一个 **n-gram 短语**中，而 CNN 能很好的对 n-gram 建模。

### 常见的卷积结构
> [一文了解各种卷积结构原理及优劣](https://zhuanlan.zhihu.com/p/28186857) - 知乎 &
  vdumoulin/[conv_arithmetic](https://github.com/vdumoulin/conv_arithmetic) - GitHUub

#### 基本卷积
<table style="width:100%; table-layout:fixed;">
  <tr>
    <td>No padding, no strides</td>
    <td>Arbitrary padding, no strides</td>
    <td>Half padding, no strides</td>
    <td>Full padding, no strides</td>
  </tr>
  <tr>
    <td><img width="150px" src="../assets/no_padding_no_strides.gif"></td>
    <td><img width="150px" src="../assets/arbitrary_padding_no_strides.gif"></td>
    <td><img width="150px" src="../assets/same_padding_no_strides.gif"></td>
    <td><img width="150px" src="../assets/full_padding_no_strides.gif"></td>
  </tr>
  <tr>
    <td>No padding, strides</td>
    <td>Padding, strides</td>
    <td>Padding, strides (odd)</td>
  </tr>
  <tr>
    <td><img width="150px" src="../assets/no_padding_strides.gif"></td>
    <td><img width="150px" src="../assets/padding_strides.gif"></td>
    <td><img width="150px" src="../assets/padding_strides_odd.gif"></td>
  </tr>
</table>

<!-- TODO: 更细的分类 -->

#### 转置卷积
- 转置卷积（Transposed Convolution），又称反卷积（Deconvolution）、Fractionally Strided Convolution
  > 反卷积的说法不够准确，数学上有定义真正的反卷积，两者的操作是不同的
- 转置卷积是卷积的**逆过程**，如果把基本的卷积（+池化）看做“缩小分辨率”的过程，那么转置卷积就是“**扩充分辨率**”的过程。
  - 为了实现扩充的目的，需要对输入以某种方式进行**填充**。
- 转置卷积与数学上定义的反卷积不同——在数值上，它不能实现卷积操作的逆过程。其内部实际上执行的是常规的卷积操作。
  - 转置卷积只是为了**重建**先前的空间分辨率，执行了卷积操作。
- 虽然转置卷积并不能还原数值，但是用于**编码器-解码器结构**中，效果仍然很好。——这样，转置卷积可以同时实现图像的**粗粒化**和卷积操作，而不是通过两个单独过程来完成。

<table style="width:100%; table-layout:fixed;">
  <tr>
    <td>No padding, no strides, transposed</td>
    <td>Arbitrary padding, no strides, transposed</td>
    <td>Half padding, no strides, transposed</td>
    <td>Full padding, no strides, transposed</td>
  </tr>
  <tr>
    <td><img width="150px" src="../assets/no_padding_no_strides_transposed.gif"></td>
    <td><img width="150px" src="../assets/arbitrary_padding_no_strides_transposed.gif"></td>
    <td><img width="150px" src="../assets/same_padding_no_strides_transposed.gif"></td>
    <td><img width="150px" src="../assets/full_padding_no_strides_transposed.gif"></td>
  </tr>
  <tr>
    <td>No padding, strides, transposed</td>
    <td>Padding, strides, transposed</td>
    <td>Padding, strides, transposed (odd)</td>
  </tr>
  <tr>
    <td><img width="150px" src="../assets/no_padding_strides_transposed.gif"></td>
    <td><img width="150px" src="../assets/padding_strides_transposed.gif"></td>
    <td><img width="150px" src="../assets/padding_strides_odd_transposed.gif"></td>
  </tr>
</table>

#### 空洞卷积
- 空洞卷积（Atrous Convolutions）也称扩张卷积（Dilated Convolutions）、膨胀卷积。
  <div align="center"><img src="../assets/conv_dilation.gif" height="200" /><br/>No padding, no strides.</div>

**空洞卷积的作用**
- 空洞卷积使 CNN 能够**捕捉更远的信息，获得更大的感受野**；同时不增加参数的数量，也不影响训练的速度。
- 示例：Conv1D + 空洞卷积
  <div align="center"><img src="../assets/普通卷积与膨胀卷积.png" height="200" /></div>


#### 可分离卷积
- 可分离卷积（separable convolution）
- TODO

### 门卷积
- 类似 LSTM 的过滤机制，实际上是卷积网络与**门限单元**（Gated Linear Unit）的组合
  > [卷积新用之语言模型](https://blog.csdn.net/stdcoutzyx/article/details/55004458) - CSDN博客 
- 核心公式
  <div align="center"><a href=""><img src="../assets/公式_20180720110804.png" /></a></div>
  <!-- \boldsymbol{Y}=\text{Conv1D}_{(1)}(\boldsymbol{X}) \otimes \sigma\Big(\text{Conv1D}_{(2)}(\boldsymbol{X})\Big)dsymbol{X})\Big) -->
  
  > 中间的运算符表示**逐位相乘**—— Tensorflow 中由 `tf.multiply(a, b)` 实现，其中 a 和 b 的 shape 要相同；后一个卷积使用`sigmoid`激活函数

#### 门卷积是如何防止梯度消失的
- 因为公式中有一个卷积没有经过激活函数，所以对这部分求导是个常数，所以梯度消失的概率很小。
- 如果还是担心梯度消失，还可以加入**残差**
  <div align="center"><a href=""><img src="../assets/公式_20180720113735.png" /></a></div>
  <!-- \boldsymbol{Y}={\color{Red} \boldsymbol{X} \,+\;} \text{Conv1D}_{(1)}(\boldsymbol{X}) \otimes \sigma\Big(\text{Conv1D}_{(2)}(\boldsymbol{X})\Big) -->

  更直观的理解：
  <div align="center"><a href=""><img src="../assets/公式_20180720120400.png" /></a></div>
  <!-- \begin{aligned}\boldsymbol{Y}=&\,\boldsymbol{X} + {\color{Red}\text{Conv1D}_{(1)}(\boldsymbol{X})}\otimes \sigma\Big(\text{Conv1D}_{(2)}(\boldsymbol{X})\Big)\\=&\,\boldsymbol{X} + {\color{Red}\Big(\text{Conv1D}_{(1)}(\boldsymbol{X}) - \boldsymbol{X}\Big)}\otimes \sigma\Big(\text{Conv1D}_{(2)}(\boldsymbol{X})\Big)\\ =&\,\boldsymbol{X}\otimes \Big[1-\sigma\Big(\text{Conv1D}_{(2)}(\boldsymbol{X})\Big)\Big] + \text{Conv1D}_{(1)}(\boldsymbol{X}) \otimes \sigma\Big(\text{Conv1D}_{(2)}(\boldsymbol{X})\Big)\\ =&\,\boldsymbol{X}\otimes \Big(1-\boldsymbol{\sigma}\Big) + \text{Conv1D}_{(1)}(\boldsymbol{X}) \otimes \boldsymbol{\sigma} \end{aligned} -->

  即信息以 `1-σ` 的概率直接通过，以 `σ` 的概率经过变换后通过——类似 GRU
  > 因为`Conv1D(X)`没有经过激活函数，所以实际上它只是一个线性变化；因此与 `Conv1D(X) - X` 是等价的
  >
  > [基于CNN的阅读理解式问答模型：DGCNN](https://kexue.fm/archives/5409#门机制) - 科学空间|Scientific Spaces —— 当时评测的第一名

### 如何构造位置向量
