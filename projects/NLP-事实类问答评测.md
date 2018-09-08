NLP-事实类问答评测
===

Index
---
<!-- TOC -->

- [任务描述](#任务描述)
- [基础模型 - BiDAF](#基础模型---bidaf)

<!-- /TOC -->

## 任务描述
- 针对每个问题 q，给定与之对应的若干候选答案篇章 a1，a2，…，an，要求设计算法从候选篇章中**抽取合适的词语、短语或句子**，形成一段正确、完整、简洁的文本，作为预测答案 apred，目标是 apred 能够正确、完整、简洁地回答问题 q。

- **示例**
  ```
  问题: 中国最大的内陆盆地是哪个
  答案：塔里木盆地
  材料：
    1. 中国新疆的塔里木盆地，是世界上最大的内陆盆地，东西长约1500公里，南北最宽处约600公里。盆地底部海拔1000米左右，面积53万平方公里。
    2. 中国最大的固定、半固定沙漠天山与昆仑山之间又有塔里木盆地，面积53万平方公里，是世界最大的内陆盆地。盆地中部是塔克拉玛干大沙漠，面积33.7万平方公里，为世界第二大流动性沙漠。
  ```

- **数据下载**
  - [CIPS-SOGOU问答比赛](http://task.www.sogou.com/cips-sogou_qa/) （少量）
  - [百度 WebQA V2.0](http://ai.baidu.com/broad/download)
  - [百度 WebQA V1.0 预处理版](https://pan.baidu.com/s/1SADkZjF7kdH2Qk37LTdXKw)（密码: kc2q）
    > [【语料】百度的中文问答数据集WebQA](https://spaces.ac.cn/archives/4338) - 科学空间|Scientific Spaces 


## 基础模型 - BiDAF
> [1611.01603] [Bidirectional Attention Flow for Machine Comprehension](https://arxiv.org/abs/1611.01603) 

**5/6 层模型结构**
1. 嵌入层（字+词）
1. Encoder 层
1. Attention 交互层
1. Decoder 层
1. 输出层