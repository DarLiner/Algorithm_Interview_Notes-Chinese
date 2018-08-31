Attention机制-论文收录说明
===

Index
---
<!-- TOC -->

- [[2017].Attention_is_All_You_Need](#2017attention_is_all_you_need)
- [[2017].A_Structured_Self-attentive_Sentence_Embedding](#2017a_structured_self-attentive_sentence_embedding)
- [[2018].Densely_Connected_CNN_with_Multiscale_Feature_Attention](#2018densely_connected_cnn_with_multiscale_feature_attention)
- [[2018].Next_Item_Recommendation_with_Self-Attention](#2018next_item_recommendation_with_self-attention)

<!-- /TOC -->

## [2017].Attention_is_All_You_Need
- 谷歌大脑出品，很有名的一篇文章

## [2017].A_Structured_Self-attentive_Sentence_Embedding
- [[2017].Attention_is_All_You_Need](#2017attention_is_all_you_need) 中引用了本文
- 本文使用 Self-Attention 学习**任务无关**的 Sentence Embedding
  > 我曾在搜狗的面试中被问到“知道哪些方法可以学习任务无关的句向量吗？”

## [2018].Densely_Connected_CNN_with_Multiscale_Feature_Attention
- PaperWeekly 周推荐 > https://www.paperweekly.site/papers/2240
- 源码：https://github.com/wangshy31/Densely-Connected-CNN-with-Multiscale-Feature-Attention

- 本文是清华大学发表于 IJCAI 2018 的工作。针对文本分类任务中卷积神经网络通常无法灵活学习可变 n 元特征（n-gram）的问题，论文提出了一种具有**适应式注意力机制**的**密集连接的卷积神经网络**。该模型通过建立底层特征和高层特征之间的**跨层连接**，从而获得了丰富的**多尺度特征**，而注意力模型能够**自适应地选择合适尺度的特征**以适用于各种不同的文本分类问题。该法面向六个公开数据集均实现了超过基线的预测精度。

## [2018].Next_Item_Recommendation_with_Self-Attention
- PaperWeekly 周推荐 > https://www.paperweekly.site/papers/2246

- 本文提出了一种基于 self-attention 的基于序列的**推荐算法**，该算法是用 self-attention 从用户的**交互记录**中自己的去学习用户**近期的兴趣**，同时该模型也保留了用户**长久的兴趣**。整个网络是在 **metric learning** 的框架下，是第一次将 self-attention 和 metric learning 的结合的尝试。

- 实验结果表明，通过 self-attention，模型可以很好的学习用户的短期兴趣爱好，并且能有效的提升模型效果。通过和近期的文章得对比发现，该方法可以在很大程度上改善**序列化推荐**的效果。