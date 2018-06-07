## 问题与答案

1. ### 如何设置网络的初始值

    一般总是使用服从（截断）高斯或均匀分布的随机值，具体是高斯还是均匀分布影响不大，但是也没有详细的研究。
    
    但是，**初始值的大小**会对优化结果和网络的泛化能力产生较大的影响：更大的初始值有助于避免冗余的单元和梯度消失；但如果初始值太大，又会造成梯度爆炸。
    
    > 《深度学习》 ch8.4 - 参数初始化策略

    一些启发式初始化策略通常是根据输入与输出的单元数来决定初始权重的大小，比如 Glorot and Bengio (2010) 中建议建议使用的标准初始化，其中 m 为输入数，n 为输出数

    <a href="http://www.codecogs.com/eqnedit.php?latex=W_{i,j}&space;\sim&space;U(-\sqrt{\frac{6}{m&plus;n}},\sqrt{\frac{6}{m&plus;n}})" target="_blank"><img src="http://latex.codecogs.com/gif.latex?W_{i,j}&space;\sim&space;U(-\sqrt{\frac{6}{m&plus;n}},\sqrt{\frac{6}{m&plus;n}})" title="W_{i,j} \sim U(-\sqrt{\frac{6}{m+n}},\sqrt{\frac{6}{m+n}})" /></a>

    还有一些方法推荐使用随机正交矩阵来初始化权重 (Saxe et al., 2013)。

    > 常用的初始化策略可以参考 Keras 中文文档：[初始化方法Initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations/)


2. ### 梯度爆炸的解决办法
    
    1. 梯度截断/裁剪——如果梯度超过某个阈值，就对其进行限制
        
        来看一下 Tensorflow 提供的几种方法：

        - `tf.clip_by_value(t, clip_value_min, clip_value_max)`
        - `tf.clip_by_norm(t, clip_norm)`
        - `tf.clip_by_average_norm(t, clip_norm)`
        - `tf.clip_by_global_norm(t_list, clip_norm)`

        这里主要说一下`tf.clip_by_global_norm`的操作：

        ```
        To perform the clipping, the values `t_list[i]` are set to:

            t_list[i] * clip_norm / max(global_norm, clip_norm)

        where:

            global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))
        ```

        用法：

        ```
        train_op = tf.train.AdamOptimizer()
        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)

        clip_norm = 100
        clipped_gradients, global_norm = tf.clip_by_global_norm(gradients, clip_norm)

        optimizer_op = train_op.apply_gradients(zip(clipped_gradients, params))
        ```

        > clip_norm 的设置视 loss 的大小而定，如果比较大，那么可以设为 100 或以上，如果比较小，可以设为 10 或以下。
    
    2. 良好的参数初始化策略也能缓解梯度爆炸问题（权重正则化）

        > [如何设置网络的初始值](#如何设置网络的初始值)
    
    3. 使用线性整流激活函数，如 ReLU 等


3. ### MLP 的万能近似定理

    一个前馈神经网络如果具有至少一个非线性输出层，那么只要给予网络足够数量的隐藏单元，它就可以以任意的精度来近似任何从一个有限维空间到另一个有限维空间的函数。

    > 《深度学习》 ch6.4.1 - 万能近似性质和深度


4. ### 在 MLP 中，深度与宽度的关系，及其表示能力的差异
    
    隐藏层的数量称为模型的**深度**，隐藏层的维数（单元数）称为该层的**宽度**。
    
    **万能近似定理**表明一个单层的网络就足以表达任意函数，但是该层的维数可能非常大，且几乎没有泛化能力；此时，使用更深的模型能够减少所需的单元数，同时增强泛化能力（减少泛化误差）。参数数量相同的情况下，浅层网络比深层网络更容易过拟合。

    > 《深度学习》 ch6.4 - 架构设计；这一节的内容比较分散，想要更好的回答这个问题，需要理解深度学习的本质——学习多层次组合（ch1.2），这才是现代深度学习的基本原理。


5. ### 稀疏表示，低维表示，独立表示

    无监督学习任务的目的是找到数据的“最佳”表示。“最佳”可以有不同的表示，但是一般来说，是指该表示在比本身表示的信息更简单的情况下，尽可能地保存关于 x 更多的信息。

    低维表示、稀疏表示和独立表示是最常见的三种“简单”表示：1）低维表示尝试将 x 中的信息尽可能压缩在一个较小的表示中；2）稀疏表示将数据集嵌入到输入项大多数为零的表示中；3）独立表示试图分开数据分布中变化的来源，使得表示的维度是统计独立的。

    这三种表示不是互斥的，比如主成分分析（PCA）就试图同时学习低维表示和独立表示。

    表示的概念是深度学习的核心主题之一。
    
    > 《深度学习》 ch5.8 - 无监督学习算法


6. ### 函数在某个区间的饱和与**平滑性**对基于梯度的学习的影响

    > 《深度学习》 ch5.11.2 - 局部不变性和平滑正则化