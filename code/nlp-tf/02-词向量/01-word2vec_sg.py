import csv
import math
import random
import numpy as np
import tensorflow as tf

from .config import config_sg as config
from .data_helper import *

# 准备数据
filename = "./data/wikipedia2text-extracted.txt.bz2"  # for Ipython
# words = read_data(filename)
words = read_data_small(filename)
print('Data size %d' % len(words))
print('Example words (start): ', words[:10])
print('Example words (end): ', words[-10:])

data, count, word2id, id2word = build_dataset(words)
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])
del words

# 准备验证集
valid_window, valid_size = config.valid_window, config.valid_size
valid_examples = np.array(random.sample(range(valid_window), valid_size))
valid_examples = np.append(valid_examples, random.sample(range(1000, 1000 + valid_window), valid_size), axis=0)

# Graph
tf.reset_default_graph()
g = tf.get_default_graph()

# Input
batch_size = config.batch_size
train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# Word2Vec Model
# Embedding layer
vocabulary_size, embedding_size = config.vocabulary_size, config.embedding_size
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embeddings, train_dataset)

# Softmax Weights and Biases
softmax_W = tf.Variable(
    tf.truncated_normal([vocabulary_size, embedding_size],
                        stddev=0.5 / math.sqrt(embedding_size)))
softmax_b = tf.Variable(tf.random_uniform([vocabulary_size], 0.0, 0.01))

# 负采样
num_sampled = config.num_sampled
loss = tf.reduce_mean(
    tf.nn.sampled_softmax_loss(
        weights=softmax_W, biases=softmax_b, inputs=embed,
        labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))

# train_op
optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

# 计算相似度
# norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
# normalized_embeddings = embeddings / norm
normalized_embeddings = tf.nn.l2_normalize(embeddings, axis=1)  # 等价于以上两行
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
# 计算 cosine 相似度（内积）
similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

# Train
skip_losses = []
num_steps = config.num_steps
with tf.Session(graph=g) as sess:
    """"""
    # init
    tf.global_variables_initializer().run()

    # The average loss is an estimate of the loss over the last 2000 batches.
    average_loss = 0
    for step in range(num_steps):

        batch_data, batch_labels = generate_batch_sg(data, batch_size, config.window_size)

        # run train_op and get loss
        _, loss_val = sess.run([optimizer, loss], feed_dict={train_dataset: batch_data,
                                                             train_labels: batch_labels})

        # Update the average loss
        average_loss += loss_val

        if (step + 1) % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000

            skip_losses.append(average_loss)
            print('Average loss at step %d: %f' % (step + 1, average_loss))
            average_loss = 0

        # 评价
        if (step + 1) % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = id2word[valid_examples[i]]
                top_k = config.top_k
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = id2word[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)

    embeddings_sg = normalized_embeddings.eval()

# We will save the word vectors learned and the loss over time
# as this information is required later for comparisons
np.save('./out/embeddings_sg', embeddings_sg)

with open('skip_losses.csv', 'wt') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(skip_losses)


# TODO(huay): 可视化
