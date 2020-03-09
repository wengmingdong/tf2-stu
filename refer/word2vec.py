#忘记那个博客粘帖，这里做为笔记保存了

from __future__ import division, print_function, absolute_import

import collections
import os
import random
from urllib import request
import zipfile

import numpy as np
import tensorflow as tf

# 训练参数
learning_rate = 0.1
batch_size = 128
num_steps = 3000000
display_step = 10000
eval_step = 200000

# 评估参数
eval_words = ['five', 'of', 'going', 'hardware', 'american', 'britain']

# Word2Vec 参数
embedding_size = 200 # 嵌入向量的维度 vector.
max_vocabulary_size = 50000 # 词汇表中不同单词的总数words in the vocabulary.
min_occurrence = 10  # 删除出现小于n次的所有单词
skip_window = 3 # 左右各要考虑多少个单词
num_skips = 2 # 重复使用输入生成标签的次数
num_sampled = 64 # 负采样数量

# 下载一小部分维基百科文章集
url = 'http://mattmahoney.net/dc/text8.zip'
data_path = './data/text8.zip'
if not os.path.exists(data_path):
    print("Downloading the dataset... (It may take some time)")
    filename, _ = request.urlretrieve(url, data_path)
    print("Done!")

# 解压数据集文件，文本已处理完毕
with zipfile.ZipFile(data_path) as f:
    text_words = f.read(f.namelist()[0]).lower().split()

# 构建词典并用 UNK 标记替换频数较低的词
count = [('UNK', -1)]
# 检索最常见的单词
count.extend(collections.Counter(text_words).most_common(max_vocabulary_size - 1))
# 删除少于'min_occurrence'次数的样本
for i in range(len(count) - 1, -1, -1):
    if count[i][1] < min_occurrence:
        count.pop(i)
    else:
        #该集合是有序的，因此在当出现小于'min_occurrence'时停止
        break
# 计算单词表单词个数
vocabulary_size = len(count)
# 为每一个词分配id
word2id = dict()
for i, (word, _)in enumerate(count):
    word2id[word] = i

data = list()
unk_count = 0
for word in text_words:
     # 检索单词id，或者如果不在字典中则为其指定索引0（'UNK'）
    index = word2id.get(word, 0)
    if index == 0:
        unk_count += 1
    data.append(index)
count[0] = ('UNK', unk_count)
id2word = dict(zip(word2id.values(), word2id.keys()))

print("Words count:", len(text_words))
print("Unique words:", len(set(text_words)))
print("Vocabulary size:", vocabulary_size)
print("Most common words:", count[:10])

data_index = 0
# 为skip-gram模型生成训练批次
def next_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # 得到窗口长度( 当前单词左边和右边 + 当前单词)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    #回溯一点，以避免在批处理结束时跳过单词
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

# 确保在CPU上分配以下操作和变量
# (某些操作在GPU上不兼容)
with tf.device('/cpu:0'):
    # 创建嵌入变量（每一行代表一个词嵌入向量） embedding vector).
    embedding = tf.Variable(tf.random.normal([vocabulary_size, embedding_size]))
    # 构造NCE损失的变量
    nce_weights = tf.Variable(tf.random.normal([vocabulary_size, embedding_size]))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

def get_embedding(x):
    with tf.device('/cpu:0'):
       # 对于X中的每一个样本查找对应的嵌入向量
        x_embed = tf.nn.embedding_lookup(embedding, x)
        return x_embed

def nce_loss(x_embed, y):
    with tf.device('/cpu:0'):
        # 计算批处理的平均NCE损失
        y = tf.cast(y, tf.int64)
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=y,
                           inputs=x_embed,
                           num_sampled=num_sampled,
                           num_classes=vocabulary_size))
        return loss

# 评估
def evaluate(x_embed):
    with tf.device('/cpu:0'):
         # 计算输入数据嵌入与每个嵌入向量之间的余弦相似度
        x_embed = tf.cast(x_embed, tf.float32)
        x_embed_norm = x_embed / tf.sqrt(tf.reduce_sum(tf.square(x_embed)))
        embedding_norm = embedding / tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keepdims=True), tf.float32)
        cosine_sim_op = tf.matmul(x_embed_norm, embedding_norm, transpose_b=True)
        return cosine_sim_op

# 定义优化器
optimizer = tf.optimizers.SGD(learning_rate)

# 优化过程
def run_optimization(x, y):
    with tf.device('/cpu:0'):
       # 将计算封装在GradientTape中以实现自动微分
        with tf.GradientTape() as g:
            emb = get_embedding(x)
            loss = nce_loss(emb, y)

        # 计算梯度
        gradients = g.gradient(loss, [embedding, nce_weights, nce_biases])

         # 按gradients更新 W 和 b
        optimizer.apply_gradients(zip(gradients, [embedding, nce_weights, nce_biases]))


# 用于测试的单词
x_test = np.array([word2id[w] for w in eval_words])

# 针对给定步骤数进行训练
for step in range(1, num_steps + 1):
    batch_x, batch_y = next_batch(batch_size, num_skips, skip_window)
    run_optimization(batch_x, batch_y)

    if step % display_step == 0 or step == 1:
        loss = nce_loss(get_embedding(batch_x), batch_y)
        print("step: %i, loss: %f" % (step, loss))

    # 评估
    if step % eval_step == 0 or step == 1:
        print("Evaluation...")
        sim = evaluate(get_embedding(x_test)).numpy()
        for i in range(len(eval_words)):
            top_k = 8  # 最相似的单词数量
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = '"%s" nearest neighbors:' % eval_words[i]
            for k in range(top_k):
                log_str = '%s %s,' % (log_str, id2word[nearest[k]])
            print(log_str)