import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing

learning_rate = 0.1

class Model(object):
    def __init__(self, num_users, num_items, k):
        self.num_users = num_users
        self.num_items = num_items
        self.k = k
        self.feature_user = tf.Variable(tf.random.normal([num_users, k]))  # 生成10*1的张量
        self.feature_item = tf.Variable(tf.random.normal([num_items, k]))
    def __call__(self, user_ids, item_ids):
        embbeding_u = tf.nn.embedding_lookup(self.feature_user, list(np.array(user_ids)-1))
        embbeding_i = tf.nn.embedding_lookup(self.feature_item, list(np.array(item_ids)-1))
        rt_ui = tf.linalg.diag_part(tf.matmul(embbeding_u, embbeding_i, transpose_b=True))
        return rt_ui

    def get_user_embbeding(self, user_ids):
        embbeding_u = tf.nn.embedding_lookup(self.feature_user, list(np.array(user_ids)-1))
        return embbeding_u

    def get_items_embbeding(self, item_ids):
        embbeding_i = tf.nn.embedding_lookup(self.feature_item, list(np.array(item_ids)-1))
        return embbeding_i

def loss(predicted_y, desired_y, embbeding_u, embbeding_i, ld):
  plos = tf.reduce_mean(tf.square(predicted_y - desired_y))
  plos_w = tf.reduce_mean(ld*tf.multiply(embbeding_u, embbeding_u)) + tf.reduce_mean(ld*tf.multiply(embbeding_i, embbeding_i))
  plos = plos + plos_w
  return plos

# 定义优化器
optimizer = tf.optimizers.SGD(learning_rate)

def train(model, inputs_user_ids, inputs_item_ids, outputs, idx):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs_user_ids, inputs_item_ids), outputs,
                            model.get_user_embbeding(inputs_user_ids),
                            model.get_items_embbeding(inputs_item_ids), 0.001)
    if idx % 100000 == 0:
        print('idx:%d, loss:%f' % (idx, current_loss))

    gradients = t.gradient(current_loss, [model.feature_user, model.feature_item])
    optimizer.apply_gradients(zip(gradients, [model.feature_user, model.feature_item]))

df = pd.read_csv('./data/ml-1m/ratings.dat', sep='\t', names=['user', 'item', 'rating', 'timestamp'], header=None)
df = df.drop('timestamp', axis=1)

# 开始数据处理
num_items = df.item.nunique()
num_users = df.user.nunique()
print("USERS: {} ITEMS: {}".format(num_users, num_items))

r = df['rating'].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(r.reshape(-1,1))
df_normalized = pd.DataFrame(x_scaled)
df['rating'] = df_normalized

user_ids = df['user'].values.astype(int)
item_ids = df['item'].values.astype(int)
outputs = df['rating'].values.astype(float)

print(len(user_ids), len(item_ids), len(outputs))
#最大用户id
max_user_id = np.amax(user_ids)
#最大电影id
max_item_id = np.amax(item_ids)

model = Model(max_user_id, max_item_id, 20)

print(model.feature_user[:10])

max_len = len(user_ids)
batch_size = 1000
epochs = range(1000)
for epoch in epochs:
    cur_idx = 0
    print('Epoch %2d' % (epoch))
    while True:
        start_idx = cur_idx
        end_idx = cur_idx + batch_size
        if start_idx >= max_len:
            break
        if end_idx > max_len:
            end_idx = max_len

        cur_user_ids = user_ids[start_idx:end_idx]
        cur_item_ids = item_ids[start_idx:end_idx]
        cur_outputs = outputs[start_idx:end_idx]

        train(model, cur_user_ids, cur_item_ids, cur_outputs, cur_idx)
        cur_idx += batch_size


print(model.feature_user[:10])



