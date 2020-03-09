import tensorflow.python as tf
from tensorflow.python import keras
import numpy as np
# keras.backend.clear_session()
tf.enable_eager_execution()

x = keras.Input(shape=(10))
label_1 = keras.Input(shape=(10))
label_2 = keras.Input(shape=(10))
pred_1 = keras.layers.Dense(10)(x)
pred_2 = keras.layers.Dense(10)(x)

model = keras.Model(inputs=[x, label_1, label_2], outputs=[pred_1, pred_2])
model.summary()

train_set = tf.data.Dataset.from_tensor_slices({'input_1': np.random.randint(1, 4, 1000).reshape(100, 10), 'input_2': np.random.randint(5, 8, 1000).reshape(100, 10), 'input_3': np.random.randint(9, 12, 1000).reshape(100, 10)}).repeat().batch(32)
train_set = train_set.map(lambda x: {'input_1': float(x['input_1']) + .1, 'input_2': float(x['input_2']) - .2, 'input_3': float(x['input_3']) - .5})

def losses(labels: list, preds: list):
    l = 0
    for i in range(len(labels)):
        # 这里我可以给不同的label不同的loss操作
        l += tf.reduce_sum(((labels[i] - preds[i])**2) * (i + 1))
    return l
#Lambda层
model.add_loss(losses([label_1, label_2], [pred_1, pred_2]))
model.compile('adam')

model.fit(train_set, steps_per_epoch=30)