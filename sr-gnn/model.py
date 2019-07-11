# -*- coding: utf-8 -*-
# @Time    : 2019/7/8 16:14
# @Author  : skydm
# @Email   : wzwei1636@163.com
# @File    : model.py
# @Software: PyCharm

import random
import numpy as np
import tensorflow as tf
import tensorflow.python.keras as keras
import tensorflow.python.keras.backend as K

# 设置随机种子，方便复现
seed = 1234
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)


def build_model(batch_size, items_num, hidden_size, steps):
    '''
    :param batch_size:
    :param items_num:
    :param hidden_size:
    :param steps: gnn propogation steps
    :return:
    '''
    # 长度固定(一个batch_size的各异的商品数), session点击序列的商品id保持不动
    items = keras.layers.Input(shape=(None,), name='items', dtype="int32")
    # 对应的索引, 与items相对(点击序列长度)
    seq_index = keras.layers.Input(shape=(None, ), name="seq_index", dtype="int32")
    # 与seq_index相对，每个session序列重新编码之后最大的索引
    last_index = keras.layers.Input(shape=(1, ), name="last_index", dtype="int32")
    # 入度的邻接矩阵
    adj_in = keras.layers.Input(shape=(None, None), name="adj_in", dtype="float32")
    # 出度的邻接矩阵， 一个session序列对应一个邻接矩阵
    adj_out = keras.layers.Input(shape=(None, None), name="adj_out", dtype="float32")
    # session原始点击序列的mask序列
    mask = keras.layers.Input(shape=(None, 1), name="mask", dtype="float32")
    # session对应的下一次点击商品id
    label = keras.layers.Input(shape=(1,), name="label", dtype="int32")

    # inputs = [items, seq_index, last_index, adj_in, adj_out, mask, label]
    # 构建商品的embedding矩阵
    items_embedding = keras.layers.Embedding(input_dim=items_num,
                                             output_dim=hidden_size,
                                             embeddings_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1e-4, seed=seed))

    # 定义个可训练的tensor, shape: (items_num - 1, hidden_size)
    class Bias(keras.layers.Layer):
        def build(self, input_shape):
            self.y_emb = self.add_weight(shape=(items_num - 1, hidden_size), initializer='zeros', dtype=tf.float32, name='vocab_emb')
            self.built = True

        def call(self, x):
            return tf.matmul(x, self.y_emb, transpose_b=True)

    # 获取原始session的商品id的embdding矩阵
    items_emb = items_embedding(items)  # (batch_size, uniq_max, hidden_size)   # 输入的长度是确定的

    init_state = items_emb

    item_in = keras.layers.Dense(hidden_size)   # 训练出(hidden_size, hidden_size)的权重矩阵
    item_out = keras.layers.Dense(hidden_size)  # 共享, 从而学习全局权重矩阵

    seq_dense = keras.layers.Dense(hidden_size)
    last_dense = keras.layers.Dense(hidden_size)


    # gnn传播的步数
    for i in range(steps):
        init_state = keras.layers.Lambda(lambda x: tf.reshape(x, [batch_size, -1, hidden_size]))(init_state)    # (batch_size*uniq_max, hidden_size)
        # 对商品序列的embedding矩阵进行线性变换
        # (batch_size, uniq_max, uniq_max) * (batch_size, uniq_max, hidden_size) => (batch_size, uniq_max, hidden_size)
        state_in = item_in(init_state)      # (batch_size, uniq_max, hidden_size)
        state_out = item_out(init_state)    # (batch_size, uniq_max, hidden_size)
        # 与邻接矩阵进行矩乘法  分别获取[batch_size, uniq_max, hidden_size]
        state_adj_in = keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1]))([adj_in, state_in])
        state_adj_out = keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1]))([adj_out, state_out])
        # 进行拼接，作为GRU的输入[batch_size, uniq_max, 2 * hidden_size]
        gru_input = keras.layers.Lambda(lambda x: tf.concat([x[0], x[1]], axis=2))([state_adj_in, state_adj_out])
        # 缩减维度，进行一步gru迭代[batch_size * uniq_max, 2 * hidden_size]
        gru_input = keras.layers.Lambda(lambda x: tf.reshape(x, [-1, 2 * hidden_size]))(gru_input)
        # 扩展维度, 在时间步上进行填充(batch_size * uniq_max, 1, 2 * hidden_size)
        gru_input = keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(gru_input)
        # 经过GRU, 得到(batch_size, hidden_size)
        _, init_state = keras.layers.GRU(hidden_size, return_sequences=True, return_state=True)(gru_input)

    final_state = init_state    # (batch_size * uniq_max, hidden_size)
    # 从获取每一行session点击序列的索引
    seq_reshape = keras.layers.Lambda(lambda x: tf.reshape(x, [-1]))(seq_index)       # (batch_size * seq_len) seq_len是点击序列长度
    seq = keras.layers.Lambda(lambda x: tf.gather(final_state, x))(seq_reshape)       # (batch_size * seq_len, hidden_size)
    seq = keras.layers.Lambda(lambda x: tf.reshape(x, [batch_size, -1, hidden_size]))(seq)  # (batch_size, seq_len, hidden_size)
    # last = keras.layers.Lambda(lambda x: tf.squeeze(tf.gather(final_state, x)))(last_index)     # (batch_size, hidden_size)
    last = keras.layers.Lambda(lambda x: tf.gather(final_state, x))(last_index)  # (batch_size, 1, hidden_size)
    # 注意使用tf.squeeze删除维度为1，会造成张量的rank消息 its rank is undefined, but the layer requires a defined rank.
    last = keras.layers.Lambda(lambda x: tf.reshape(x, [-1, hidden_size]))(last)

    seq_fc = seq_dense(seq)     # (batch_size, seq_len, hidden_size)
    last_fc = last_dense(last)  # (batch_size, hidden_size)

    add = keras.layers.Add()([seq_fc, last_fc])  # (batch_size, seq_len, hidden_size)
    add_sigmoid = keras.layers.Lambda(lambda x: tf.sigmoid(x))(add)
    weights = keras.layers.Dense(1)(add_sigmoid)    # (batch_size, seq_len, 1)
    weights = keras.layers.Multiply()([weights, mask])  # (batch_size, seq_len, 1)  对应元素相乘
    weights_mask = keras.layers.Multiply()([seq, weights])   # (batch_size, seq_len, hidden_size)

    global_attention = keras.layers.Lambda(lambda x: tf.reduce_sum(weights_mask, axis=1))(weights_mask)     # (batch_size, hidden_size)
    final_attention = keras.layers.Lambda(lambda x: tf.concat([x[0], x[1]], axis=1))([global_attention, last])  # (batch_size, 2*hidden_size)
    final_attention_fc = keras.layers.Dense(hidden_size)(final_attention)   # (batch_size, hidden_size)

    # 现在需要定义个可训练的tensor, shape: (items_num - 1, hidden_size)
    logits = Bias()(final_attention_fc)    # (batch_size, items_num - 1)

    model = keras.models.Model(inputs=[items, seq_index, last_index, adj_in, adj_out, mask], outputs=logits)

    # 计算损失函数
    loss = K.categorical_crossentropy(target=label, output=logits, from_logits=True)
    model.add_loss(loss)

    model.compile(optimizer=keras.optimizers.SGD(1e-3))

    return model