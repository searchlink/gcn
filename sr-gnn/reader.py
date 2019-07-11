# -*- coding: utf-8 -*-
# @Time    : 2019/7/8 11:00
# @Author  : skydm
# @Email   : wzwei1636@163.com
# @File    : reader.py
# @Software: PyCharm

import numpy as np
import pickle
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


class Data:
    def __init__(self, path, batch_size, shuffle=False):
        data = pickle.load(open(path, "rb"))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.length = len(data[0])
        self.input = list(zip(data[0], data[1]))    # 生成类似([1, 1], 2)
        self.batch_data = self.generate_batch()    # 生成batch_list

    def generate_batch(self):
        '''返回每个batch_size所在的下标'''
        if self.shuffle:
            np.random.shuffle(self.input)

        n_batch = self.length // self.batch_size
        if self.length % self.batch_size !=0:
            n_batch = n_batch + 1
        print(n_batch)
        batch_data = []
        for i in range(n_batch):
            if i != n_batch - 1:
                batch_data.append(self.input[i*self.batch_size:(i+1)*self.batch_size])
            else:
                batch_data.append(self.input[(self.length - self.batch_size):self.length])
        return batch_data

    def get_graph_data(self, index):
        '''
        这里将输入数据转换成为图的结构矩阵
        输入：
            u_A = np.array([[0,1,0,0], [0, 0, 1, 1], [0, 1, 0,0], [0,0,0,0]])
        进行验证论文
        '''
        cur_batch = [list(e) for e in self.batch_data[index]]
#         cur_batch = self.input[cur_batch[0]:(cur_batch[0] + 1)]  # 获取batch_size的数据
        inputs = [tuple_[0] for tuple_ in cur_batch]
        label = [tuple_[1] - 1 for tuple_ in cur_batch]

        # 获取batch_size的max_len
        seq_len_batch = [len(seq) for seq in inputs]
        max_seq_len = max(seq_len_batch)

        # 生成padding序列、mask序列
        mask = [[1] * l + [0] * (max_seq_len - l) for l in seq_len_batch]

        # 一个batch_size的所有唯一node
        last_id = []
        for e in cur_batch:
            last_id.append(len(e[0]) - 1)
            e[0] += [0] * (max_seq_len - len(e[0]))

        max_unq_len = 0
        for e in cur_batch:
            max_unq_len = max(max_unq_len, len(np.unique(e[0])))

        # 这里需要注意到转化为邻接矩阵的时候， 此时点击序列还要再进行一次变换，即索引序列也要变换
        items, adj_in, adj_out, seq_index, last_index = [], [], [], [], []

        id = 0
        for e in cur_batch:
            node = np.unique(e[0])
            items.append(node.tolist() + (max_unq_len - len(node)) * [0])
            adj = np.zeros((max_unq_len, max_unq_len))
            for i in np.arange(len(e[0]) - 1):
                if e[0][i + 1] == 0:
                    break
                u = np.where(node == e[0][i])[0][0]     # 查找下标
                v = np.where(node == e[0][i + 1])[0][0]
                adj[u][v] = 1   # 相邻节点

            u_sum_in = np.sum(adj, axis=0)     # 入度(n_col)
            u_sum_in[np.where(u_sum_in == 0)] = 1   # normalize
            adj_in.append(np.divide(adj, u_sum_in).transpose())   # normalize(相加为1) 确保最后一维相同  (n_row, n_col) / (n_col)

            u_sum_out = np.sum(adj, axis=1)     # 出度(n_row)
            u_sum_out[np.where(u_sum_out == 0)] = 1     # normalize
            adj_out.append(np.divide(adj.transpose(), u_sum_out).transpose())     # (n_row, n_col) / (n_row) => (n_col, n_row) / (n_row)

            seq_index.append([np.where(node == i)[0][0] + id * max_unq_len for i in e[0]])
            last_index.append(np.where(node == e[0][last_id[id]])[0][0] + id * max_unq_len)

            id += 1

        items = np.array(items).astype("int32").reshape((self.batch_size, -1, 1))
        seq_index = np.array(seq_index).astype("int32").reshape((self.batch_size, -1))
        last_index = np.array(last_index).astype("int32").reshape((self.batch_size, 1))
        adj_in = np.array(adj_in).astype("float32").reshape((self.batch_size, max_unq_len, max_unq_len))
        adj_out = np.array(adj_out).astype("float32").reshape((self.batch_size, max_unq_len, max_unq_len))
        mask = np.array(mask).astype("float32").reshape((self.batch_size, -1, 1))
        label = np.array(label).astype("float32").reshape((self.batch_size, 1))

        # return zip(items, seq_index, last_index, adj_in, adj_out, mask, label)
        return items, seq_index, last_index, adj_in, adj_out, mask, label