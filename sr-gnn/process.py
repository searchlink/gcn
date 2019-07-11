#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-07-06 16:43
# @Author  : skydm
# @Site    : 
# @File    : process.py
# @Software: PyCharm

import csv
import time
import pickle

file = "/home/wangwei/ctr_model/SR-GNN/train-item-views.csv"

with open(file, "r") as f:
    reader = csv.DictReader(f, delimiter=";")
    sess_clicks = {}    # 保存点击序列
    sess_date = {}  # 保留时间，方便后续划分测试集和训练集

    # i = 0
    for data in reader:
        # i = i + 1
        # if i > 200:
        #     break
        sessid = data['session_id']
        # 对应的点击商品id和相应的时间戳
        item = (data['item_id'], int(data['timeframe']))
        # 转化为时间戳，方便后续划分训练和测试数据集
        date = time.mktime(time.strptime(data['eventdate'], '%Y-%m-%d'))
        sess_date[sessid] = date

        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]

    # 开始按照时间戳排序
    for i in sess_clicks.keys():
        sorted_clicks = sorted(sess_clicks[i], key=lambda x: x[1])
        sess_clicks[i] = [c[0] for c in sorted_clicks]

# 过滤session长度为1的数据
for i in list(sess_clicks.keys()):
    if len(sess_clicks[i]) == 1:
        del sess_clicks[i]
        del sess_date[i]

# 记录每个商品id出现的次数
item_counts = {}
for s in list(sess_clicks.keys()):
    seq = sess_clicks[s]
    for item_id in seq:
        if item_id in item_counts:
            item_counts[item_id] += 1
        else:
            item_counts[item_id] = 1

# # 按照点击次数进行排序, 返回list
# sorted_counts = sorted(item_counts.items(), key=lambda x: x[1])

# 过滤掉序列中出现次数小于阈值的商品id
length = len(sess_clicks)
for s in list(sess_clicks.keys()):
    seq = sess_clicks[s]
    # 过滤掉序列中出现次数小于5的商品id
    filterseq = list(filter(lambda x: item_counts[x] >=5, seq))
    if len(filterseq) < 2:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filterseq

# 划分训练集和测试集
maxdate = max(list(sess_date.values()))
# 7天用来测试
splitdate = maxdate - 86400 * 7

train_sess = list(filter(lambda x: x[1] < splitdate, sess_date.items()))
test_sess = list(filter(lambda x: x[1] > splitdate, sess_date.items()))

train_sess = sorted(train_sess, key=lambda x: x[1])
test_sess = sorted(test_sess, key=lambda x: x[1])


print(len(train_sess))
print(len(test_sess))

# 重新编码，生成字典表
item_dict = {}
item_ctr = 1  # 初始化id值

# 生成训练数据
train_ids = []
train_seqs = []
train_dates = []
for s, date in train_sess:
    seq = sess_clicks[s]
    outseq= []
    for i in seq:
        if i in item_dict:
            outseq += [item_dict[i]]
        else:
            outseq += [item_ctr]
            item_dict[i] = item_ctr
            item_ctr += 1
    if len(outseq) < 2:
        continue

    train_ids += [s]
    train_seqs += [outseq]
    train_dates += [date]

print(item_ctr)

# 生成测试数据, 转变为序列, 忽略没出现在训练集中的item
test_ids = []
test_seqs = []
test_dates = []
for s, date in test_sess:
    seq = sess_clicks[s]
    outseq = []
    for i in seq:
        if i in item_dict:
            outseq += [item_dict[i]]
    if len(outseq) < 2:
        continue

    test_ids += [s]
    test_seqs += [outseq]
    test_dates += [date]

# 处理数据, 生成的数据格式为:(session_list, label_list)
def process_final_formal(seq_list, date_list):
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    for id, seq, date in zip(range(len(seq_list)), seq_list, date_list):
        for i in range(1, len(seq)):   # 单个session的序列长度
            target = seq[-i]
            labs += [target]    # 生成对应的target
            out_seqs += [seq[:-i]]
            out_dates += [date]
            ids += [id]
    return out_seqs, out_dates, labs, ids

train_seqs, train_dates, train_labs, train_ids = process_final_formal(train_seqs, train_dates)
test_seqs, test_dates, test_labs, test_ids = process_final_formal(test_seqs, test_dates)

# 进一步生成元组
train = (train_seqs, train_labs)
test = (test_seqs, test_labs)

pickle.dump(train, open('/home/wangwei/ctr_model/SR-GNN/train.txt', 'wb'))
pickle.dump(test, open('/home/wangwei/ctr_model/SR-GNN/test.txt', 'wb'))
'''
session id的点击序列：
[[1, 1, 2], [3, 3], [4, 5], [6, 7], [8, 9, 10, 10], [11, 12, 13], [14, 15, 16, 16, 17, 18, 19], [4, 20], [21, 22], [23, 24]]
生成的样本输入：
train_s[:10]
[[1, 1], [1], [3], [4], [6], [8, 9, 10], [8, 9], [8], [11, 12], [11]]
生成的标签
train_l[:10]
[2, 1, 3, 5, 7, 10, 10, 9, 13, 12]
'''