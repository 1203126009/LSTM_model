#!/usr/bin/python
# -*- coding:utf-8 -*-


#import jieba
#import collections
#import string
import json
#from zhon.hanzi import punctuation
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tensorflow.contrib import keras as kr

def prepare_data(feas, dim=27, sequence_length=100):
    data = np.zeros([dim, sequence_length])
    h, w = feas.shape
    if h != dim:
        print("dim error\n")
        return data
    if w <= sequence_length:
        for seq_index, seq in enumerate(feas):
            data[seq_index, :len(seq)] = seq
    else:
        for seq_index, seq in enumerate(data):
            data[seq_index, :len(data)] = feas[seq_index, :len(data)]
    return data


def load_data_and_labels(train_data_file, dim, sequence_length):
    """
    :param train_data_file:
    :param test_data_file:
    :param sequence_length:
    :param vocabulary_size:
    :return:
    """
    words = []
    contents = []
    train_datas = []
    test_datas = []
    labels = []
    # 生成训练数据集
    with open(train_data_file, 'r') as f:
        train_datas=[]
        train_labels=[]
        test_datas = []
        test_labels = []
        count = 0
        for line in f.readlines():
            item = json.loads(line)
            label = int(item['label'])
            feas = np.array(item['data']).T
            #print(feas.shape)
            data = prepare_data(feas, dim, sequence_length)
            if count % 5 == 0:
                test_labels.append(label)
                test_datas.append(data.T)
            else:
                train_datas.append(data.T)
                train_labels.append(label)
            count += 1

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(np.array(train_labels))
    onehot_encoder = OneHotEncoder(sparse=False)
    # shape: [None, num_classes]
    train_labels = onehot_encoder.fit_transform(integer_encoded.reshape(len(integer_encoded), 1))

    integer_encoded = label_encoder.fit_transform(np.array(test_labels))
    onehot_encoder = OneHotEncoder(sparse=False)
    test_labels = onehot_encoder.fit_transform(integer_encoded.reshape(len(integer_encoded), 1))

    return train_datas, train_labels, test_datas, test_labels


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset
    :param data:
    :param batch_size:
    :param num_epochs:
    :param shuffle:
    :return:
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index: end_index]


if __name__ == '__main__':
    train_datas, train_labels, test_datas, test_labels = \
        load_data_and_labels('./train_data.json', 27, 100)
    print(len(train_datas), len(train_labels))
    print(len(test_datas), len(test_labels))
    # print(word2id['UNK'], word2id['PAD'])
    # print(train_datas.shape, train_labels.shape, test_datas.shape, test_labels.shape)
    batchs = batch_iter(list(zip(train_datas, train_labels)), 64, 1)
    for i, batch in enumerate(batchs):
        x_batch, y_batch = zip(*batch)
        print(i, len(x_batch), len(y_batch))
        print(y_batch)
