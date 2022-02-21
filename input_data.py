import csv
import numpy as np
import pickle
import random
import tensorflow as tf
import pandas as pd
from data_generator import DataGenerator
from maml2 import *
from tensorflow.python.platform import flags
import matplotlib.pylab as plt
from sklearn.preprocessing import MinMaxScaler
# from data_generator import *
import math
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy.linalg as la
FLAGS = flags.FLAGS

def plot_weather_temp(itr, inputa, labela, model_preda):
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(inputa, labela, color='black')
    ax.scatter(inputa, model_preda, color='red')
    plt.xlabel("month")
    plt.ylabel("temperature")
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.2)
    plt.ylim([0,35])
    plt.savefig(str(FLAGS.logdir) + "_" + str(FLAGS.datasource) + "_train_" + str(FLAGS.train) + "_" + str(itr) + ".png")
    plt.close(fig)
    return


def preprocess_data_forecasting(data, time_len, rate, validation_rate, seq_len, pre_len,meta_batch_size):
    total_x = list()
    total_y = list()
    scaler = MinMaxScaler()
    time_len = data.shape[0]
    num_nodes = data.shape[1]
    print(time_len, num_nodes)
    data1 = np.mat(data, dtype=np.float32)
    scaler.fit(data1[0:int(len(data1))])
    data1 = scaler.transform(data1)

    for i in range(time_len - seq_len - pre_len):
        a = data1[i: i + seq_len + pre_len]
        total_x.append(np.transpose(a[0: seq_len]))
        total_y.append(np.transpose(a[[seq_len + pre_len - 1]]))

    data_len = len(total_x)

    train_size = int(data_len * rate)
    validation_size = int(data_len * validation_rate)

    train_x = np.array(total_x[0:train_size])
    train_y = np.array(total_y[0:train_size])
    test_x = np.array(total_x[train_size:])
    test_y = np.array(total_y[train_size:])

    print('train_x:', np.shape(train_x))
    print('train_y:', np.shape(train_y))
    print('test_x:', np.shape(test_x))
    print('test_y:', np.shape(test_y))

    return train_x, train_y, test_x, test_y


def preprocess_data_forecasting1(data, time_len, rate, validation_rate, seq_len, pre_len,meta_batch_size):
    total_x = list()
    total_y = list()
    time_len = data.shape[0]
    num_nodes = data.shape[1]
    print(time_len, num_nodes)

    for i in range(time_len - seq_len - pre_len):
        a = data[i: i + seq_len + pre_len]
        total_x.append(np.transpose(a[0: seq_len]))
        total_y.append(np.transpose(a[[seq_len + pre_len - 1]]))

    data_len = len(total_x)

    train_size = int(data_len * rate)
    print(train_size)
    validation_size = int(data_len * validation_rate)

    train_x = np.array(total_x[0:train_size])
    train_y = np.array(total_y[0:train_size])
    test_x = np.array(total_x[train_size:])
    test_y = np.array(total_y[train_size:])
    if not len(train_x) == len(test_x):  # 홀수여서 안맞을때 일단 맞추려고 해놓음.. train test가 갯수가 안맞아도 돌아가야할텐데말여..
        train_x = np.array(total_x[0:train_size])
        train_y = np.array(total_y[0:train_size])
        test_x = np.array(total_x[train_size:-1])
        test_y = np.array(total_y[train_size:-1])

    train_x1 = np.reshape(train_x, (-1, 1, seq_len))
    train_y1 = np.reshape(train_y, (-1, 1, 1))
    test_x1 = np.reshape(test_x, (-1, 1, seq_len))
    test_y1 = np.reshape(test_y, (-1, 1, 1))

    print('train_x:', np.shape(train_x1))
    print('train_y:', np.shape(train_y1))
    print('test_x:', np.shape(test_x1))
    print('test_y:', np.shape(test_y1))

    return train_x1, train_y1, test_x1, test_y1


def generate_ILI_batch(train_x, train_y, test_x, test_y,meta_batch_size):
    len_data = np.shape(train_x)[0]
    temp_ind = np.arange(np.shape(train_x)[0])
    len_data1 = np.shape(test_x)[0]
    # temp_ind1 = np.arange(np.shape(test_x)[0])
    num_node = np.shape(train_x)[1]
    np.random.shuffle(temp_ind)
    np.random.shuffle(temp_ind)

    temp_trainX = train_x[temp_ind]
    temp_trainY = train_y[temp_ind]
    temp_testX = test_x[temp_ind]
    temp_testY = test_y[temp_ind]
    i = 0
    j = 0
    data_batch = []
    label_batch = []
    data_batch_test = []
    label_batch_test = []
    while i < len_data:
        if i + meta_batch_size > len_data:
            break
        mini_batch = temp_trainX[i:i + meta_batch_size]
        mini_label = temp_trainY[i:i + meta_batch_size]
        data_batch.append(mini_batch)
        label_batch.append(mini_label)
        i += meta_batch_size
    while j < len_data1:
        if j + meta_batch_size > len_data1:
            break
        mini_batch1 = temp_testX[j:j + meta_batch_size]
        mini_label1 = temp_testY[j:j + meta_batch_size]
        data_batch_test.append(mini_batch1)
        label_batch_test.append(mini_label1)
        j += meta_batch_size

    data_batch1 = np.asarray(data_batch)
    label_batch1 = np.asarray(label_batch)
    data_batch2 = np.asarray(data_batch_test)
    label_batch12 = np.asarray(label_batch_test)
    print(np.shape(data_batch1), np.shape(label_batch1), np.shape(data_batch2), np.shape(label_batch12), 'train batch')
    return data_batch1, label_batch1, data_batch2, label_batch12


def generate_ILI_batch_dim1(train_x, train_y, test_x, test_y,meta_batch_size):
    len_data = np.shape(train_x)[0]
    temp_ind = np.arange(np.shape(train_x)[0])
    len_data1 = np.shape(test_x)[0]
    temp_ind1 = np.arange(np.shape(test_x)[0])
    num_node = np.shape(train_x)[1]

    np.random.shuffle(temp_ind)
    np.random.shuffle(temp_ind1)

    temp_trainX = train_x[temp_ind]
    temp_trainY = train_y[temp_ind]  # 이거 타겟노드 아닌걸로 해야되는데
    temp_testX = test_x[temp_ind1]
    temp_testY = test_y[temp_ind1]
    i = 0
    j = 0
    data_batch = []
    label_batch = []
    data_batch_test = []
    label_batch_test = []
    while i < len_data:
        if i + meta_batch_size > len_data:
            break
        mini_batch = temp_trainX[i:i + meta_batch_size]
        mini_label = temp_trainY[i:i + meta_batch_size]
        data_batch.append(mini_batch)
        label_batch.append(mini_label)
        i += meta_batch_size
    while j < len_data1:
        if j + meta_batch_size > len_data1:
            break
        mini_batch1 = temp_testX[j:j + meta_batch_size]
        mini_label1 = temp_testY[j:j + meta_batch_size]
        data_batch_test.append(mini_batch1)
        label_batch_test.append(mini_label1)
        j += meta_batch_size
    data_batch1 = np.asarray(data_batch)
    label_batch1 = np.asarray(label_batch)
    data_batch2 = np.asarray(data_batch_test)
    label_batch12 = np.asarray(label_batch_test)
    # print(np.shape(data_batch1), np.shape(label_batch1), 'train batch')
    return data_batch1, label_batch1, data_batch2, label_batch12


def generate_test_ILI_batch(train_x, train_y, test_x, test_y):
    len_data = np.shape(train_x)[0]
    len_data1 = np.shape(test_x)[0]
    num_node = np.shape(train_x)[1]

    i = 0
    j = 0
    data_batch = []
    label_batch = []
    data_batch_test = []
    label_batch_test = []
    while i < len_data:
        if i + 1 > len_data:
            break
        mini_batch = train_x[i:i + 1]
        mini_label = train_y[i:i + 1]
        data_batch.append(mini_batch)
        label_batch.append(mini_label)
        i += 1
    while j < len_data1:
        if j + 1 > len_data1:
            break
        mini_batch1 = test_x[j:j + 1]
        mini_label1 = test_y[j:j + 1]
        data_batch_test.append(mini_batch1)
        label_batch_test.append(mini_label1)
        j += 1
    data_batch1 = np.asarray(data_batch)
    label_batch1 = np.asarray(label_batch)
    data_batch2 = np.asarray(data_batch_test)
    label_batch12 = np.asarray(label_batch_test)
    print(np.shape(data_batch1), np.shape(label_batch1), 'test batch')
    return data_batch1, label_batch1, data_batch2, label_batch12


def evaluation(a, b):
    rmse = math.sqrt(mean_squared_error(a, b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a - b, 'fro') / la.norm(a, 'fro')

    a_mean = np.mean(a, axis=0)
    b_mean = np.mean(b, axis=0)
    t1 = np.sum(np.multiply(a - a_mean, b - b_mean), axis=0)
    t2 = np.multiply(np.sqrt(np.sum(np.multiply(a - a_mean, a - a_mean), axis=0)),
                     np.sqrt(np.sum(np.multiply(b - b_mean, b - b_mean), axis=0)))
    pcc = np.mean(np.divide(t1, t2 + 1.0e-20))

    ab = np.absolute(np.subtract(a, b))

    div_a = np.where(a == 0, np.ones_like(a), a)
    mape = np.mean(100 * np.divide(ab, div_a))
    Residual = pd.DataFrame(b_mean - a_mean)

    return rmse, mae, 1 - F_norm, pcc, mape, Residual