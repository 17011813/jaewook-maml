"""
Usage Instructions:
    10-shot weather:
        python main.py --datasource=weather --logdir=logs/weather_prev/ --metatrain_iterations=70000 --norm=None --update_batch_size=10
        python main.py --datasource=weather --train=False --test_set=True --logdir=logs/weather_prev/ --metatrain_iterations=70000 --norm=None --update_batch_size=10



    To run evaluation, use the '--train=False' flag and the '--test_set=True' flag to use the test set.

    For omniglot and miniimagenet training, acquire the dataset online, put it in the correspoding data directory, and see the python script instructions in that directory to preprocess the data.
"""
import csv
import numpy as np
import pickle
import random
import tensorflow as tf
import pandas as pd
from data_generator import DataGenerator
from maml2 import MAML
from tensorflow.python.platform import flags
import matplotlib.pylab as plt
from sklearn.preprocessing import MinMaxScaler
from data_generator import *
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy.linalg as la
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

# from pytest import test

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'ILI', 'sinusoid or omniglot or miniimagenet or weather')
flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
# oracle means task id is input (only suitable for sinusoid)
flags.DEFINE_string('baseline', None, 'oracle, or None')

## Training options
flags.DEFINE_integer('pretrain_iterations', 1000, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 1000,
                     'number of metatraining iterations.')  # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('test_iterations', 1000, 'number of Metatest iterations.')
flags.DEFINE_integer('meta_batch_size', 32, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 10,
                     'number of examples used for inner gradient update (K for K-shot learning).')  #
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.')  # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')
flags.DEFINE_integer('seq_len', 20, '  time length of inputs.')
flags.DEFINE_integer('pre_len', 5, 'time length of prediction.')
flags.DEFINE_float('train_rate', 0.5, 'rate of training set.')
flags.DEFINE_float('validation_rate', 0.1, 'rate of validation rate.')
flags.DEFINE_float('target_node', 1, 'Meta test node.')

## Model options
flags.DEFINE_string('norm', 'None', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', 'logs/weather_prev/', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', False, 'resume training if there is a model available')
flags.DEFINE_bool('train', False, 'True to train, False to test.')
flags.DEFINE_bool('Do_meta', False, 'True to meta learning, False to original deep learning.')

flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', True, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1,
                     'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1,
                   'value of inner gradient step step during training. (use if you want to test with a different value)')  # 0.1 for omniglot

train_rate = FLAGS.train_rate
seq_len = FLAGS.seq_len
validation_rate = FLAGS.validation_rate
pre_len = FLAGS.pre_len
meta_batch_size = FLAGS.meta_batch_size
target_node = FLAGS.target_node


def plot_weather_temp(itr, inputa, labela, model_preda):
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(inputa, labela, color='black')
    ax.scatter(inputa, model_preda, color='red')
    plt.xlabel("month")
    plt.ylabel("temperature")
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.2)
    plt.ylim([0, 35])
    plt.savefig(
        str(FLAGS.logdir) + "_" + str(FLAGS.datasource) + "_train_" + str(FLAGS.train) + "_" + str(itr) + ".png")
    plt.close(fig)
    return


def preprocess_data_forecasting(data, time_len, rate, validation_rate, seq_len, pre_len):
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


def preprocess_data_forecasting1(data, time_len, rate, validation_rate, seq_len, pre_len):
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
    validation_size = int(data_len * validation_rate)

    train_x = np.array(total_x[0:train_size])
    train_y = np.array(total_y[0:train_size])
    test_x = np.array(total_x[train_size:])
    test_y = np.array(total_y[train_size:])

    train_x1 = np.reshape(train_x, (-1, seq_len))
    train_y1 = np.reshape(train_y, (-1, 1))
    test_x1 = np.reshape(test_x, (-1, seq_len))
    test_y1 = np.reshape(test_y, (-1, 1))

    print('train_x:', np.shape(train_x1))
    print('train_y:', np.shape(train_y1))
    print('test_x:', np.shape(test_x1))
    print('test_y:', np.shape(test_y1))

    return train_x1, train_y1, test_x1, test_y1


def generate_ILI_batch(train_x, train_y, test_x, test_y):
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
    print(np.shape(data_batch1), np.shape(label_batch1), 'train batch')
    return data_batch1, label_batch1, data_batch2, label_batch12


def generate_ILI_batch_dim1(train_x, train_y, test_x, test_y):
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


data = pd.read_csv('./data/ILI_region_feature_2002-2020.csv')

Mtrain_data = data.drop(columns=[str(int(target_node))])
print(Mtrain_data.shape, "Meta train data")
Mtest_data = data.iloc[:, [target_node]]
print(Mtest_data.shape, "Meta test data, Target Node:", int(target_node))
Mtrain_data1 = np.mat(Mtrain_data, dtype=np.float32)
Mtest_data1 = np.mat(Mtest_data, dtype=np.float32)

scaler = MinMaxScaler()
scaler.fit(Mtrain_data1[0:int(len(Mtrain_data1)*FLAGS.train_rate)])
Mtrain_data1 = scaler.transform(Mtrain_data1)
# print(Mtrain_data1,'SSSSSSSSSSSSS')
scaler1 = MinMaxScaler()
scaler1.fit(Mtest_data1[0:int(len(Mtest_data1)*FLAGS.train_rate)])
Mtest_data1 = scaler1.transform(Mtest_data1)


# data_generator = DataGenerator(FLAGS.update_batch_size * 2, FLAGS.meta_batch_size,0,0,0,0)


time_len = Mtrain_data.shape[0]
num_nodes = Mtrain_data.shape[1]
num_classes = num_nodes
multitask_weights, reg_weights = [], []

fout = open(str(FLAGS.logdir) + "_" + str(FLAGS.datasource) + "_train.txt", 'w')
print(fout)
fout2 = open(str(FLAGS.logdir) + "_" + str(FLAGS.datasource) + "_validate.txt", 'w')
print(Mtrain_data1)
train_x, train_y, test_x, test_y = preprocess_data_forecasting1(Mtrain_data1, time_len, train_rate, validation_rate,
                                                                    seq_len, pre_len)
train_x1, train_y1, test_x1, test_y1 = preprocess_data_forecasting1(Mtest_data1, time_len, train_rate,
                                                                        validation_rate, seq_len, pre_len)
# scaler = MinMaxScaler()
# scaler.fit(train_x1)
# train_x1 = scaler.transform(train_x1)
# test_x = scaler.transform(test_x)
# test_x=pd.DataFrame(test_x)
# test_x.to_csv('sex12.csv')
# scaler1 = MinMaxScaler()
# scaler1.fit(train_y)
# train_y = scaler1.transform(train_y)

regressor = MLPRegressor(hidden_layer_sizes=(10, 10, 10, 10))
# regressor = RandomForestRegressor(n_estimators=400)
# for i in range(len(train_y1)):
#     train_y1[i] = int(train_y1[i]*10)
# print(train_y1)
# train_y1[1] = 8
# train_y1[2] = 9
# train_y1[3] = 7
#
# regressor = RandomForestClassifier(max_depth=17)
regressor.fit(train_x1, train_y1.ravel())
# new_label = regressor.predict_proba((train_x1))
# print(new_label.shape)
# reg2 = RandomForestRegressor()
# reg2.fit(train_x1, new_label.ravel())

Y_pred = regressor.predict(test_x1)
Y_pred = np.reshape(Y_pred, (-1, 1))
Y_pred = scaler1.inverse_transform(Y_pred)
test_y2 = np.reshape(test_y1, (-1, 1))
test_labels = scaler1.inverse_transform(test_y1)

rmse, mae, _, pcc, mape, Residual = evaluation(test_labels, Y_pred)
print('test_rmse: %r' % (rmse),
                  'test_pcc:{:.4}'.format(pcc),  # rmse
                  'test_mape:{:.4}'.format(mape),
                  'test_mae:{:.4}'.format(mae))

