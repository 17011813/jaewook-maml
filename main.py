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
from maml import *
from tensorflow.python.platform import flags
import matplotlib.pylab as plt
from sklearn.preprocessing import MinMaxScaler
from data_generator import *
import math
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy.linalg as la

# from pytest import test

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'ILI', 'sinusoid or omniglot or miniimagenet or weather')
flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
# oracle means task id is input (only suitable for sinusoid)
flags.DEFINE_string('baseline', None, 'oracle, or None')

## Training options
flags.DEFINE_integer('pretrain_iterations', 1000, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 1000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('test_iterations', 1000, 'number of Metatest iterations.')
flags.DEFINE_integer('meta_batch_size', 32, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 10, 'number of examples used for inner gradient update (K for K-shot learning).')     #
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')
flags.DEFINE_integer('seq_len',20 , '  time length of inputs.')
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
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_bool('Do_meta', True, 'True to meta learning, False to original deep learning.')

flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', True, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot

train_rate =  FLAGS.train_rate
seq_len = FLAGS.seq_len
validation_rate =  FLAGS.validation_rate
pre_len = FLAGS.pre_len
meta_batch_size = FLAGS.meta_batch_size
target_node =  FLAGS.target_node

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

    train_x1 = np.reshape(train_x, (-1, 1, seq_len))
    train_y1 = np.reshape(train_y, (-1, 1, 1))
    test_x1 = np.reshape(test_x, (-1, 1, seq_len))
    test_y1 = np.reshape(test_y, (-1, 1, 1))

    print('train_x:', np.shape(train_x1))
    print('train_y:', np.shape(train_y1))
    print('test_x:', np.shape(test_x1))
    print('test_y:', np.shape(test_y1))

    return train_x1, train_y1, test_x1, test_y1

def generate_ILI_batch(train_x, train_y,  test_x, test_y):

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
        print(np.shape(data_batch1),np.shape(label_batch1),'train batch')
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
print(Mtrain_data.shape,"Meta train data")
Mtest_data = data.iloc[:,[target_node]]
print(Mtest_data.shape, "Meta test data, Target Node:", int(target_node))
Mtrain_data1 = np.mat(Mtrain_data, dtype=np.float32)
Mtest_data1 = np.mat(Mtest_data, dtype=np.float32)
scaler = MinMaxScaler()
scaler.fit(Mtrain_data1[0:int(len(Mtrain_data1))])
Mtrain_data1 = scaler.transform(Mtrain_data1)

scaler1 = MinMaxScaler()
scaler1.fit(Mtest_data1[0:int(len(Mtest_data1))])
Mtest_data1 = scaler1.transform(Mtest_data1)


# data_generator = DataGenerator(FLAGS.update_batch_size * 2, FLAGS.meta_batch_size,0,0,0,0)

def train(model, saver, sess, exp_string, resume_itr=0):
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 1000
    if FLAGS.datasource == 'ILI':
        PRINT_INTERVAL = 100#1000
        PRINT_FIG_INTERVAL = 100  # 1000
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
    else:
        PRINT_INTERVAL = 100
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []

    time_len = Mtrain_data.shape[0]
    num_nodes = Mtrain_data.shape[1]
    num_classes = num_nodes
    multitask_weights, reg_weights = [], []

    fout = open(str(FLAGS.logdir) + "_" + str(FLAGS.datasource) + "_train.txt", 'w')
    print(fout)
    fout2 = open(str(FLAGS.logdir) + "_" + str(FLAGS.datasource) + "_validate.txt", 'w')
    print(Mtrain_data1)
    train_x, train_y, test_x, test_y =  preprocess_data_forecasting1(Mtrain_data1, time_len, train_rate, validation_rate, seq_len, pre_len)
    train_x1, train_y1, test_x1, test_y1 =  preprocess_data_forecasting1(Mtest_data1, time_len, train_rate, validation_rate, seq_len, pre_len)

    metatrain_trainx, metatrain_trainy, metatrain_testx, metatrain_testy = generate_ILI_batch(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
    metatest_trainx, metatest_trainy, metatest_testx, metatest_testy = generate_ILI_batch(train_x=train_x1, train_y=train_y1, test_x=test_x1, test_y=test_y1)
    # print(metatrain_trainx)
    print(metatrain_trainx.shape, metatrain_trainy.shape,metatrain_testx.shape, metatrain_testy.shape,'Meta train data shapes')
    print(metatest_trainx.shape, metatest_trainy.shape,metatest_testx.shape, metatest_testy.shape,'Meta test data shapes')  # (18, 25, 10, 20) (18, 25, 1, 10)

    support_size_tr = int(metatrain_trainx.shape[0]/2)
    support_size_te = int(metatest_trainx.shape[0]/2)
    # print(metatrain_trainx.shape, metatrain_trainy.shape)  # (18, 25, 10, 20) (18, 25, 1, 10)
    for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
        feed_dict = {}
        for i in range(support_size_tr):
            inputa = metatrain_trainx[i, :, :]
            labela = metatrain_trainy[i, :,:]
            inputb = metatrain_testx[i, :, :] # b used for testing
            labelb = metatrain_testy[i, :, :]
            feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb}

            if itr < FLAGS.pretrain_iterations:
                input_tensors = [model.pretrain_op]
                # print("Pre-train") pre train은 loss a, 즉 support set에 대한 학습인데..
            else:
                input_tensors = [model.metatrain_op]
                # print("Meta train") train시 여기가 실행

            if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
                input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])

            result = sess.run(input_tensors, feed_dict) # 얘가 i 안에서 계속 실행이되어야하는거아닌가..
        # print(result,'result??')
        if itr % SUMMARY_INTERVAL == 0:
            prelosses.append(result[-2])
            if FLAGS.log:
                train_writer.add_summary(result[1], itr)
            postlosses.append(result[-1])

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            if itr < FLAGS.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
            print_str += ': prelosses - ' + str(np.mean(prelosses)) + ', postlosses - ' + str(np.mean(postlosses))
            print(print_str)
            fout.write("\rtrain_accuracy\t{}".format(print_str))

            prelosses, postlosses = [], []

        # 그냥 train에서 학습된 정도 보려고 plot 그리는 부분
        if (itr != 0) and itr % PRINT_FIG_INTERVAL == 0:
            # prediction from model
            model_preda = sess.run(model.outputas, feed_dict)
            # plot prediction
            # print(model_preda.shape,'model_preda')
            # plot_weather_temp(itr, inputa, labela, model_preda)


        if (itr!=0) and itr % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

        # Meta test validation
        if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0 and (FLAGS.datasource !='ILI' or FLAGS.datasource !='weather'):
            predb_list = []
            for j in range(support_size_te):
                inputa = metatest_trainx[j, :, :]
                labela = metatest_trainy[j, :,:]
                inputb = metatest_testx[j, :, :]  # b used for testing
                labelb = metatest_testy[j, :, :]
                feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}

                input_tensors = [model.total_loss1, model.total_losses2[FLAGS.num_updates-1]]

                result = sess.run(input_tensors, feed_dict)
                model_predb = sess.run(model.outputbs, feed_dict)
                predb_list.append(model_predb)
            predb_result = np.reshape(predb_list, (-1, 1))
            test_y2 = np.reshape(test_y1, (-1, 1))
            # print (predb_result.shape,'model_preda')
            test_labels = scaler1.inverse_transform(test_y2)
            predict = scaler1.inverse_transform(predb_result)
            print(predict.shape, test_labels.shape, 'predict.shape,test_labels.shape')
            rmse, mae, _, pcc, mape, Residual = evaluation(test_labels, predict)
            print('Test during Training!!','Test Iter:{}'.format(j), 'test_rmse: %r' % (rmse),
                  'test_pcc:{:.4}'.format(pcc),  # rmse
                  'test_mape:{:.4}'.format(mape),
                  'test_mae:{:.4}'.format(mae))
            print('Validation results: ' + str(result[0]) + ', ' + str(result[1]))
            fout2.write("\rvalidation_result\t{}\t{}".format(str(result[0]), str(result[1])))

    saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))
    fout.close()
    fout2.close()

            # calculated for omniglot
NUM_TEST_POINTS = 600
# testing
NUM_TEST_POINTS = 1


def test(model, saver, sess, exp_string, test_num_updates=None):
    num_classes = 1
    np.random.seed(1)
    random.seed(1)
    time_len = Mtest_data.shape[0]
    num_nodes = Mtest_data.shape[1]
    metaval_accuracies = []
    train_x1, train_y1, test_x1, test_y1 = preprocess_data_forecasting1(Mtest_data1, time_len, train_rate,
                                                                        validation_rate, seq_len, pre_len)
    metatest_trainx, metatest_trainy, metatest_testx, metatest_testy = generate_test_ILI_batch(train_x=train_x1,
                                                                                          train_y=train_y1,
                                                                                          test_x=test_x1,
                                                                                          test_y=test_y1)
    num_testbatch = int(metatest_trainx.shape[0])
    print(num_testbatch,'support_size_te')
    test_loss, test_rmse, test_mae, test_acc, test_pcc, test_var, test_pred, test_mape = [], [], [], [], [], [], [], []
    if FLAGS.Do_meta:
        iter = 50
    else:
        iter = FLAGS.test_iterations

    for j in range(iter):
        predb_list = []
        # losses = []
        feed_dict = {}
        for i in range(num_testbatch):
            inputa = metatest_trainx[i, :, :, :]
            labela = metatest_trainy[i, :, :, :]
            inputb = metatest_testx[i, :, :, :]  # b used for testing
            labelb = metatest_testy[i, :, :, :]

            feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb}

            # result = sess.run([model.metaval_total_loss1] +  model.metaval_total_losses2, feed_dict)
            # test_finetuning = sess.run([model.finetuning_op], feed_dict)
            # print(test_finetuning)
            # result_testlosses = sess.run(model.total_losses2, feed_dict)
            # metaval_accuracies.append(result)

            model_preda = sess.run(model.outputas, feed_dict)
            model_predb = sess.run(model.outputbs, feed_dict)
            predb_list.append(model_predb)
        if j % 10 == 0:
            predb_result1 = np.reshape(predb_list, (-1, 1))
            test_y2 = np.reshape(test_y1, (-1, 1))
            # print (predb_result.shape,'model_preda')
            test_labels = scaler1.inverse_transform(test_y2)
            predict = scaler1.inverse_transform(predb_result1)
            print (predict.shape,test_labels.shape, 'predict.shape,test_labels.shape')
            rmse, mae, _, pcc, mape, Residual = evaluation(test_labels,predict)
            test_rmse.append(rmse)  # rmse
            test_mae.append(mae)
            test_pcc.append(test_pcc)
            test_mape.append(test_mape)
            test_pred.append(predict)
            test_pred.append(predict)
            print('Test Iter:{}'.format(j), 'test_rmse: %r' % (rmse),
                  'test_pcc:{:.4}'.format(pcc),  # rmse
                  'test_mape:{:.4}'.format(mape),
                  'test_mae:{:.4}'.format(mae))
        test_finetuning = sess.run([model.finetuning_op], feed_dict)

    index = test_rmse.index(np.min(test_rmse))
    print('training_epoch:%r' % (FLAGS.test_iterations),
              'seq_len:%r' % (seq_len),
              'pre_len:%r' % (pre_len),
              'min_rmse:%r' % (np.min(test_rmse)),
              'min_mae:%r' % (test_mae[index]),
              'min_pcc:%r' % (test_pcc[index]),
              'best epoch:%r' % (index)
              )
    # plot test case
    # plot_weather_temp(0,inputa,labela,model_preda)
    # metaval_accuracies = np.array(metaval_accuracies)
    # means = np.mean(metaval_accuracies, 0)
    # stds = np.std(metaval_accuracies, 0)
    # ci95 = 1.96*stds/np.sqrt(NUM_TEST_POINTS)



    out_filename = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.csv'
    out_pkl = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.pkl'
    print('out_filename:',out_filename)
    print('out_pkl:',out_pkl)

    # with open(out_pkl, 'wb') as f:
    #     pickle.dump({'mses': metaval_accuracies}, f)
    # with open(out_filename, 'w') as f:
    #     writer = csv.writer(f, delimiter=',')
    #     writer.writerow(['update'+str(i) for i in range(len(means))])
    #     writer.writerow(means)
    #     writer.writerow(stds)
    #     writer.writerow(ci95)

def main():

    if  FLAGS.datasource == 'ILI':
        if FLAGS.train:
            test_num_updates = 5
        else:
            test_num_updates = 1 # 원래는 10이었지만.. 굳이...?


    # if FLAGS.train == False:
    #     orig_meta_batch_size = FLAGS.meta_batch_size
    #     # always use meta batch size of 1 when testing.
    #     FLAGS.meta_batch_size = 1


    # dim_output = data_generator.dim_output
    # dim_input = data_generator.dim_input
    dim_output = 1
    dim_input = seq_len
    # print(dim_input,'dim_input')
    tf_data_load = False
    input_tensors = None

    model = MAML(dim_input, dim_output, test_num_updates=test_num_updates, num_nodes = data.shape[1]-1)
    if FLAGS.train:
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    elif FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        FLAGS.meta_batch_size = 1
        model.construct_model(input_tensors=input_tensors, prefix='metaval_')

    # if tf_data_load:
    #     model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
    model.summ_op = tf.summary.merge_all()

    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)

    sess = tf.InteractiveSession()

    if FLAGS.train == False:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    exp_string = 'cls_'+str(FLAGS.num_classes)+'.mbs_'+str(FLAGS.meta_batch_size) + '.ubs_' + str(FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(FLAGS.train_update_lr)

    if FLAGS.num_filters != 64:
        exp_string += 'hidden' + str(FLAGS.num_filters)
    if FLAGS.max_pool:
        exp_string += 'maxpool'
    if FLAGS.stop_grad:
        exp_string += 'stopgrad'
    if FLAGS.baseline:
        exp_string += FLAGS.baseline
    if FLAGS.norm == 'batch_norm':
        exp_string += 'batchnorm'
    elif FLAGS.norm == 'layer_norm':
        exp_string += 'layernorm'
    elif FLAGS.norm == 'None':
        exp_string += 'nonorm'
    else:
        print('Norm setting not recognized.')



    resume_itr = 0
    model_file = None

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    if FLAGS.resume or not FLAGS.train: # resume 꺼놓으면 무조건 처음부터 다시 train
        # print("test???")
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        if FLAGS.Do_meta:
            if FLAGS.test_iter > 0:
                model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
                print("111111Restoring model weights from " + model_file)
            if model_file:
                ind1 = model_file.index('model')
                resume_itr = int(model_file[ind1+5:])
                print("Restoring model weights from " + model_file)
                print(ind1,resume_itr,'ind1, resume_itr')
                saver.restore(sess, model_file)
        else:
            model_file = None

    if FLAGS.train:
        train(model, saver, sess, exp_string, resume_itr)
    else:
        test(model, saver, sess, exp_string, test_num_updates)


if __name__ == "__main__":
    main()

