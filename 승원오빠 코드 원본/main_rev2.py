import csv
import numpy as np
import pickle
import random
import tensorflow as tf
import pandas as pd
from data_generator import DataGenerator
from maml_rev2 import MAML
from input_data import *
from tensorflow.python.platform import flags
import matplotlib.pylab as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from data_generator import *
import math
import seaborn as sns
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy.linalg as la
from sklearn.manifold import TSNE
from numpy import dot
from numpy.linalg import norm

# from pytest import test

FLAGS = flags.FLAGS

## Dataset/method options
# flags.DEFINE_string('datasource', 'covid', "'ILI' or 'covid'")
flags.DEFINE_string('datasource', 'ILI', "'ILI' or 'covid'")
flags.DEFINE_string('meta_data', 'Region', "'Region', 'Japan', or 'States' ")
flags.DEFINE_string('target_data', 'Region', "'Region', 'Japan', or 'States' ")
flags.DEFINE_bool('drop_target', False, 'whether to drop target data in training set')
flags.DEFINE_bool('interpretability', True, 'whether to drop target data in training set')
flags.DEFINE_integer('target_node', 0, 'index of target node')

flags.DEFINE_bool('taskpool', True, 'whether to use a taskpool')
flags.DEFINE_integer('pruning_interval', 20, 'whether to use a taskpool')
flags.DEFINE_float('pruning_threshold', 0.0, 'whether to use a taskpool')

## Training options
flags.DEFINE_integer('pretrain_iterations', 40, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 300, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
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

## Model options
flags.DEFINE_string('norm', 'None', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', 'logs/weather_prev/', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_bool('Do_meta', True, 'True to meta learning, False to original deep learning.')

flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', True, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot

meta_data_name = FLAGS.meta_data
target_data_name = FLAGS.target_data

seq_len = FLAGS.seq_len
pre_len = FLAGS.pre_len

train_rate = FLAGS.train_rate
validation_rate = FLAGS.validation_rate
meta_batch_size = FLAGS.meta_batch_size

if FLAGS.datasource == "ILI":
    if meta_data_name == "Region":
        meta_data = pd.read_csv('./data/ILI_region_feature_2002-2020.csv')
    elif meta_data_name == "States":
        meta_data = pd.read_csv('./data/ILI_states_feature_2010-2020.csv')
    elif meta_data_name == "Japan":
        meta_data = pd.read_csv('./data/ILI_japan_feature_2012-2020.csv')

    if target_data_name == "Region":
        target_data = pd.read_csv('./data/ILI_region_feature_2002-2020.csv')
    elif target_data_name == "States":
        target_data = pd.read_csv('./data/ILI_states_feature_2010-2020.csv')
    elif target_data_name == "Japan":
        target_data = pd.read_csv('./data/ILI_japan_feature_2012-2020.csv')
else:
    meta_data = pd.read_csv('./data/new_confirmed_no_minus.csv')
    target_data = pd.read_csv('./data/new_confirmed_no_minus.csv')

target_node = FLAGS.target_node

if meta_data_name == target_data_name:
    if FLAGS.drop_target:
        meta_train_data = meta_data.drop(columns=[str(int(target_node))])
    else:
        meta_train_data = meta_data

meta_test_data = target_data.iloc[:, [target_node]]

print(meta_train_data.shape, "Meta train data")
print(meta_test_data.shape, "Meta test data, Target Node:", int(target_node))

time_len = meta_train_data.shape[0]
meta_num_nodes = meta_train_data.shape[1]
target_num_nodes = target_data.shape[1]

data_list = None

if FLAGS.interpretability:
    if target_data_name == "Region":
        data = pd.read_csv('./data/ILI_region_feature_2002-2020.csv')
    elif target_data_name == "States":
        data = pd.read_csv('./data/ILI_states_feature_2010-2020.csv')
    elif target_data_name == "Japan":
        data = pd.read_csv('./data/ILI_japan_feature_2012-2020.csv')
    else:
        data = pd.read_csv('./data/new_confirmed_no_minus.csv')

    data_list = list()
    for i in range(target_num_nodes):
        temp_data = data.iloc[:, [i]]
        temp_data = np.mat(temp_data, dtype=np.float32)

        temp_scaler = MinMaxScaler()
        temp_scaler.fit(temp_data[0:int(time_len * FLAGS.train_rate)])
        temp_data = temp_scaler.transform(temp_data)

        data_list.append(temp_data)

meta_train_train_x_list = list()
meta_train_train_y_list = list()
meta_train_test_x_list = list()
meta_train_test_y_list = list()

for i in range(meta_num_nodes):
    temp_data = meta_train_data.iloc[:, [i]]
    temp_data = np.mat(temp_data, dtype=np.float32)

    temp_scaler = MinMaxScaler()
    temp_scaler.fit(temp_data[0:int(time_len * FLAGS.train_rate)])
    temp_data = temp_scaler.transform(temp_data)

    train_train_x, train_train_y, train_test_x, train_test_y = preprocess_data_forecasting(temp_data, time_len,
                                                                                        0.5, validation_rate,
                                                                                        seq_len, pre_len,
                                                                                        meta_batch_size)

    meta_train_train_x, meta_train_train_y, meta_train_test_x, meta_train_test_y = generate_ILI_batch(train_x=train_train_x,
                                                                                                  train_y=train_train_y,
                                                                                                  test_x=train_test_x,
                                                                                                  test_y=train_test_y, meta_batch_size=meta_batch_size)

    meta_train_train_x_list.append(meta_train_train_x)
    meta_train_train_y_list.append(meta_train_train_y)
    meta_train_test_x_list.append(meta_train_test_x)
    meta_train_test_y_list.append(meta_train_test_y)

meta_test_data = np.mat(meta_test_data, dtype=np.float32)

scaler = MinMaxScaler()
scaler.fit(meta_test_data[0:int(time_len * FLAGS.train_rate)])
meta_test_data = scaler.transform(meta_test_data)


def cos_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))


def train(model, saver, sess, exp_string, resume_itr=0):
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 1000
    PRINT_INTERVAL = 20
    TEST_PRINT_INTERVAL = PRINT_INTERVAL

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/'+target_data_name+'/'+ str(target_node)+ '/' + exp_string, sess.graph)

    print('Done initializing, starting training.')

    pre_losses, post_losses = [], []

    fout = open(str(FLAGS.logdir) + "_" + str(FLAGS.datasource) + "_train.txt", 'w')
    print(fout)
    fout2 = open(str(FLAGS.logdir) + "_" + str(FLAGS.datasource) + "_validate.txt", 'w')

    test_train_x, test_train_y, test_test_x, test_test_y = preprocess_data_forecasting(meta_test_data, time_len,
                                                                                       train_rate, validation_rate,
                                                                                       seq_len, pre_len,
                                                                                       meta_batch_size)

    meta_test_train_x_val, meta_test_train_y_val, meta_test_test_x_val, meta_test_test_y_val = generate_test_ILI_batch(
        train_x=test_train_x,
        train_y=test_train_y,
        test_x=test_test_x,
        test_y=test_test_y)

    is_in_taskpool = None
    PRUNING_INTERVAL = None
    if FLAGS.taskpool:
        is_in_taskpool = [True for k in range(meta_num_nodes)]
        PRUNING_INTERVAL = FLAGS.pruning_interval

    for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
        print("--- %d th iteration ---" % itr)
        for k in range(meta_num_nodes):
            if FLAGS.taskpool:
                if is_in_taskpool[k] is False:
                    continue

            print("------------------------- %d th task starts --------------------" % k)
            meta_train_train_x = meta_train_train_x_list[k]
            meta_train_train_y = meta_train_train_y_list[k]

            support_size_tr = int(meta_train_train_x.shape[0])

            for i in range(support_size_tr):
                sample_ind = np.random.randint(0, support_size_tr)
                input_a = meta_train_train_x[i, :, :, :]
                label_a = meta_train_train_y[i, :, :, :]
                input_b = meta_train_train_x[sample_ind, :, :, :]
                label_b = meta_train_train_y[sample_ind, :, :, :]
                feed_dict = {model.inputa: input_a, model.inputb: input_b,  model.labela: label_a, model.labelb: label_b}

                if itr < FLAGS.pretrain_iterations:
                    input_tensors = [model.pretrain_op]
                else:
                    input_tensors = [model.metatrain_op]

                if itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0:
                    input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])

                result = sess.run(input_tensors, feed_dict)

        if itr % SUMMARY_INTERVAL == 0:
            pre_losses.append(result[-2])
            if FLAGS.log:
                train_writer.add_summary(result[1], itr)
            post_losses.append(result[-1])

        if itr != 0 and itr % PRINT_INTERVAL == 0:
            if itr < FLAGS.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
            print_str += ': prelosses - ' + str(np.mean(pre_losses)) + ', postlosses - ' + str(np.mean(post_losses))
            print(print_str)
            fout.write("\rtrain_accuracy\t{}".format(print_str))

            pre_losses, post_losses = [], []

        if itr != 0 and itr % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + target_data_name+'/' + str(target_node) + '/' + exp_string + '/model' + str(itr))

        if FLAGS.taskpool:
            if itr >= FLAGS.pretrain_iterations and itr % PRUNING_INTERVAL == 0:
                grad_list = list()
                grad_index_list = list()
                for k in range(meta_num_nodes):
                    if FLAGS.taskpool:
                        if is_in_taskpool[k] is False:
                            continue

                    print("------------------------- %d th task starts --------------------" % k)
                    meta_train_train_x = meta_train_train_x_list[k]
                    meta_train_train_y = meta_train_train_y_list[k]

                    support_size_tr = int(meta_train_train_x.shape[0])

                    temp_grad_list = list()
                    for i in range(support_size_tr):
                        input_a = meta_train_train_x[i, :, :, :]
                        label_a = meta_train_train_y[i, :, :, :]
                        feed_dict = {model.inputa: input_a, model.labela: label_a, model.meta_lr: 0.0}

                        batch_gr = sess.run(model.gra_a, feed_dict)
                        temp_grad_list.append(batch_gr)
                    temp_grad = np.sum(temp_grad_list, axis=0)
                    temp_grad = temp_grad[:, 0:1]
                    # length = np.shape(temp_grad)[0]
                    # temp_grad = temp_grad[length-2:length-1]
                    temp_grad = np.reshape(temp_grad, newshape=[-1])
                    flatten = list()
                    for k in range(np.shape(temp_grad)[0]):
                        for_flatten = temp_grad[k]
                        for_flatten = np.reshape(for_flatten, newshape=[-1])
                        flatten.append(for_flatten)
                    flatten = np.concatenate(flatten, axis=0)
                    print(flatten)
                    grad_list.append(flatten)
                    grad_index_list.append(k)
                grad_data = np.array(grad_list)

                standard_scaler = StandardScaler()
                grad_data = standard_scaler.fit_transform(grad_data)

                new_ind = grad_index_list.index(target_node)

                for n2 in range(len(grad_index_list)):
                    cosine = cos_sim(grad_data[new_ind], grad_data[n2])

                    if cosine < FLAGS.pruning_threshold:
                        print(cosine)
                        region_ind = grad_index_list(n2)
                        print("-------------------- region %d deleted ---------------------" % region_ind)
                        is_in_taskpool[region_ind] = False


        # Meta test validation
        if itr != 0 and itr % TEST_PRINT_INTERVAL == 0:
            predb_list = []

            num_testbatch = int(meta_test_train_x_val.shape[0])
            for i in range(1):
                for j in range(num_testbatch):
                    inputa = meta_test_train_x_val[j, :, :, :]
                    labela = meta_test_train_y_val[j, :, :, :]
                    feed_dict1 = {model.inputa: inputa, model.labela: labela}
                    test_finetuning = sess.run([model.pretrain_op], feed_dict1)

            predb_list = []
            for v in range(int(meta_test_test_x_val.shape[0])):
                inputb = meta_test_test_x_val[v, :, :, :]  # b used for testing
                labelb = meta_test_test_y_val[v, :, :, :]
                feed_dict = {model.inputa: inputb, model.labela: labelb}
                model_predb = sess.run(model.outputas, feed_dict)
                predb_list.append(model_predb)
            predb_result1 = np.reshape(predb_list, (-1, 1))
            test_y2 = np.reshape(test_test_y, (-1, 1))
            test_labels = scaler.inverse_transform(test_y2)
            predict = scaler.inverse_transform(predb_result1)
            print(predict.shape, test_labels.shape, 'predict.shape,test_labels.shape')
            rmse, mae, _, pcc, mape, residual = evaluation(test_labels, predict)
            print('Test Iter:{}'.format(itr), 'test_rmse: %r' % rmse,
                  'test_pcc:{:.4}'.format(pcc),  # rmse
                  'test_mape:{:.4}'.format(mape),
                  'test_mae:{:.4}'.format(mae))

    saver.save(sess, FLAGS.logdir + '/' + target_data_name + '/' + str(target_node) +'/' + exp_string + '/model')
    fout.close()
    fout2.close()


def test(model, sess, exp_string):
    np.random.seed(1)
    random.seed(1)

    test_train_x, test_train_y, test_test_x, test_test_y = preprocess_data_forecasting(meta_test_data, time_len,
                                                                                        train_rate, validation_rate,
                                                                                        seq_len, pre_len,
                                                                                        meta_batch_size)

    meta_test_train_x, meta_test_train_y, meta_test_test_x, meta_test_test_y = generate_ILI_batch(train_x=test_train_x,
                                                                                                  train_y=test_train_y,
                                                                                                  test_x=test_test_x,
                                                                                                  test_y=test_test_y,
                                                                                                  meta_batch_size=meta_batch_size)
    meta_test_train_x_val, meta_test_train_y_val, meta_test_test_x_val, meta_test_test_y_val = generate_test_ILI_batch(
        train_x=test_train_x,
        train_y=test_train_y,
        test_x=test_test_x,
        test_y=test_test_y)


    num_testbatch = int(meta_test_train_x.shape[0])
    print(num_testbatch, 'support_size_te')
    test_loss, test_rmse, test_mae, test_acc, test_pcc, test_var, test_pred, test_mape = [], [], [], [], [], [], [], []
    if FLAGS.Do_meta:
        Dometa ='Dometa'
    else:
        Dometa = 'Nometa'
    file_name = FLAGS.logdir + '/' + target_data_name + '/report_%s_%0.2d_%0.2d_%0.2d_%0.2d_%0.2d.csv' % (Dometa, target_node, seq_len, pre_len, FLAGS.pretrain_iterations, FLAGS.metatrain_iterations)
    result_file = open(file_name, mode='w')
    print("result file name : %s" % file_name)
    result_file.write("EPOCH,RMSE,PCC,MAPE,MAE\n")
    if FLAGS.Do_meta:
        iter = 50
    else:
        iter = FLAGS.test_iterations

    for j in range(iter):
        predb_list = []
        for i in range(num_testbatch):
            sample_ind = np.random.randint(0, num_testbatch)
            inputa = meta_test_train_x[sample_ind, :, :, :]
            labela = meta_test_train_y[sample_ind, :, :, :]
            feed_dict1 = {model.inputa: inputa,  model.labela: labela}
            model_preda = sess.run(model.outputas, feed_dict1)
            # test_finetuning = sess.run([model.finetuning_op], feed_dict1)
        if j % 10 == 0:
            for v in range(int(meta_test_test_x_val.shape[0])):
                inputb = meta_test_test_x_val[v, :, :, :]  # b used for testing
                labelb = meta_test_test_y_val[v, :, :, :]
                feed_dict = {model.inputa: inputb, model.labela: labelb}
                model_predb = sess.run(model.outputas, feed_dict)
                predb_list.append(model_predb)
            predb_result1 = np.reshape(predb_list, (-1, 1))
            test_y2 = np.reshape(test_test_y, (-1, 1))
            test_labels = scaler.inverse_transform(test_y2)
            predict = scaler.inverse_transform(predb_result1)
            print (predict.shape,test_labels.shape, 'predict.shape,test_labels.shape')
            rmse, mae, _, pcc, mape, residual = evaluation(test_labels,predict)
            test_rmse.append(rmse)  # rmse
            test_mae.append(mae)
            test_pcc.append(pcc)
            test_mape.append(mape)
            test_pred.append(predict)
            test_pred.append(predict)
            print('Test Iter:{}'.format(j), 'test_rmse: %r' % rmse,
                  'test_pcc:{:.4}'.format(pcc),  # rmse
                  'test_mape:{:.4}'.format(mape),
                  'test_mae:{:.4}'.format(mae))
            result_file.write("%d,%f,%f,%f,%f\n" % (j, rmse, pcc, mape, mae))
        test_finetuning = sess.run([model.finetuning_op], feed_dict1)

    index = test_rmse.index(np.min(test_rmse))
    index1 = test_pcc.index(np.min(test_pcc))
    print('training_epoch:%r' % FLAGS.test_iterations,
              'seq_len:%r' % seq_len,
              'pre_len:%r' % pre_len,
              'min_rmse:%r' % np.min(test_rmse),
              'min_mae:%r' % test_mae[index],
              'max_pcc:%r' % test_pcc[index1],
              'best epoch:%r' % index
              )

    out_filename = FLAGS.logdir + '/' + target_data_name+'/' + str(target_node) + '/' + exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.csv'
    out_pkl = FLAGS.logdir + '/' + target_data_name+'/' + str(target_node) + '/' + exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.pkl'
    print('out_filename:', out_filename)
    print('out_pkl:', out_pkl)


def get_grad(model, sess):
    np.random.seed(1)
    random.seed(1)

    grad_list = list()
    for i in range(target_num_nodes):
        test_train_x, test_train_y, test_test_x, test_test_y = preprocess_data_forecasting(data_list[i], time_len,
                                                                                            train_rate, validation_rate,
                                                                                            seq_len, pre_len,
                                                                                            meta_batch_size)

        meta_test_train_x, meta_test_train_y, meta_test_test_x, meta_test_test_y = generate_ILI_batch(train_x=test_train_x,
                                                                                                      train_y=test_train_y,
                                                                                                      test_x=test_test_x,
                                                                                                      test_y=test_test_y,
                                                                                                      meta_batch_size=1)

        num_testbatch = int(meta_test_train_x.shape[0])
        print(num_testbatch, 'support_size_te')

        if FLAGS.Do_meta:
            Dometa = 'Dometa'
        else:
            Dometa = 'Nometa'
        file_name = FLAGS.logdir + '/' + target_data_name + '/report_%s_%0.2d_%0.2d_%0.2d_%0.2d_%0.2d.csv' % (Dometa, target_node,seq_len, pre_len, FLAGS.pretrain_iterations,FLAGS.metatrain_iterations)
        result_file = open(file_name, mode='w')
        print("result file name : %s" % file_name)
        result_file.write("EPOCH,RMSE,PCC,MAPE,MAE\n")

        temp_grad_list = list()
        for j in range(num_testbatch):
            inputa = meta_test_train_x[j, :, :, :]
            labela = meta_test_train_y[j, :, :, :]
            inputb = meta_test_train_x[0, :, :, :]
            labelb = meta_test_train_y[0, :, :, :]
            feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb,
                         model.meta_lr: 0.0}

            batch_gr = sess.run(model.gvs, feed_dict)
            temp_grad_list.append(batch_gr)
        print("-----------------------------temp_grad_list------------------------")
        temp_grad = np.sum(temp_grad_list, axis=0)
        temp_grad = temp_grad[:,0:1]
        length = np.shape(temp_grad)[0]
        # temp_grad = temp_grad[length-2:length-1]
        temp_grad = np.reshape(temp_grad, newshape=[-1])
        flatten = list()
        for k in range(np.shape(temp_grad)[0]):
            for_flatten = temp_grad[k]
            for_flatten = np.reshape(for_flatten, newshape=[-1])
            flatten.append(for_flatten)
        flatten = np.concatenate(flatten, axis=0)
        print(flatten)
        grad_list.append(flatten)
    grad_data = np.array(grad_list)

    standard_scaler = StandardScaler()
    grad_data = standard_scaler.fit_transform(grad_data)

    tsne_model = TSNE(n_components=2)
    res = tsne_model.fit_transform(grad_data)

    fig = plt.figure(figsize=(13, 13))
    plt.axis('off')
    sns.set_style('darkgrid')
    sns.scatterplot(res[:, 0], res[:, 1], hue=range(target_num_nodes), legend='full', palette=sns.color_palette("bright", target_num_nodes))

    legends = ["region %d" % j for j in range(target_num_nodes)]
    plt.legend(legends)
    plt.savefig("test_fig.png")

    file_name1 = "./report.csv"
    result_file1 = open(file_name1, mode='w')
    print("result file name : %s" % file_name1)

    for n1 in range(target_num_nodes):
        print("-------------------- region %d ---------------------" % n1)
        for n2 in range(target_num_nodes):
            cosine = cos_sim(grad_data[n1], grad_data[n2])
            print(cosine)
            result_file1.write("{:.10f},".format(cosine))
        result_file1.write("\n")

def main():
    if FLAGS.train:
        test_num_updates = 5
    else:
        test_num_updates = 1  # 원래는 10이었지만.. 굳이...?

    input_tensors = None

    model = MAML(seq_len, 1, test_num_updates=test_num_updates)
    if FLAGS.train:
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    else:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        FLAGS.meta_batch_size = 1
        model.construct_model(input_tensors=input_tensors, prefix='metaval_')

    model.summ_op = tf.summary.merge_all()

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)

    sess = tf.InteractiveSession()

    if FLAGS.train == False:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    exp_string = '.mbs_'+str(FLAGS.meta_batch_size) + '.ubs_' + str(FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(FLAGS.train_update_lr)

    if FLAGS.num_filters != 64:
        exp_string += 'hidden' + str(FLAGS.num_filters)
    if FLAGS.max_pool:
        exp_string += 'maxpool'
    if FLAGS.stop_grad:
        exp_string += 'stopgrad'
    if FLAGS.norm == 'batch_norm':
        exp_string += 'batchnorm'
    elif FLAGS.norm == 'layer_norm':
        exp_string += 'layernorm'
    elif FLAGS.norm == 'None':
        exp_string += 'nonorm'
    else:
        print('Norm setting not recognized.')

    resume_itr = 0

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    if FLAGS.resume or not FLAGS.train: # resume 꺼놓으면 무조건 처음부터 다시 train
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + target_data_name + '/' + str(target_node) + '/' + exp_string)
        if FLAGS.Do_meta:
            if FLAGS.test_iter > 0:
                model_file = model_file[:model_file.index('model')] + 'model'
                print("111111Restoring model weights from " + model_file)
            if model_file:
                ind1 = model_file.index('model')
                # resume_itr = int(model_file[ind1+5:])
                print("Restoring model weights from " + model_file)
                saver.restore(sess, model_file)

    # get_grad(model, sess)

    if FLAGS.train:
        train(model, saver, sess, exp_string, resume_itr)
    else:
        test(model, sess, exp_string)


if __name__ == "__main__":
    main()

