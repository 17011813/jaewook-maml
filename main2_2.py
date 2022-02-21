import csv
import numpy as np
import pickle
import random
import tensorflow as tf
import pandas as pd
from data_generator import DataGenerator
from maml2 import *
from input_data import *
from tensorflow.python.platform import flags
import matplotlib.pylab as plt
from sklearn.preprocessing import MinMaxScaler
# from data_generator import *
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
flags.DEFINE_bool('Do_meta', True, 'True to meta learning, False to original deep learning.')

flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', True, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot

data_name = "Region"

train_rate =  FLAGS.train_rate
seq_len = FLAGS.seq_len
validation_rate =  FLAGS.validation_rate
pre_len = FLAGS.pre_len
meta_batch_size = FLAGS.meta_batch_size

target_node =  1
if data_name == "Region":
    data = pd.read_csv('./data/ILI_region_feature_2002-2020.csv')
elif data_name == "States":
    data = pd.read_csv('./data/ILI_states_feature_2010-2020.csv')
elif data_name == "Japan":
    data = pd.read_csv('./data/ILI_japan_feature_2012-2020.csv')

Mtrain_data = data.drop(columns=[str(int(target_node))])
print(Mtrain_data.shape, "Meta train data")
Mtest_data = data.iloc[:, [target_node]]
print(Mtest_data.shape, "Meta test data, Target Node:", int(target_node))
Mtrain_data1 = np.mat(Mtrain_data, dtype=np.float32)
Mtest_data1 = np.mat(Mtest_data, dtype=np.float32)
scaler = MinMaxScaler()
# scaler.fit(Mtrain_data1[0:int(len(Mtrain_data1))])
scaler.fit(Mtrain_data1[0:int(len(Mtrain_data1) * FLAGS.train_rate)])
Mtrain_data1 = scaler.transform(Mtrain_data1)
# print(Mtrain_data1,'SSSSSSSSSSSSS')
scaler1 = MinMaxScaler()
scaler1.fit(Mtest_data1[0:int(len(Mtest_data1) * FLAGS.train_rate)])
Mtest_data1 = scaler1.transform(Mtest_data1)

def train(model, saver, sess, exp_string, resume_itr=0):
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 1000
    if FLAGS.datasource == 'ILI':
        PRINT_INTERVAL = 100#1000
        PRINT_FIG_INTERVAL = 100  # 1000
        TEST_PRINT_INTERVAL = PRINT_INTERVAL
    else:
        PRINT_INTERVAL = 100
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/'+data_name+'/'+ str(target_node)+ '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []

    time_len = Mtrain_data.shape[0]
    num_nodes = Mtrain_data.shape[1]
    num_classes = num_nodes
    multitask_weights, reg_weights = [], []

    fout = open(str(FLAGS.logdir) + "_" + str(FLAGS.datasource) + "_train.txt", 'w')
    print(fout)
    fout2 = open(str(FLAGS.logdir) + "_" + str(FLAGS.datasource) + "_validate.txt", 'w')
    # print(Mtrain_data1)
    train_x, train_y, test_x, test_y =  preprocess_data_forecasting1(Mtrain_data1, time_len, train_rate, validation_rate, seq_len, pre_len,meta_batch_size)
    train_x1, train_y1, test_x1, test_y1 =  preprocess_data_forecasting1(Mtest_data1, time_len, train_rate, validation_rate, seq_len, pre_len,meta_batch_size)

    metatrain_trainx, metatrain_trainy, metatrain_testx, metatrain_testy = generate_ILI_batch(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y,meta_batch_size=meta_batch_size)
    metatest_trainx, metatest_trainy, metatest_testx, metatest_testy = generate_ILI_batch(train_x=train_x1, train_y=train_y1, test_x=test_x1, test_y=test_y1,meta_batch_size=meta_batch_size)
    # print(metatrain_trainx)
    metatest_trainx1, metatest_trainy1, metatest_testx1, metatest_testy1 = generate_test_ILI_batch(train_x=train_x1,
                                                                                                   train_y=train_y1,
                                                                                                   test_x=test_x1,
                                                                                                   test_y=test_y1)
    print(metatrain_trainx.shape, metatrain_trainy.shape,metatrain_testx.shape, metatrain_testy.shape,'Meta train data shapes')
    print(metatest_trainx.shape, metatest_trainy.shape,metatest_testx.shape, metatest_testy.shape,'Meta test data shapes')  # (18, 25, 10, 20) (18, 25, 1, 10)

    support_size_tr = int(metatrain_trainx.shape[0]/2)
    support_size_te = int(metatest_trainx.shape[0]/2)
    # print(metatrain_trainx.shape, metatrain_trainy.shape)  # (18, 25, 10, 20) (18, 25, 1, 10)
    for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
        feed_dict = {}
        for i in range(support_size_tr):
            inputa = metatrain_trainx[i, :, :, :]
            labela = metatrain_trainy[i, :,:, :]
            inputb = metatrain_testx[i, :, :, :] # b used for testing
            labelb = metatrain_testy[i, :, :, :]
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
            saver.save(sess, FLAGS.logdir + '/'+data_name+'/'+ str(target_node)+  '/' + exp_string + '/model' + str(itr))

        # Meta test validation
        if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0:
            predb_list = []
            for j in range(int(metatest_trainx1.shape[0])):
                inputa = metatest_trainx1[j, :, :, :]
                # print(inputa.shape)
                labela = metatest_trainy1[j, :,:, :]
                inputb = metatest_testx1[j, :, :, :]  # b used for testing
                labelb = metatest_testy1[j, :, :, :]
                feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}

                input_tensors = [model.total_loss1, model.total_losses2[FLAGS.num_updates-1]]

                result = sess.run(input_tensors, feed_dict)
                model_predb = sess.run(model.outputbs, feed_dict)
                # print(np.shape(model_predb))
                predb_list.append(model_predb[-1])
                # print(predb_list)
            predb_result = np.reshape(predb_list, (-1, 1))
            test_y2 = np.reshape(test_y1, (-1, 1))
            # print (predb_result.shape,'model_preda')
            test_labels = scaler1.inverse_transform(test_y2)
            predict = scaler1.inverse_transform(predb_result)
            print(predict.shape, test_labels.shape, 'predict.shape,test_labels.shape')
            rmse, mae, _, pcc, mape, Residual = evaluation(test_labels, predict)
            print('Test during Training!!','Test Iter:{}'.format(itr), 'test_rmse: %r' % (rmse),
                  'test_pcc:{:.4}'.format(pcc),  # rmse
                  'test_mape:{:.4}'.format(mape),
                  'test_mae:{:.4}'.format(mae))
            print('Validation results: ' + str(result[0]) + ', ' + str(result[1]))
            fout2.write("\rvalidation_result\t{}\t{}".format(str(result[0]), str(result[1])))

    saver.save(sess, FLAGS.logdir + '/' + data_name + '/' + str(target_node) +'/' + exp_string + '/model' + str(itr))
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
                                                                        validation_rate, seq_len, pre_len,meta_batch_size)
    metatest_trainx1, metatest_trainy1, metatest_testx1, metatest_testy1 = generate_test_ILI_batch(train_x=train_x1,
                                                                                          train_y=train_y1,
                                                                                          test_x=test_x1,
                                                                                          test_y=test_y1)
    metatest_trainx, metatest_trainy, metatest_testx, metatest_testy = generate_ILI_batch(train_x=train_x1,
                                                                                               train_y=train_y1,
                                                                                               test_x=test_x1,
                                                                                               test_y=test_y1,meta_batch_size=meta_batch_size)
    num_testbatch = int(metatest_trainx.shape[0])
    print(num_testbatch,'support_size_te')
    test_loss, test_rmse, test_mae, test_acc, test_pcc, test_var, test_pred, test_mape = [], [], [], [], [], [], [], []
    if FLAGS.Do_meta:
        Dometa ='Dometa'
    else:
        Dometa = 'Nometa'
    file_name = FLAGS.logdir + '/' + data_name + '/report_%s_%0.2d_%0.2d_%0.2d_%0.2d_%0.2d.csv' % (Dometa, target_node,seq_len, pre_len, FLAGS.pretrain_iterations,FLAGS.metatrain_iterations)
    result_file = open(file_name, mode='w')
    print("result file name : %s" % file_name)
    result_file.write("EPOCH,RMSE,PCC,MAPE,MAE\n")
    if FLAGS.Do_meta:
        iter = 50
    else:
        iter = FLAGS.test_iterations

    for j in range(iter):
        predb_list = []
        # losses = []
        feed_dict = {}
        feed_dict1 = {}
        for i in range(num_testbatch):
            inputa = metatest_trainx[i, :, :, :]
            labela = metatest_trainy[i, :, :, :]
            feed_dict1 = {model.inputa: inputa,  model.labela: labela}
            model_preda = sess.run(model.outputas, feed_dict1)
            # test_finetuning = sess.run([model.finetuning_op], feed_dict1) # 원래는 이렇게 하는게 맞지만?? Train/Test가 너무 경향이 다를때는 마지막 배치만 가지고 학습하는게 좋음
        if j % 10 == 0:
            for v in range(int(metatest_trainx1.shape[0])):
                inputb = metatest_testx1[v, :, :, :]  # b used for testing
                labelb = metatest_testy1[v, :, :, :]
                feed_dict = {model.inputa: inputb, model.labela: labelb}
                model_predb = sess.run(model.outputas, feed_dict)
                predb_list.append(model_predb)
            predb_result1 = np.reshape(predb_list, (-1, 1))
            test_y2 = np.reshape(test_y1, (-1, 1))
            # print (predb_result.shape,'model_preda')
            test_labels = scaler1.inverse_transform(test_y2)
            predict = scaler1.inverse_transform(predb_result1)
            print (predict.shape,test_labels.shape, 'predict.shape,test_labels.shape')
            rmse, mae, _, pcc, mape, Residual = evaluation(test_labels,predict)
            test_rmse.append(rmse)  # rmse
            test_mae.append(mae)
            test_pcc.append(pcc)
            test_mape.append(mape)
            test_pred.append(predict)
            test_pred.append(predict)
            print('Test Iter:{}'.format(j), 'test_rmse: %r' % (rmse),
                  'test_pcc:{:.4}'.format(pcc),  # rmse
                  'test_mape:{:.4}'.format(mape),
                  'test_mae:{:.4}'.format(mae))
            result_file.write("%d,%f,%f,%f,%f\n" % (j, rmse, pcc, mape, mae))
        test_finetuning = sess.run([model.finetuning_op], feed_dict1)

    index = test_rmse.index(np.min(test_rmse))
    index1 = test_pcc.index(np.min(test_pcc))
    print('training_epoch:%r' % (FLAGS.test_iterations),
              'seq_len:%r' % (seq_len),
              'pre_len:%r' % (pre_len),
              'min_rmse:%r' % (np.min(test_rmse)),
              'min_mae:%r' % (test_mae[index]),
              'max_pcc:%r' % (test_pcc[index1]),
              'best epoch:%r' % (index)
              )

    out_filename = FLAGS.logdir + '/'+data_name+'/'+ str(target_node)+ '/' + exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.csv'
    out_pkl = FLAGS.logdir + '/'+data_name+'/'+ str(target_node)+ '/' + exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.pkl'
    print('out_filename:',out_filename)
    print('out_pkl:',out_pkl)
def main():

    if  FLAGS.datasource == 'ILI':
        if FLAGS.train:
            test_num_updates = 5
        else:
            test_num_updates = 1 # 원래는 10이었지만.. 굳이...?



    # for i in range(np.shape())
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
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/'+data_name+'/'+ str(target_node)+'/' + exp_string)
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

