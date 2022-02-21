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
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_bool('Do_meta', False, 'True to meta learning, False to original deep learning.')

flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', True, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)')

data_name = "Region"

train_rate =  FLAGS.train_rate
seq_len = FLAGS.seq_len
validation_rate =  FLAGS.validation_rate
pre_len = FLAGS.pre_len
meta_batch_size = FLAGS.meta_batch_size


if data_name == "Region":
    data = pd.read_csv('./data/ILI_region_feature_2002-2020.csv')
elif data_name == "States":
    data = pd.read_csv('./data/ILI_states_feature_2010-2020.csv')
elif data_name == "Japan":
    data = pd.read_csv('./data/ILI_japan_feature_2012-2020.csv')

test_loss, test_rmse, test_mae, test_acc, test_pcc, test_var, test_pred, test_mape = [], [], [], [], [], [], [], []
for i in range(np.shape(data)[1]):
    if FLAGS.Do_meta:
        Dometa ='Dometa'
    else:
        Dometa = 'Nometa'
    target_node = i
    file_name = FLAGS.logdir + '/' + data_name + '/report_%s_%0.2d_%0.2d_%0.2d_%0.2d_%0.2d.csv' % (
    Dometa, target_node, seq_len, pre_len, FLAGS.pretrain_iterations, FLAGS.metatrain_iterations)
    result = np.array(pd.read_csv(file_name))
    # index = test_rmse.index(np.min(result[:,1]))
    # index1 = test_pcc.index(np.min(result[:,2]))
    print('min_rmse:%r' % (np.min(result[:,1])),
          'max_pcc:%r' % (np.min(result[:,2])))
    test_rmse.append(np.min(result[:,1]))
    test_pcc.append(np.min(result[:,2]))

print('Mean result for dataset:'+data_name+'----rmse:%r' % (np.mean(test_rmse)),
          'pcc:%r' % (np.mean(test_pcc)))


