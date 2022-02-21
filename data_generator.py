""" Code for loading data. """
import numpy as np
import os
import random
import tensorflow as tf
import pandas as pd
from tensorflow.python.platform import flags
from utils import get_images
import matplotlib.pylab as plt
import csv
import numpy as np
import pickle
import random
import tensorflow as tf
import pandas as pd
from maml import MAML
from tensorflow.python.platform import flags
import matplotlib.pylab as plt
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy.linalg as la

FLAGS = flags.FLAGS

class DataGenerator(object):
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """
    def __init__(self, num_samples_per_class, batch_size, rate, validation_rate, pre_len, seq_len, config={}):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = 1  # by default 1 (only relevant for classification problems)

        if  FLAGS.datasource == 'weather':
            self.generate = self.generate_weather_temperature_batch
            # self.generate = self.generate_sinusoid_batch
            self.load_data = self.load_ILI_Region_data
            self.generate_test = self.generate_test_weather_temperature_batch
            self.amp_range = config.get('amp_range', [0.0, 35.0])
            self.phase_range = config.get('phase_range', [1, 12])
            self.dim_input = 1
            self.dim_output = 1
            self.input_range = config.get('input_range', [-5.0, 0.0])
        elif FLAGS.datasource == 'ILI':
            self.generate = self.generate_ILI_batch
            self.generate_test = self.generate_test_ILI_batch
            self.dim_input = 20
            self.dim_output = 1
            self.rate = rate
            self.validation_rate = validation_rate
            self.pre_len = pre_len
            self.seq_len = seq_len
        else:
            raise ValueError('Unrecognized data source')

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

    def generate_weather_temperature_batch(self, train=True, input_idx=None):
        # set up input conditions
        csv_name = "data/ILI_region_feature_2002-2020.csv"
        # csv_name = "data/tokyo_temp_196801-201712.csv"

        # load csv file
        temp_data = np.genfromtxt(csv_name, delimiter=',')  # (600, 2)
        max_data = np.max(temp_data)
        min_data = np.min(temp_data)
        # print(temp_data.shape,max_data,min_data,"temp_data.shape")

        # randomly select data from csv
        # (500, 2)
        selected_data = np.array(random.sample(temp_data.tolist(), self.batch_size*self.num_samples_per_class*self.dim_input)) # batch_size = 25, num_samples_per_class = 10 인데 이 변수가 K-Shot 할때 K래;;그럼 시계열 개수인가.. 나라 수인가..
        # (500, ) (500, )
        label_batch, data_batch = np.transpose(selected_data)
        # data_batch = np.transpose(selected_data)
        # print(selected_data.shape, data_batch.shape  "label_batch, data_batch") # (500, 2)  (500, )

        # add noise for 10%
        noise = np.random.uniform(-0.1, 0.1, self.batch_size*self.num_samples_per_class*self.dim_input)
        noisy_data_batch = data_batch + data_batch * noise

        # mock conditions
        # amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.batch_size])
        amp = np.random.uniform(min_data, max_data, [self.batch_size])

        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size])

        label_batch1 = np.reshape(label_batch, [self.batch_size, self.num_samples_per_class, self.dim_input])
        data_batch1 = np.reshape(data_batch, [self.batch_size, self.num_samples_per_class, self.dim_input])
        noisy_data_batch1 = np.reshape(noisy_data_batch, [self.batch_size, self.num_samples_per_class, self.dim_input])
        # print(label_batch1.shape,   "label_batch1")  # (25, 20, 1)


        return label_batch1, data_batch1, amp, phase
        # 이거 noise가 들어간 data가 출력이 되는게 맞는건가..
        # return label_batch1, noisy_data_batch1, amp, phase



    def generate_test_weather_temperature_batch(self, train=True, input_idx=None):
        # set up input conditions
        csv_name = "data/test.csv"
        num_samples_per_class = 1
        batch_size = 12

        # load csv file
        temp_data = np.genfromtxt(csv_name, delimiter=',')

        # randomly select data from csv
        selected_data = np.array(random.sample(temp_data.tolist(), batch_size*num_samples_per_class*self.dim_input))
        label_batch, data_batch = np.transpose(selected_data)

        # add noise for 10%
        noise = np.random.uniform(-0.1, 0.1, batch_size*num_samples_per_class*self.dim_input)
        noisy_data_batch = data_batch + data_batch * noise

        # mock conditions
        amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.batch_size])
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size])

        label_batch1 = np.reshape(label_batch, [batch_size, num_samples_per_class, self.dim_input])
        noisy_data_batch1 = np.reshape(noisy_data_batch, [batch_size, num_samples_per_class, self.dim_input])

        return label_batch1, noisy_data_batch1, amp, phase

    def preprocess_data_forecasting(data, time_len, rate, validation_rate, seq_len, pre_len):

        total_x = list()
        total_y = list()

        for i in range(time_len - seq_len - pre_len):
            a = data[i: i + seq_len + pre_len]
            total_x.append(np.transpose(a[0: seq_len]))
            total_y.append(a[seq_len + pre_len - 1])

        data_len = len(total_x)

        train_size = int(data_len * rate)
        validation_size = int(data_len * validation_rate)

        train_x = np.array(total_x[0:train_size])
        train_y = np.array(total_y[0:train_size])
        valid_x = np.array(total_x[train_size:train_size + validation_size])
        valid_y = np.array(total_y[train_size:train_size + validation_size])
        test_x = np.array(total_x[train_size + validation_size:])
        test_y = np.array(total_y[train_size + validation_size:])

        print('train_x:', np.shape(train_x))
        print('train_y:', np.shape(train_y))
        print('valid_x:', np.shape(valid_x))
        print('valid_y:', np.shape(valid_y))
        print('test_x:', np.shape(test_x))
        print('test_y:', np.shape(test_y))

        return train_x, train_y, valid_x, valid_y, test_x, test_y


    #             self.amp_range = config.get('amp_range', [0.1, 5.0])
    #             self.phase_range = config.get('phase_range', [0, np.pi])
    #             self.input_range = config.get('input_range', [-5.0, 5.0])
    def generate_sinusoid_batch(self, train=True, input_idx=None):
        # Note train arg is not used (but it is used for omniglot method.
        # input_idx is used during qualitative testing --the number of examples used for the grad update
        amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.batch_size])
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size])
        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
        for func in range(self.batch_size):
            init_inputs[func] = np.random.uniform(self.input_range[0], self.input_range[1],
                                                  [self.num_samples_per_class, 1]) # -5에서

            outputs[func] = amp[func] * np.sin(init_inputs[func] - phase[func])
        # print(init_inputs.shape,'s')
        return init_inputs, outputs, amp, phase