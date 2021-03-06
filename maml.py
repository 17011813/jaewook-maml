""" Code for the MAML algorithm and network definitions. """
from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf
try:
    import special_grads
except KeyError as e:
    print('WARN: Cannot define MaxPoolGrad, likely already defined for this version of tensorflow: %s' % e,
          file=sys.stderr)

from tensorflow.python.platform import flags
from utils import mse, xent, conv_block, normalize

FLAGS = flags.FLAGS

class MAML:
    def __init__(self, dim_input=1, dim_output=1, test_num_updates=5, num_nodes=0):
        """ must call construct_model() after initializing MAML! """
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = FLAGS.update_lr
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        self.classification = False
        self.test_num_updates = test_num_updates
        self.num_nodes = num_nodes
        if FLAGS.datasource == 'sinusoid':
            self.dim_hidden = [40, 40]
            self.loss_func = mse
            self.forward = self.forward_fc
            self.construct_weights = self.construct_fc_weights
        elif FLAGS.datasource == 'weather':
            self.dim_hidden = [40, 40]
            self.loss_func = mse
            self.forward = self.forward_fc
            self.construct_weights = self.construct_fc_weights
        elif FLAGS.datasource == 'ILI':
            self.dim_hidden = [10,5,1]
            self.loss_func = mse
            self.forward = self.forward_fc
            self.construct_weights = self.construct_fc_weights

        else:
            raise ValueError('Unrecognized data source.')

    def construct_model(self, input_tensors=None, prefix='metatrain_'):
        # a: training data for inner gradient, b: test data for meta gradient
        if input_tensors is None:
            self.inputa = tf.placeholder(tf.float32)
            self.inputb = tf.placeholder(tf.float32)
            self.labela = tf.placeholder(tf.float32, shape=[None, 1, 1])
            self.labelb = tf.placeholder(tf.float32, shape=[None, 1, 1])
        else:
            self.inputa = input_tensors['inputa']
            self.inputb = input_tensors['inputb']
            self.labela = input_tensors['labela']
            self.labelb = input_tensors['labelb']

        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
                print("Weights~~~~~~~~~~~~~")
            else:
                # Define the weights
                self.weights = weights = self.construct_weights()
                print("Weights1111SSSSSSSSSS~~~~~~~~~~~~~")
            # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates

            lossesa, outputas, lossesb, outputbs = [], [], [], []
            accuraciesa, accuraciesb = [], []
            num_updates = max(self.test_num_updates, FLAGS.num_updates)  # ?????? 5??? ??????..
            outputbs = [[]]*num_updates
            lossesb = [[]]*num_updates
            accuraciesb = [[]]*num_updates

            def task_metalearn(inp, reuse=True):
                """ Perform gradient descent for one task in the meta-batch. """
                inputa, inputb, labela, labelb = inp
                task_outputbs, task_lossesb = [], []

                # inputa = tf.transpose(inputa, [0,2,1])
                # print(inputa.shape,'inputa.shape')
                # print(weights,'weights.shape')

                task_outputa = self.forward(inputa, weights, reuse=reuse)  # only reuse on the first iter
                task_lossa = self.loss_func(task_outputa, labela)

                grads = tf.gradients(task_lossa, list(weights.values()))
                # if FLAGS.stop_grad: # stop_grad??? False?????? ?????? ????????????
                #     grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(weights.keys(), grads))
                #fast weight
                fast_weights = dict(zip(weights.keys(), [weights[key] - self.update_lr*gradients[key] for key in weights.keys()]))
                output = self.forward(inputb, fast_weights, reuse=True)
                task_outputbs.append(output)
                task_lossesb.append(self.loss_func(output, labelb))

                for j in range(num_updates - 1):  # 5???
                    loss = self.loss_func(self.forward(inputa, fast_weights, reuse=True), labela)
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    # if FLAGS.stop_grad:
                    #     grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(fast_weights.keys(), grads))
                    fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.update_lr*gradients[key] for key in fast_weights.keys()]))
                    output = self.forward(inputb, fast_weights, reuse=True)
                    task_outputbs.append(output)
                    task_lossesb.append(self.loss_func(output, labelb))

                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]
                return task_output

            def task_metatest(inp, reuse=True):
                """ Perform gradient descent for one task in the meta-batch. """
                inputa, inputb, labela, labelb = inp
                task_outputbs, task_lossesb = [], []

                # inputa = tf.transpose(inputa, [0,2,1])
                # print(inputa.shape,'inputa.shape')
                # print(weights,'weights.shape')
                task_outputa = self.forward(inputa, weights, reuse=False)  # only reuse on the first iter
                task_lossa = self.loss_func(task_outputa, labela)

                output = self.forward(inputb, weights, reuse=False)
                task_outputbs.append(output)
                task_lossesb.append(self.loss_func(output, labelb))

                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]
                return task_output

            if FLAGS.norm is not 'None':
                # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
                unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

            out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates]
            # result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            # outputas, outputbs, lossesa, lossesb  = result

            # result_test = tf.map_fn(task_metatest, elems=(self.inputa, self.inputb, self.labela, self.labelb),
            #                    dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            # outputas, outputbs, lossesa, lossesb = result_test
            # print(outputas, outputbs, 'outputas.shape, outputbs.shape')
            # print(prefix,'prefix')
        ## Performance & Optimization
        if 'metatrain_' in prefix:
            print("Meta train!!!!!!!!!!")
            result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb),
                               dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            outputas, outputbs, lossesa, lossesb = result
            self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            # after the map_fn
            self.outputas, self.outputbs = outputas, outputbs
            self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)

            if FLAGS.metatrain_iterations > 0:
                print('Meta train - optimizing')
                optimizer = tf.train.AdamOptimizer(self.meta_lr)
                self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[FLAGS.num_updates-1])
                self.metatrain_op = optimizer.apply_gradients(gvs)

        elif 'metaval_' in prefix:
            print('Meta TEST!!!!!!!!!!!!! in maml')
            result_test = tf.map_fn(task_metatest, elems=(self.inputa, self.inputb, self.labela, self.labelb),
                                    dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            outputas, outputbs, lossesa, lossesb = result_test

            self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            self.outputas, self.outputbs = outputas, outputbs
            self.finetuning_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)
            # self.finetuning_op = tf.train.AdamOptimizer(0).minimize(total_loss1)
            # self.gvs = gvs = optimizer.compute_gradients(total_loss1)
            # self.finetuning_op = optimizer.apply_gradients(gvs)
        ## Summaries
        tf.summary.scalar(prefix+'Pre-update loss', total_loss1)

        for j in range(num_updates):
            tf.summary.scalar(prefix+'Post-update loss, step ' + str(j+1), total_losses2[j])

    ### Network construction functions (fc networks and conv networks)
    def construct_fc_weights(self):
        weights = {}
        weights['w1'] = tf.Variable(tf.truncated_normal([self.dim_input, self.dim_hidden[0]], stddev=0.01))
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
        for i in range(1,len(self.dim_hidden)):
            weights['w'+str(i+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[i-1], self.dim_hidden[i]], stddev=0.01))
            weights['b'+str(i+1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]))
        weights['w'+str(len(self.dim_hidden)+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[-1], self.dim_output], stddev=0.01))
        weights['b'+str(len(self.dim_hidden)+1)] = tf.Variable(tf.zeros([self.dim_output]))
        return weights

    def forward_fc(self, inp, weights, reuse=False):
        hidden = normalize(tf.matmul(inp, weights['w1']) + weights['b1'], activation=tf.nn.relu, reuse=reuse, scope='0')
        for i in range(1,len(self.dim_hidden)):
            hidden = normalize(tf.matmul(hidden, weights['w'+str(i+1)]) + weights['b'+str(i+1)], activation=tf.nn.relu, reuse=reuse, scope=str(i+1))
        return tf.matmul(hidden, weights['w'+str(len(self.dim_hidden)+1)]) + weights['b'+str(len(self.dim_hidden)+1)]

    # def construct_conv_weights(self):
    #     weights = {}
    #
    #     dtype = tf.float32
    #     conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
    #     fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)
    #     k = 3
    #
    #     weights['conv1'] = tf.get_variable('conv1', [k, k, self.channels, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
    #     weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]))
    #     weights['conv2'] = tf.get_variable('conv2', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
    #     weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]))
    #     weights['conv3'] = tf.get_variable('conv3', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
    #     weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]))
    #     weights['conv4'] = tf.get_variable('conv4', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
    #     weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]))
    #     if FLAGS.datasource == 'miniimagenet':
    #         # assumes max pooling
    #         weights['w5'] = tf.get_variable('w5', [self.dim_hidden*5*5, self.dim_output], initializer=fc_initializer)
    #         weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
    #     else:
    #         weights['w5'] = tf.Variable(tf.random_normal([self.dim_hidden, self.dim_output]), name='w5')
    #         weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
    #     return weights
    #
    # def forward_conv(self, inp, weights, reuse=False, scope=''):
    #     # reuse is for the normalization parameters.
    #     channels = self.channels
    #     inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])
    #
    #     hidden1 = conv_block(inp, weights['conv1'], weights['b1'], reuse, scope+'0')
    #     hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], reuse, scope+'1')
    #     hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], reuse, scope+'2')
    #     hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], reuse, scope+'3')
    #     if FLAGS.datasource == 'miniimagenet':
    #         # last hidden layer is 6x6x64-ish, reshape to a vector
    #         hidden4 = tf.reshape(hidden4, [-1, np.prod([int(dim) for dim in hidden4.get_shape()[1:]])])
    #     else:
    #         hidden4 = tf.reduce_mean(hidden4, [1, 2])
    #
    #     return tf.matmul(hidden4, weights['w5']) + weights['b5']


