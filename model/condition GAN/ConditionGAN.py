import tensorflow as tf
import numpy as np
import time
import scipy.io as sio
from function.analysis import (loadData, pickTransferInput, sythesis, nextbatch)
# from function.layer import gateCNN

TrainDataPath = '../../vcc2016/mat/Train/'
TestDataPath = '../../vcc2016/mat/Test/'


tS = time.time()
trainData = loadData(TrainDataPath)
# testData = loadData(TestDataPath)
tE = time.time()
print("loading data time: %f" % (tE-tS))

dataSize = 513
latentSize = 64
speakerN = 10
N = 500
L = 80
tstep = 100
hidNum = 1000

def gateCNN(input, shape, name):
    with tf.variable_scope(name):
        U = tf.get_variable('U', shape=shape, initializer=tf.random_normal_initializer(stddev=0.1))
        b_U = tf.get_variable('b_U', shape=shape[1], initializer=tf.constant_initializer(0.01))
        V = tf.get_variable('V', shape=shape, initializer=tf.random_normal_initializer(stddev=0.1))
        b_V = tf.get_variable('b_V', shape=shape[1], initializer=tf.constant_initializer(0.01))
        x = tf.nn.relu(tf.matmul(input, U) + b_U)
        w = tf.sigmoid(tf.matmul(input, V) + b_V)
        output = tf.multiply(x, w)
        return output


source = tf.placeholder(tf.float32, shape = [None, tstep*L])
target = tf.placeholder(tf.float32, shape = [None, tstep*L])
gCNN_in = tf.reshape(source, [-1, L])
gCNN_out = gateCNN(gCNN_in, [L, N], 'gateCNN')
lstm_in = tf.reshape(gCNN_out, [-1, tstep, N])

lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(hidNum, forget_bias=1.0)
lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(hidNum, forget_bias=1.0)
lstm_cell_3 = tf.contrib.rnn.BasicLSTMCell(hidNum, forget_bias=1.0)
lstm_cell_4 = tf.contrib.rnn.BasicLSTMCell(hidNum, forget_bias=1.0)

with tf.variable_scope('lstm_cell_1'):
    x, _ = tf.nn.dynamic_rnn(lstm_cell_1, lstm_in, dtype=tf.float32)

with tf.variable_scope('lstm_cell_2'):
    x, _ = tf.nn.dynamic_rnn(lstm_cell_2, x, dtype=tf.float32)

with tf.variable_scope('lstm_cell_3'):
    x, _ = tf.nn.dynamic_rnn(lstm_cell_3, x, dtype=tf.float32)

with tf.variable_scope('lstm_cell_4'):
    lstm_out, _ = tf.nn.dynamic_rnn(lstm_cell_4, x, dtype=tf.float32)

lstm_out = tf.reshape(lstm_out, [-1, hidNum])
W_fully = tf.get_variable('W_fully', shape=[1000, 500], initializer=tf.random_normal_initializer(stddev=0.1))
b_fully = tf.get_variable('b_fully', shape=[500], initializer=tf.constant_initializer(0.01))
W = tf.nn.softmax(tf.matmul(lstm_out, W_fully) + b_fully)
B = tf.get_variable('Basis', shape=[N, L], initializer=tf.random_normal_initializer(stddev=0.001))
recover = tf.matmul(W, B)
recover = tf.reshape(recover, [-1, tstep*L])

loss=tf.reduce_mean(tf.abs(recover-target))
train_step = tf.train.AdamOptimizer(0.00001).minimize(loss)

batchSize = 128
src = 'SF1'
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(20):
        for j in range(20):
                x_batch, y_batch = nextbatch(trainData, src, batchSize)
                train_step.run(feed_dict = {source: x_batch, target: y_batch})
        x_Evabatch, y_Evabatch = nextbatch(trainData, src, batchSize)
        eva_logp= sess.run(loss, feed_dict={source: x_Evabatch, target: y_Evabatch})
        print('epoch %d' %(i))
        print('train log-probability: %f' %(np.mean(eva_logp)))

voice, filename = pickTransferInput(TestDataPath, 'SF1')
output = sess.run(recover, feed_dict={source: voice})
opname = filename + '.wav'
sio.savemat(opname, mdict={'x': output})
