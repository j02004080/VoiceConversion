import tensorflow as tf
import numpy as np
import time
import scipy.io as sio
import matplotlib.pyplot as plt
from function.analysis import (loadData, pickTransferInput, sythesis, nextbatch)
import function.layer as fl

TrainDataPath = '../../vcc2016/mat/Train/'
TestDataPath = '../../vcc2016/mat/Test/'

dataSize = 513
latentSize = 64
speakerN = 10
N = 500
L = 80
tstep = 100
hidNum = 1000

tS = time.time()
trainData = loadData(TrainDataPath, L, tstep)
# testData = loadData(TestDataPath)
tE = time.time()
print("loading data time: %f" % (tE-tS))

arch = {'channel' : N, 'kernel': [1, L], 'stride': [1,1]}

source = tf.placeholder(tf.float32, shape = [None, tstep*L])
target = tf.placeholder(tf.float32, shape = [None, tstep*L])
x = tf.reshape(source, [-1, tstep, L, 1])
GCNN_en1 = fl.gatedCNN(x, arch, 'GCNN_en1')
W = tf.reshape(GCNN_en1, [-1, N])

flat = tf.contrib.layers.layer_norm(tf.reshape(GCNN_en1, [-1, tstep, arch['channel']]))
lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(N, forget_bias=1.0)
with tf.variable_scope('lstm_cell_1'):
    lstm_out, _ = tf.nn.dynamic_rnn(lstm_cell_1, flat, dtype=tf.float32)

# x = tf.contrib.layers.layer_norm(x)
# lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(hidNum, forget_bias=1.0)
# with tf.variable_scope('lstm_cell_2'):
#     x, _ = tf.nn.dynamic_rnn(lstm_cell_2, x, dtype=tf.float32)
#
# x = tf.contrib.layers.layer_norm(x)
# lstm_cell_3 = tf.contrib.rnn.BasicLSTMCell(hidNum, forget_bias=1.0)
# with tf.variable_scope('lstm_cell_3'):
#     lstm_out, _ = tf.nn.dynamic_rnn(lstm_cell_3, x, dtype=tf.float32)

B = tf.get_variable('Basis', shape=[N, L], initializer=tf.random_normal_initializer(stddev=0.01))
M = tf.reshape(lstm_out, [-1, N])
recover = tf.matmul(tf.multiply(W,M), B)
recover = tf.reshape(recover, [-1, tstep*L])




loss=tf.reduce_mean(tf.abs(recover-target))
# loss=tf.nn.l2_loss(target-recover)
train_step = tf.train.AdamOptimizer(0.0003).minimize(loss)

batchSize = 128
src = 'SF1'
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tS = time.time()
e=[]
for i in range(10):
        for j in range(100):
                x_batch, y_batch = nextbatch(trainData, src, batchSize)
                train_step.run(feed_dict = {source: y_batch, target: y_batch})
        x_Evabatch, y_Evabatch = nextbatch(trainData, src, batchSize)
        eva_loss= sess.run(loss, feed_dict={source: y_Evabatch, target: y_Evabatch})
        e.append(eva_loss)
        print('epoch %d' %(i))
        print('loss: %f' %(np.mean(eva_loss)))

voice, filename = pickTransferInput(TestDataPath, 'TM1', L, tstep)
output, base = sess.run([recover, B], feed_dict={source: voice})
output = output.reshape([1, -1])
opname = '../../output/TimeDomain' + filename
sio.savemat(opname, mdict={'x': output, 'base': base})
plt.plot(e)

tE = time.time()
print("Training time: %f sec" % (tE-tS))