import tensorflow as tf
import numpy as np
import time
import scipy.io as sio
from function.analysis import (loadData, pickTransferInput, sythesis, nextbatch)
from function.layer import gateCNN

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

source = tf.placeholder(tf.float32, shape = [None, tstep*L])
target = tf.placeholder(tf.float32, shape = [None, tstep*L])
gCNN_in = tf.reshape(source, [-1, L])
gcnn_en1 = gateCNN(gCNN_in, [L, N], 'gateCNN_en1')
B = tf.get_variable('Basis', shape=[N, L], initializer=tf.random_normal_initializer(stddev=0.01))
lstm_out = tf.matmul(gcnn_en1, B)
# gcnn_de1 = gateCNN(latent, [N, N], 'gateCNN_de1')
# gcnn_de2 = gateCNN(gcnn_de1, [N, L], 'gateCNN_de2')
# lstm_in = tf.reshape(gcnn_de2, [-1, tstep, L])
#
# lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(L, forget_bias=1.0)
# lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(L, forget_bias=1.0)
# with tf.variable_scope('lstm_cell_1'):
#     lstm_out, _ = tf.nn.dynamic_rnn(lstm_cell_1, lstm_in, dtype=tf.float32)

# with tf.variable_scope('lstm_cell_2'):
#     lstm_out, _ = tf.nn.dynamic_rnn(lstm_cell_2, x, dtype=tf.float32)

# with tf.variable_scope('lstm_cell_3'):
#     x, _ = tf.nn.dynamic_rnn(lstm_cell_3, x, dtype=tf.float32)
#
# with tf.variable_scope('lstm_cell_4'):
#     lstm_out, _ = tf.nn.dynamic_rnn(lstm_cell_4, x, dtype=tf.float32)

recover = tf.reshape(lstm_out, [-1, tstep*L])
loss=tf.reduce_mean(tf.abs(recover-target))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

batchSize = 128
src = 'SF1'
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(50):
        for j in range(40):
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
