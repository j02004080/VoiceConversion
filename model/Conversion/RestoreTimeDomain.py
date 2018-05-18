import tensorflow as tf
import numpy as np
import time
import scipy.io as sio
import matplotlib.pyplot as plt
from util import (loadData, pickTransferInput, nextbatch)
import layer
from speakerReg import speakerReg

TrainDataPath = '../../vcc2016/TFrecords/Time/Train/'
TestDataPath = '../../vcc2016/TFrecords/Time/Test/'

dataSize = 513
latentSize = 64
speakerN = 10
N = 500
L = 80
tstep = 100
hidNum = 1000
lamb = 0
tS = time.time()
trainData, Label = loadData(TrainDataPath, L, tstep)
# testData = loadData(TestDataPath)
tE = time.time()
print("loading data time: %f" % (tE-tS))

CGNNarch = {'channel' : N, 'kernel': [1, L], 'stride': [1,1]}
Regarch = {'channel' : [16, 32], 'kernel': [[1, 512], [1, 3]], 'stride': [[1,250], [1,2]], 'speaker_dim': speakerN}

source = tf.placeholder(tf.float32, shape = [None, tstep*L])
label = tf.placeholder(tf.float32, shape = [None, speakerN])
latent = tf.placeholder(tf.float32, shape = [None, N])
RegNet_en = speakerReg(Regarch, 'RegNet_en')
RegNet_de = speakerReg(Regarch, 'RegNet_de')

x = tf.reshape(source, [-1, tstep, L, 1])
GCNN_en1 = layer.gatedCNN(x, CGNNarch, 'GCNN_en1')

lstm_in = tf.reshape(GCNN_en1, [-1, tstep, CGNNarch['channel']])
W = tf.reshape(lstm_in, [-1, N])
Wt = RegNet_de.emb_trans(lstm_in, label)
Wt = tf.reshape(Wt, [-1, N])

flat = tf.contrib.layers.layer_norm(lstm_in)
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
# M = RegNet_de.emb_trans(lstm_out, label)
Op_Reg_en = RegNet_en.regnition(latent)
# Op_Reg_de = RegNet_de.regnition(M)
recover = tf.matmul(tf.multiply(Wt,M), B)
recover = tf.reshape(recover, [-1, tstep*L])




# loss=tf.reduce_mean(tf.abs(recover-source))

loss_Reg_en, Acu_Reg_en = RegNet_en.loss(Op_Reg_en, label)
# loss_Reg_de, Acu_Reg_de = RegNet_de.loss(Op_Reg_de, label)
loss=tf.nn.l2_loss(source-recover)

Reg_en_train = tf.train.AdamOptimizer(0.0005).minimize(loss_Reg_en)
# Reg_de_train = tf.train.AdamOptimizer(0.00001).minimize(loss_Reg_de)
train_step = tf.train.AdamOptimizer(0.0003).minimize(loss)

batchSize = 128
src = 'SF1'
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tS = time.time()
e=[]
saver = tf.train.Saver()

for i in range(10):
        # if i>2:
        #         lamb = 1
        for j in range(100):
                x_batch,label_batch  = nextbatch(trainData, Label, batchSize)
                train_step.run(feed_dict = {source: x_batch, label: label_batch})
                # Reg_de_train.run(feed_dict={source: x_batch, label: label_batch})
        x_Evabatch, label_Evabatch = nextbatch(trainData, Label, batchSize)
        eva_loss= sess.run([loss], feed_dict={source: x_Evabatch, label: label_Evabatch})
        e.append(eva_loss)
        print('epoch %d' %(i))
        print('reconstruct loss: %f' %(np.mean(eva_loss)))


for i in range(10):
        for j in range(100):
                x_batch, label_batch = nextbatch(trainData, Label, batchSize)
                de = sess.run(W, feed_dict={source: x_batch, label: label_batch})
                Reg_en_train.run(feed_dict={latent: de, label: label_batch})
        reg_loss, enacu = sess.run([loss_Reg_en, Acu_Reg_en], feed_dict={latent: de, label: label_batch})
        print('epoch %d' % (i))
        print('reg loss: %f' % (np.mean(reg_loss)))
        print('Encode accuracy: %f' % (np.mean(enacu)))

voice, trg_label, filename = pickTransferInput(TestDataPath, 'SF1', 'SM1', L*tstep)
output, emb, de = sess.run([recover, RegNet_en.speaker_emb, W], feed_dict={source: voice, label: trg_label})
reg_ans, acu = sess.run([Op_Reg_en, Acu_Reg_en], feed_dict={latent: de, label: trg_label})


saver.save(sess, '../../output/ckpt/test_model.ckt', global_step=0)

output = output.reshape([1, -1])
opname = '../../output/TimeDomain' + filename
sio.savemat(opname, mdict={'x': output, 'emb': emb})
plt.plot(e)

tE = time.time()
print("Training time: %f sec" % (tE-tS))