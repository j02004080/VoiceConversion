import tensorflow as tf
import numpy as np
import time
import scipy.io as sio
import matplotlib.pyplot as plt
from util import (loadData, pickTransferInput, nextbatch)
import layer
from speakerReg import speakerReg
from WGAN import WGAN

TrainDataPath = '../../vcc2016/TFrecords/Time/Train/'
TestDataPath = '../../vcc2016/TFrecords/Time/Test/'

is_training = True
dataSize = 513
latentSize = 64
speakerN = 10
N = 500
L = 80
tstep = 100
lamb=0
global_setp = 1

tS = time.time()
trainData, Label = loadData(TrainDataPath, L, tstep)
# testData = loadData(TestDataPath)
tE = time.time()
print("loading data time: %f" % (tE-tS))

CGNNarch = {'channel' : N, 'kernel': [1, L], 'stride': [1,1]}
Regarch = {'channel' : [16, 32], 'kernel': [[1, 512], [1, 3]], 'stride': [[1,250], [1,2]], 'speaker_dim': speakerN}

GANarch = {
    'speaker_dim': speakerN,
    'gen': {
        'channel' : [32, 1], 'kernel': [[3, 3], [3, 3]], 'stride': [[2,2], [2,2]], 'hidNum': [256, 64, 256, 500]
    },
    'dis':{
        'channel' : [16, 32], 'kernel': [[1, 512], [1, 3]], 'stride': [[1, 250], [1,2]]
    }}

source = tf.placeholder(tf.float32, shape = [None, tstep*L])
label = tf.placeholder(tf.float32, shape = [None, speakerN])
W_ori = tf.placeholder(tf.float32, shape = [None, N])

# RegNet_ori = speakerReg(Regarch, 'RegNet_ori')
# RegNet_M = speakerReg(Regarch, 'RegNet_M')
W_GAN = WGAN(GANarch, 'WGAN')

x = tf.reshape(source, [-1, tstep, L, 1])
GCNN_en1 = layer.gatedCNN(x, CGNNarch, 'GCNN_en1')

lstm_in = tf.reshape(GCNN_en1, [-1, tstep, CGNNarch['channel']])
W = tf.reshape(lstm_in, [-1, N])

flat = tf.contrib.layers.layer_norm(lstm_in)
lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(N, forget_bias=1.0)
with tf.variable_scope('lstm_cell_1'):
    lstm_out, _ = tf.nn.dynamic_rnn(lstm_cell_1, flat, dtype=tf.float32)

B = tf.get_variable('Basis', shape=[N, L], initializer=tf.random_normal_initializer(stddev=0.01))

M = tf.reshape(lstm_out, [-1, N])


recover = tf.matmul(tf.multiply(W,M), B)
recover = tf.reshape(recover, [-1, tstep*L])

W_trans = W_GAN.generator(W_ori, label)
transfer = tf.matmul(tf.multiply(W_trans,M), B)
transfer = tf.reshape(transfer, [-1, tstep*L])

# Op_Reg_ori = RegNet_ori.regnition(W_ori)
# Op_Reg_M = RegNet_M.regnition(W_trans)


# loss_Reg_ori, Acu_Reg_ori = RegNet_ori.loss(Op_Reg_ori, label)
# loss_Reg_M, Acu_Reg_M = RegNet_M.loss(Op_Reg_M, label)
loss_D, loss_G = W_GAN.loss(source, transfer)

loss=tf.nn.l2_loss(source-recover)
# loss=tf.nn.l2_loss(source-recover) - lamb*loss_Reg_ad

trainables = tf.trainable_variables()
g_vars = [v for v in trainables if 'generator' in v.name]
d_vars = [v for v in trainables if 'discriminator' in v.name]

# Reg_ori_train = tf.train.AdamOptimizer(0.0005).minimize(loss_Reg_ori)
# Reg_M_train = tf.train.AdamOptimizer(0.0005).minimize(loss_Reg_M)
train_step = tf.train.AdamOptimizer(0.0003).minimize(loss)
G_train = tf.train.AdamOptimizer(0.001, beta1=0.5, beta2=0.9).minimize(loss_G, var_list = g_vars)
D_train = tf.train.AdamOptimizer(0.001, beta1=0.5, beta2=0.9).minimize(loss_D, var_list = d_vars)

batchSize = 128
src = 'SF1'
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tS = time.time()




e=[]
saver = tf.train.Saver()

if is_training:
        for i in range(20):
                if i<5:
                        for j in range(100):
                                x_batch,label_batch  = nextbatch(trainData, Label, batchSize)
                                train_step.run(feed_dict = {source: x_batch, label: label_batch})
                else:
                        for j in range(100):
                                x_batch, label_batch = nextbatch(trainData, Label, batchSize)
                                train_step.run(feed_dict={source: x_batch, label: label_batch})
                                GAN_in = sess.run(W, feed_dict={source: x_batch, label: label_batch})
                                D_train.run(feed_dict={source: x_batch, W_ori: GAN_in, label: label_batch})
                                for k in range(4):
                                        G_train.run(feed_dict={source: x_batch, W_ori: GAN_in, label: label_batch})

                        x_Evabatch, label_Evabatch = nextbatch(trainData, Label, batchSize)
                        GAN_in = sess.run(W, feed_dict={source: x_Evabatch, label: label_Evabatch})
                        eva_loss, Dis_loss, Gen_loss= sess.run([loss, loss_D, loss_G], feed_dict={source: x_Evabatch, W_ori: GAN_in, label: label_Evabatch})
                        e.append(eva_loss)
                        print('epoch %d' %(i))
                        print('reconstruct loss: %f' %(np.mean(eva_loss)))
                        print('Discriminator loss: %f' % (Dis_loss))
                        print('Generator loss: %f' % (Gen_loss))


        # for i in range(10):
        #         for j in range(100):
        #                 x_batch, label_batch = nextbatch(trainData, Label, batchSize)
        #                 l_W, l_M = sess.run([W, M], feed_dict={source: x_batch, label: label_batch})
        #                 Reg_ori_train.run(feed_dict={latent_W: l_W, label: label_batch})
        #                 Reg_M_train.run(feed_dict={latent_M: l_M, label: label_batch})
        #         acu_M, acu_W = sess.run([Acu_Reg_M, Acu_Reg_ori], feed_dict={latent_W: l_W, latent_M: l_M, label: label_batch})
        #         print('epoch %d' % (i))
        #         print('W regnition accuracy: %f' % (np.mean(acu_W)))
        #         print('M regnition accuracy: %f' % (np.mean(acu_M)))
        # # saver.save(sess, '../../output/ckpt/test_model.ckt', global_step=global_setp)

else:
        saver.restore(sess, tf.train.latest_checkpoint('../../output/ckpt/'))
        print('Model restored.')

voice, trg_label, filename = pickTransferInput(TestDataPath, 'SF1', 'SF1', L*tstep)
W_in = sess.run(W, feed_dict={source: voice, label: trg_label})
output_rec, output_trans = sess.run([recover, transfer], feed_dict={source: voice, W_ori: W_in, label: trg_label})
# W_reg, M_reg = sess.run([Op_Reg_ori, Op_Reg_M], feed_dict={latent_W: l_W, latent_M: l_M, label: trg_label})

output_rec = output_rec.reshape([1, -1])
output_trans = output_trans.reshape([1, -1])
opname = '../../output/TimeDomain' + filename
sio.savemat(opname, mdict={'x': output_rec, 'y': output_trans})
plt.plot(e)

tE = time.time()
print("Training time: %f sec" % (tE-tS))