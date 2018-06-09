import tensorflow as tf
import numpy as np
import time
import scipy.io.wavfile
from util import (pickOne, nextbatch)
import layer

TrainDataPath = '../../vcc2016/TFrecords/raw/Train/'
TestDataPath = '../../vcc2016/TFrecords/raw/Test/'

latentSize = 32
NumOfspeaker = 10
N = 500
length = 80
tstep = 100
sampleRate = 16000
segmentLength = tstep*length

ArchOfgatedCNN = {'channel' : N,
             'kernel': [1, length],
             'stride': [1,1]}

arch = {
        'z_dim': latentSize,
        'speaker_dim': 10,
        'encoder':
            {
            'channel': [16, 32, 64],
            'kernel' : [[1, 7], [1, 7], [1, 7]],
            'stride' : [[1, 5], [1, 2], [1, 2]]
            },
        'decoder':
            {
            'channel': [32, 16, 1],
            'kernel' : [[1, 7], [1, 7], [1, 7]],
            'stride' : [[1, 5], [1, 2], [1, 2]]
            },
        'discriminator':
            {
            'channel': [8, 16, 32, 64],
            'kernel' : [[7, 1], [7, 1], [5, 1], [5, 1]],
            'stride' : [[3, 1], [3, 1], [3, 1], [1, 1]]
            }
        }
def encoder(x):
    x = tf.reshape(x, [-1, tstep, N, 1])
    unit = arch['encoder']
    c = unit['channel']
    k = unit['kernel']
    s = unit['stride']
    with tf.variable_scope('encoder', reuse = tf.AUTO_REUSE):
        for l in range(len(c)):
            x = layer.conv2d(x, c[l], k[l], s[l], layer.prelu,  name='encoder-L{}'.format(l))
        x = tf.reshape(x, [-1, 25*64])

        z_mu = tf.layers.dense(x, arch['z_dim'])
        z_var = tf.layers.dense(x, arch['z_dim'])
    return z_mu, z_var

def decoder(z, label):

    label = tf.tile(tf.expand_dims(label, 1), [1, tstep, 1])
    label = tf.reshape(label, [-1, label.get_shape()[2]])
    x = tf.layers.dense(z, 25*latentSize, bias_initializer=tf.constant_initializer(0.1))
    h = tf.layers.dense(label, 25*latentSize, bias_initializer=tf.constant_initializer(0.1))
    x_emb = x+h

    x = tf.reshape(x_emb, [-1, tstep, 25, latentSize])
    unit = arch['decoder']
    c = unit['channel']
    k = unit['kernel']
    s = unit['stride']
    with tf.variable_scope('decoder', reuse = tf.AUTO_REUSE):
        for l in range(len(c)):
            x = layer.deconv2d(x, c[l], k[l], s[l], layer.prelu,  name='decoder-L{}'.format(l))
        x = tf.reshape(x, [-1, N])
    return x, x_emb, h

source = tf.placeholder(tf.float32, shape = [None, segmentLength])
label = tf.placeholder(tf.float32, shape = [None, NumOfspeaker])

x = tf.reshape(source, [-1, tstep, length, 1])
gatedCNN_op1 = layer.gatedCNN(x, ArchOfgatedCNN, 'GCNN_en1')

gatedCNN_op1_flat = tf.reshape(gatedCNN_op1, [-1, tstep, ArchOfgatedCNN['channel']])

lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(N, forget_bias=1.0)
with tf.variable_scope('lstm_cell_1'):
    lstm_out, _ = tf.nn.dynamic_rnn(lstm_cell_1, gatedCNN_op1_flat, dtype=tf.float32)

Basis = tf.get_variable('Basis', shape=[N, length], initializer=tf.orthogonal_initializer)

Mask = tf.reshape(lstm_out, [-1, N])

z_mu, z_var = encoder(gatedCNN_op1_flat)
epsilon = tf.random_normal(tf.shape(z_var))
std_var = tf.exp(0.5*z_var)
z = z_mu + tf.multiply(std_var, epsilon)
Weight, z_emb, sp_emb = decoder(z, label)

recover = tf.matmul(tf.multiply(tf.reshape(gatedCNN_op1_flat, [-1, N]),Mask), Basis)
recover = tf.reshape(recover, [-1, segmentLength])

transfer = tf.matmul(tf.multiply(Weight,Mask), Basis)
transfer = tf.reshape(transfer, [-1, segmentLength])

recover_loss = tf.nn.l2_loss(source-recover)
transfer_loss = tf.nn.l2_loss(source-transfer)

KL = -0.5*tf.reduce_mean((1 + z_var - tf.square(z_mu) - tf.exp(z_var)), 1)


train_re_loss = tf.train.AdamOptimizer(0.0001).minimize(recover_loss)
train_trans_loss = tf.train.AdamOptimizer(0.0003).minimize(transfer_loss)
train_KL = tf.train.AdamOptimizer(0.00001).minimize(KL)


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

tS = time.time()
saver = tf.train.Saver()

batchSize = 32
is_training = True

if is_training:
        for epoch in range(10):
                for iter in range(100):
                        x_batch,label_batch  = nextbatch(TrainDataPath, segmentLength, batchSize)
                        train_re_loss.run(feed_dict = {source: x_batch, label: label_batch})
                        train_trans_loss.run(feed_dict = {source: x_batch, label: label_batch})
                        train_KL.run(feed_dict={source: x_batch, label: label_batch})
                loss_value, loss_KL = sess.run([recover_loss, KL], feed_dict = {source: x_batch, label: label_batch})
                print('epoch: %d' % (epoch))
                print('l2_loss: %f' % (loss_value))

        saver.save(sess, 'ckpt/model')
else:
        saver.restore(sess, tf.train.latest_checkpoint('ckpt/'))
        print('Model restored.')

src = 'SF1'
trg = 'SM1'

voice, trg_label, filename = pickOne(TestDataPath, src, trg, segmentLength)

Weight_ori, Weight_trans, b, m = sess.run([gatedCNN_op1_flat, Weight, Basis, Mask], feed_dict={source: voice, label: trg_label})
scipy.io.savemat(filename + '.mat', mdict={'Weight_ori': Weight_ori, 'Weight_trans': Weight_trans, 'Basis': b, 'Mask': m})

recover_voice, transfer_voice = sess.run([recover, transfer], feed_dict={source: voice, label: trg_label})
recover_voice = recover_voice.reshape([-1])
transfer_voice = transfer_voice.reshape([-1])
scipy.io.wavfile.write(filename + '_recover.wav', sampleRate, recover_voice)
scipy.io.wavfile.write(filename + '_transfer.wav', sampleRate, transfer_voice)


tE = time.time()
print("Training time: %f sec" % (tE-tS))