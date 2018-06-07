import tensorflow as tf
import numpy as np
import time
import scipy.io.wavfile
from util import (pickOne, nextbatch)
import layer

TrainDataPath = '../../vcc2016/TFrecords/raw/Train/'
TestDataPath = '../../vcc2016/TFrecords/raw/Test/'

latentSize = 128
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

Basis = tf.get_variable('Basis', shape=[N, length], initializer=tf.random_normal_initializer(stddev=0.01))

Mask = tf.reshape(lstm_out, [-1, N])

z_mu, z_var = encoder(gatedCNN_op1_flat)
epsilon = tf.random_normal(tf.shape(z_var))
std_var = tf.exp(0.5*z_var)
z = z_mu + tf.multiply(std_var, epsilon)
Weight, z_emb, sp_emb = decoder(z, label)

recover = tf.matmul(tf.multiply(Weight,Mask), Basis)
recover = tf.reshape(recover, [-1, segmentLength])

loss=tf.nn.l2_loss(source-recover)
KL = -0.5*tf.reduce_mean((1 + z_var - tf.square(z_mu) - tf.exp(z_var)), 1)

train_loss = tf.train.AdamOptimizer(0.0003).minimize(loss)
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
                        train_loss.run(feed_dict = {source: x_batch, label: label_batch})
                        train_KL.run(feed_dict={source: x_batch, label: label_batch})
                loss_value, loss_KL = sess.run([loss, KL], feed_dict = {source: x_batch, label: label_batch})
                print('epoch: %d' % (epoch))
                print('l2_loss: %f' % (loss_value))

        saver.save(sess, 'ckpt/model')
else:
        saver.restore(sess, tf.train.latest_checkpoint('ckpt/'))
        print('Model restored.')

voice, trg_label, filename = pickOne(TestDataPath, 'SF1', 'SM1', segmentLength)

emb, s_emb = sess.run([z_emb, sp_emb], feed_dict={source: voice, label: trg_label})
scipy.io.savemat(filename+'.mat', mdict={'latent': emb, 'speaker_emb': s_emb})

output = sess.run(recover, feed_dict={source: voice, label: trg_label})
output = output.reshape([-1])
scipy.io.wavfile.write(filename + '.wav', sampleRate, output)


tE = time.time()
print("Training time: %f sec" % (tE-tS))