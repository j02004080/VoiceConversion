import tensorflow as tf
from model_VAE import VAE
from speakerReg import speakerReg
from util import (nextbatch, pickOne)
import scipy
import numpy as np
import time

featureSize = 513
NumOfspeaker = 10
latentSize = 64

TrainDataPath = '../../vcc2016/spectrogram/Train/'
TestDataPath = '../../vcc2016/spectrogram/Test/'

arch = {
    'featureSize': featureSize,
    'z_dim': latentSize,
    'encoder': {
        'channel': [16, 32, 64],
        'kernel': [[7, 1], [7, 1], [7, 1]],
        'stride': [[3, 1], [3, 1], [3, 1]]
    },
    'decoder':{
        'hc': [19, 64],
        'channel': [32, 16, 1],
        'kernel': [[7, 1], [7, 1], [7, 1]],
        'stride': [[3, 1], [3, 1], [3, 1]]
    },
    'regnizer':{
        'channel': [16, 32, 64],
        'kernel': [[7, 1], [7, 1], [7, 1]],
        'stride': [[2, 1], [2, 1], [2, 1]]
    }
}

source = tf.placeholder(tf.float32, shape=[None, featureSize])
label = tf.placeholder(tf.float32, shape=[None, NumOfspeaker])

model = VAE(arch)
regnition = speakerReg(arch)

z_mu, z_logvar = model.encoder(source)
eps = tf.random_normal(tf.shape(z_logvar), dtype= tf.float32, mean = 0., stddev = 1.0, name = 'epsilon')
z = z_mu + eps*tf.exp(0.5*z_logvar)
recover, speaker_emb = model.decoder(z, label)
loss_Reg, accuracy_Reg = regnition.loss(z, label)

loss_MS = tf.nn.l2_loss(source-recover)
loss_KL = -0.5*tf.reduce_sum(1 + z_logvar - tf.square(z_mu) - tf.exp(z_logvar), reduction_indices=1)

trainables = tf.trainable_variables()

train_MS = tf.train.AdamOptimizer(0.0001).minimize(loss_MS)
train_KL = tf.train.AdamOptimizer(0.00001).minimize(loss_KL)
train_Reg = tf.train.AdamOptimizer(0.0001).minimize(loss_Reg)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
saver = tf.train.Saver()
time_start = time.time()
batchSize = 128
is_training = True

if is_training:
    for epoch in range(15):
        for iter in range(1000):
            src_batch, label_batch = nextbatch(TrainDataPath, featureSize, batchSize)
            train_MS.run(feed_dict={source: src_batch, label: label_batch})
            train_KL.run(feed_dict={source: src_batch, label: label_batch})

        ms_value, kl_value = sess.run([loss_MS, loss_KL], feed_dict={source: src_batch, label: label_batch})
        print('epoch: %d' % (epoch))
        print('MSE: %f' % (ms_value))
        print('KL: %f' % (np.mean(kl_value)))
    # for epoch in range(10):
    #     for iter in range(1000):
    #         src_batch, label_batch = nextbatch(TrainDataPath, featureSize, batchSize)
    #         train_Reg.run(feed_dict={source: src_batch, label: label_batch})
    #     reg_acu = sess.run(accuracy_Reg, feed_dict={source: src_batch, label: label_batch})
    #     print('epoch: %d' % (epoch))
    #     print('MSE: %f' % (reg_acu))
    # saver.save(sess, 'ckpt/VAE_model.ckpt')
else:
    saver.restore(sess, tf.train.latest_checkpoint('ckpt/'))
    print('Model restored.')

time_end = time.time()
print('Training time: %f sec' % (time_end - time_start))

src = 'SM1'
trg = 'SF1'
src_spec, trg_label, filename = pickOne(TestDataPath, src, trg)
output, latent, sp_emb = sess.run([recover, z, speaker_emb], feed_dict={source: src_spec, label: trg_label})
scipy.io.savemat(src + '_' + trg + filename + '.mat', mdict = {'spectrogram': output})

