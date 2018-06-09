import tensorflow as tf
from util import (nextbatch, pickOne, synthesis)
from model_VAE import VAE
import time
import numpy as np

featureSize = 513
NumOfspeaker = 10

TrainDataPath = '../../vcc2016/TFrecords/spec_norm/Train/'
TestDataPath = '../../vcc2016/TFrecords/spec_norm/Test/'

source = tf.placeholder(tf.float32, shape = [None, featureSize])
label = tf.placeholder(tf.float32, shape = [None, NumOfspeaker])

arch = {
    'featureSize': featureSize,
    'z_dim': 128,
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
    }
}

model = VAE(arch)
z_mu, z_logvar = model.encoder(source)
eps = tf.random_normal(tf.shape(z_logvar), dtype= tf.float32, mean = 0., stddev = 1.0, name = 'epsilon')
z = z_mu + eps*tf.exp(0.5*z_logvar)
recover = model.decoder(z, label)

loss_MS = tf.nn.l2_loss(source-recover)
loss_KL = -0.5*tf.reduce_sum(1 + z_logvar - tf.square(z_mu) - tf.exp(z_logvar), reduction_indices=1)

train_MS = tf.train.AdamOptimizer(0.0001).minimize(loss_MS)
train_KL = tf.train.AdamOptimizer(0.00001).minimize(loss_KL)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
saver = tf.train.Saver()
time_start = time.time()
batchSize = 128
is_training = True

if is_training:
    for epoch in range(5):
        for iter in range(2000):
            src_batch, label_batch = nextbatch(TrainDataPath, featureSize, batchSize)
            train_MS.run(feed_dict={source: src_batch, label: label_batch})
            train_KL.run(feed_dict={source: src_batch, label: label_batch})

        ms_value, kl_value = sess.run([loss_MS, loss_KL], feed_dict={source: src_batch, label: label_batch})
        print('epoch: %d' % (epoch))
        print('MSE: %f' % (ms_value))
        print('KL: %f' % (np.mean(kl_value)))
    saver.save(sess, 'ckpt/VAE_model.ckpt')
else:
    saver.restore(sess, tf.train.latest_checkpoint('ckpt/'))
    print('Model restored.')

time_end = time.time()
print('Training time: %f sec' % (time_end - time_start))

src = 'SF1'
trg = 'SM1'
src_spec, trg_label, filename = pickOne(TestDataPath, src, trg)
output = sess.run(recover, feed_dict={source: src_spec, label: trg_label})
synthesis(TestDataPath, src, trg, output, filename)

