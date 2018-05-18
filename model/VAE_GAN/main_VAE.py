import tensorflow as tf
import numpy as np
import time
import math
from function.analysis import (loadData, pickTransferInput, sythesis, nextbatch)
from function.ConvVAE import ConvVAE


TrainDataPath = '../../vcc2016/TFrecords//spec_norm//Train/'
TestDataPath = '../../vcc2016/TFrecords//spec_norm//Test/'

dataSize = 513
latentSize = 64
speakerN = 10
lam = 0

# Load data
# tS = time.time()
# trainData = loadData(TrainDataPath)
# testData = loadData(TestDataPath)
# tE = time.time()
# print("loading data time: %f" % (tE-tS))

arch = {
        'z_dim': 256,
        'speaker_dim': 10,
        'encoder':
        {'channel': [16, 32, 64],
         'kernel' : [[7, 1], [7, 1], [7, 1]],
         'stride' : [[3, 1], [3, 1], [3, 1]]
         },
        'decoder':
        {'channel': [32, 16, 1],
         'kernel' : [[7, 1], [7, 1], [7, 1]],
         'stride' : [[3, 1], [3, 1], [3, 1]]
         }
        }
source = tf.placeholder(tf.float32, shape = [None, dataSize])
y = tf.placeholder(tf.int32, shape = [None,])
VAE = ConvVAE(arch)

x = tf.reshape(source, shape=[-1, dataSize, 1, 1])
z_mean, z_var = VAE.encoder(x)
epsilon = tf.random_normal(tf.shape(z_var))
std_var = tf.exp(0.5*z_var)
z = z_mean + tf.multiply(std_var, epsilon)
x_mu, x_var = VAE.decoder(z, y)
epsilon = tf.random_normal(tf.shape(x_var))
std_var = tf.exp(0.5*x_var)
recon_x = x_mu + tf.multiply(std_var, epsilon)
#### setting

batchSize = 128

logp = 0.5*tf.reduce_mean(tf.log(math.pi) + x_var + tf.divide(tf.pow((recon_x-x_mu), 2), tf.exp(x_var)))
KL = -0.5*tf.reduce_mean((1 + z_var - tf.square(z_mean) - tf.exp(z_var)), 1)
loss = tf.reduce_mean(KL+logp)
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(20):
        for j in range(2000):
                x_batch, y_batch = nextbatch(TrainDataPath, batchSize)
                train_step.run(feed_dict = {source: x_batch , y: y_batch})
                x_batch, y_batch = nextbatch(TestDataPath, batchSize)
                train_step.run(feed_dict={source: x_batch, y: y_batch})
        x_Evabatch, y_Evabatch = nextbatch(TestDataPath, batchSize)
        eva_logp, kl= sess.run([logp , KL], feed_dict={source: x_Evabatch, y: y_Evabatch})
        print('epoch %d' %(i))
        print('train log-probability: %f' %(np.mean(eva_logp)))
        print('KL: %f' %(np.mean(kl)))

src = 'SF1'
trg = 'TM3'
srcData, trgLabel, filename = pickTransferInput(TestDataPath, src, trg)
sp = sess.run(recon_x, feed_dict={source: srcData, y: trgLabel})
sp = np.reshape(sp, [-1, 513])
sythesis(TestDataPath, src, trg, sp, filename)

