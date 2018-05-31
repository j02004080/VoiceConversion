import tensorflow as tf
import numpy as np
import time
import scipy.io as sio
from function.analysis import (loadData, pickTransferInput, sythesis, nextbatch)
from function.VAE_WGAN import ConvVAE_WGAN


TrainDataPath = '../../vcc2016/TFrecords/spec_norm/Train/'
TestDataPath = '../../vcc2016/TFrecords/spec_norm/Test/'

dataSize = 513
latentSize = 128
speakerN = 10
lam = 0

# ## Load data

tS = time.time()
trainData = loadData(TrainDataPath)
testData = loadData(TestDataPath)
tE = time.time()
print("loading data time: %f" % (tE-tS))

arch = {
        'z_dim': latentSize,
        'featureSize': dataSize,
        'speaker_dim': 10,
        'encoder':
            {
            'channel': [16, 32, 64],
            'kernel' : [[7, 1], [7, 1], [7, 1]],
            'stride' : [[3, 1], [3, 1], [3, 1]]
            },
        'generator':
            {
            'channel': [32, 16, 1],
            'kernel' : [[7, 1], [7, 1], [7, 1]],
            'stride' : [[3, 1], [3, 1], [3, 1]]
            },
        'discriminator':
            {
            'channel': [8, 16, 32, 64],
            'kernel' : [[7, 1], [7, 1], [5, 1], [5, 1]],
            'stride' : [[3, 1], [3, 1], [3, 1], [1, 1]]
            }
        }
source = tf.placeholder(tf.float32, shape = [None, dataSize])
z_in = tf.placeholder(tf.float32, shape = [None, latentSize])
y = tf.placeholder(tf.float32, shape = [None, 10])
model = ConvVAE_WGAN(arch)

z_mean, z_var = model.encoder(source)
epsilon = tf.random_normal(tf.shape(z_var))
std_var = tf.exp(0.5*z_var)
z = z_mean + tf.multiply(std_var, epsilon)
recon_x = model.generator(z, y)
transfer = model.generator(z_in, y)

#### setting

batchSize = 128

loss_GAN = model.loss(source, transfer)
Ld = loss_GAN['Ld']
Lg = loss_GAN['Lg']
MAE = tf.losses.absolute_difference(source, recon_x)
KL = -0.5*tf.reduce_sum((1 + z_var - tf.square(z_mean) - tf.exp(z_var)), 1)

trainables = tf.trainable_variables()
g_vars = [v for v in trainables if 'generator' in v.name]
d_vars = [v for v in trainables if 'discriminator' in v.name]

loss = tf.reduce_mean(KL+MAE)
VAE_train = tf.train.AdamOptimizer(0.00001).minimize(loss)
G_train = tf.train.AdamOptimizer(0.0001).minimize(Lg, var_list = g_vars)
D_train = tf.train.AdamOptimizer(0.0001).minimize(Ld, var_list = d_vars)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tS = time.time()
saver = tf.train.Saver()

is_training = True

if is_training:
    for i in range(10):
            for j in range(2000):
                    x_batch, y_batch = nextbatch(trainData, batchSize)
                    VAE_train.run(feed_dict = {source: x_batch , y: y_batch})
            x_Evabatch, y_Evabatch = nextbatch(testData, batchSize)
            eva_logp, kl= sess.run([MAE , KL], feed_dict={source: x_Evabatch, y: y_Evabatch})
            print('epoch %d' %(i))
            print('train log-probability: %f' %(np.mean(eva_logp)))
            print('KL: %f' %(np.mean(kl)))

    for i in range(20):
            for j in range(2000):
                src_batch, y_batch = nextbatch(trainData, batchSize)
                latent = sess.run(z, feed_dict={source: src_batch, y: y_batch})
                D_train.run(feed_dict={source: src_batch, z_in: latent, y: y_batch})
                for k in range(4):
                    G_train.run(feed_dict = {source: src_batch, z_in: latent, y: y_batch})

                x_batch, y_batch = nextbatch(trainData, batchSize)
                VAE_train.run(feed_dict = {source: x_batch , y: y_batch})

            src_batch, y_batch = nextbatch(trainData, batchSize)
            loss_d, loss_g = sess.run([Ld, Lg], feed_dict={source: src_batch, z_in: latent, y: y_batch})
            print('epoch %d' %(i))
            print('discriminator loss: %f' % (loss_d))
            print('generative loss: %f' %(loss_g))

    saver.save(sess, 'output/ckpt/model.ckpt')
else:
    saver.restore(sess, tf.train.latest_checkpoint('output/ckpt/'))
    print('Model restored.')

src = 'SF1'
trg = 'SF1'
srcData, trgLabel, filename = pickTransferInput(TestDataPath, src, trg)
sp, speaker_emb = sess.run([recon_x, model.speaker_emb], feed_dict={source: srcData, y: trgLabel})
sp = np.reshape(sp, [sp.shape[0], 513])
sythesis(TestDataPath, src, trg, sp, filename)
sio.savemat('speaker_emb', mdict={'speaker_emb': speaker_emb})
tE = time.time()
print("Training time: %f sec" % (tE-tS))