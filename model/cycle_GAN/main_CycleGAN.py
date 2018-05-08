import tensorflow as tf
import numpy as np
import time
from function.analysis import (loadData, pickTransferInput, sythesis, nextbatch)
from function.ConvWGAN import ConvWGAN

TrainDataPath = '../../vcc2016/mat_norm/Train/'
TestDataPath = '../../vcc2016/mat_norm/Test/'

frameSize = 128
dataSize = 513
latentSize = 64
speakerN = 10
lamb_cyc = 10
lamb_id = 5
src = 'SF1'
trg = 'TM2'
## Load data
# tS = time.time()
# trainData = loadData(TrainDataPath)
# testData = loadData(TestDataPath)
# tE = time.time()
# print("loading data time: %f" % (tE-tS))

arch = {
        'z_dim': 256,
        'speaker_dim': 10,
        'generator':
        {
        'downSample':
            {
                'channel': [16, 32, 64],
                'kernel' : [[7, 1], [7, 1], [7, 1]],
                'stride' : [[1, 1], [3, 1], [3, 1]]
            },
        'residual':
            {
                # 'channel': [128, 64,  128, 64, 128, 64],
                # 'kernel' : [[7, 3], [7, 3], [7, 3], [7, 3], [7, 3], [7, 3]],
                # 'stride' : [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
                'channel': [128, 64],
                'kernel' : [[7, 3], [7, 3]],
                'stride' : [[1, 1], [1, 1]]
            },
        'UpSample':
            {
                'channel': [32*9, 16*9, 1],
                'kernel' : [[7, 1], [7, 1], [7, 1]],
                'stride' : [[1, 1], [1, 1], [1, 1]],
                'ratio'  : [[3, 1], [3, 1]]
            }
         },
        'discriminator':
        {'channel': [16, 32, 64],
         'kernel' : [[5, 5], [5, 5], [5, 5]],
         'stride' : [[3, 2], [3, 2], [3, 2]]
         }
        }

Forward = ConvWGAN(arch)
Inverse = ConvWGAN(arch)

source = tf.placeholder(tf.float32, shape = [1, dataSize, frameSize, 1])
target = tf.placeholder(tf.float32, shape = [1, dataSize, frameSize, 1])

F_output = Forward.generator(source)
I_output = Inverse.generator(target)
F_loss_GAN = Forward.loss(F_output, target)
I_loss_GAN = Inverse.loss(I_output, source)
F_I_loss = tf.losses.absolute_difference(source, Inverse.generator(Forward.generator(source)))
I_F_loss = tf.losses.absolute_difference(target, Forward.generator(Inverse.generator(target)))

L_id = tf.losses.absolute_difference(Forward.generator(target), target) + tf.losses.absolute_difference(Inverse.generator(source), source)
L_cyc = F_I_loss + I_F_loss

Ld_F = F_loss_GAN['Ld']
Lg_F = F_loss_GAN['Lg']
Ld_I = I_loss_GAN['Ld']
Lg_I = I_loss_GAN['Lg']

G_loss = Lg_F + Lg_I
D_loss = Ld_F + Ld_I + lamb_cyc*L_cyc + lamb_id*L_id

G_train = tf.train.AdamOptimizer(0.0002).minimize(G_loss)
D_train = tf.train.AdamOptimizer(0.0001).minimize(D_loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(60):
        for j in range(162):
                x_batch, y_batch = nextbatch(TrainDataPath, src, trg)
                G_train.run(feed_dict = {source: x_batch , target: y_batch})
                D_train.run(feed_dict = {source: x_batch , target: y_batch})
        x_Evabatch, y_Evabatch = nextbatch(TestDataPath, src, trg)
        loss_g, loss_d= sess.run([G_loss, D_loss], feed_dict={source: x_Evabatch, target: y_Evabatch})
        print('epoch %d' %(i))
        print('generator loss: %f' %(np.mean(loss_g)))
        print('discriminator loss: %f' % (np.mean(loss_d)))

lamb_id = 0
for i in range(200):
        for j in range(162):
                x_batch, y_batch = nextbatch(TrainDataPath, src, trg)
                G_train.run(feed_dict = {source: x_batch , target: y_batch})
                D_train.run(feed_dict = {source: x_batch , target: y_batch})
        x_Evabatch, y_Evabatch = nextbatch(TestDataPath, src, trg)
        loss_g, loss_d= sess.run([G_loss, D_loss], feed_dict={source: x_Evabatch, target: y_Evabatch})
        print('epoch %d' %(i))
        print('generator loss: %f' %(np.mean(loss_g)))
        print('discriminator loss: %f' % (np.mean(loss_d)))

src = 'SF1'
trg = 'TM3'
srcData, trgLabel, filename = pickTransferInput(TestDataPath, src, trg)
sp = sess.run(F_output, feed_dict={source: srcData, source: trgLabel})
sp = np.reshape(sp, [-1, 513])
sythesis(TestDataPath, src, trg, sp, filename)


