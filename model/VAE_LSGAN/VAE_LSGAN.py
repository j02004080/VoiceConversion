import tensorflow as tf
import numpy as np
import scipy.io as sio
import os
import time
import pickle 
import random
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import matplotlib.pyplot as plt

SourceDataPath = 'C:\\Users\\Lab711_Jeff\\Desktop\\Voice Conversion\\vcc2016_mat\\source\\'
TargetDataPath = 'C:\\Users\\Lab711_Jeff\\Desktop\\Voice Conversion\\vcc2016_mat\\target\\'
EvaDataPath = 'C:\\Users\\Lab711_Jeff\\Desktop\\Voice Conversion\\vcc2016_mat\\evaluate\\'
dataSize = 257
latentSize = 64
speakerN = 10
lam = 0

## Load data

def loadData(path):

        Data = np.zeros((dataSize, 1))
        Phase = np.zeros((dataSize, 1))
        Label = np.zeros((1, speakerN))
        
        filename = os.listdir(path)
        for name in filename:     
                data = sio.loadmat(path+name)
                Data = np.concatenate((Data, data['spec']), axis = 1)
                Phase = np.concatenate((Phase, data['phase']), axis = 1)
                Label = np.concatenate((Label, data['label']), axis = 0)
        Data = np.transpose(Data)
        Phase = np.transpose(Phase)
        Data = np.delete(Data, 0, 0)
        Phase = np.delete(Phase, 0, 0)
        Label = np.delete(Label, 0, 0)
        return Data, Phase, Label

sourceData, sourcePhase, sourceLabel = loadData(SourceDataPath)
targetData, targetPhase, targetLabel = loadData(TargetDataPath)
EvaData, EvaPhase, EvaLabel = loadData(EvaDataPath)


def weight_ini(shape):
	initial = tf.truncated_normal(shape, stddev = 0.001)
	return tf.Variable(initial)

def bias_ini(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def VAE_batch(dat, lab, mini_batch):
        randNum = random.sample(range(dat.shape[0]), mini_batch)
        return dat[randNum,  :], lab[randNum, :]

def GAN_batch(dat1, dat2, lab, mini_batch):
        randNum_source = random.sample(range(dat1.shape[0]), mini_batch)
        randNum_target = random.sample(range(dat2.shape[0]), mini_batch)
        return dat1[randNum_source,  :], dat2[randNum_target,  :], lab[randNum_target, :]

def discriminator(spectral_x):
    dc_h1 = tf.nn.relu(tf.matmul(spectral_x, dc_w1) + dc_b1)
    dc_h2 = tf.sigmoid(tf.matmul(dc_h1, dc_w2) + dc_b2)
    return dc_h2

def encoder(x):
        h_encode1 = tf.nn.relu(tf.matmul(x, W_encode1) + b_encode1)
        z_mean = tf.matmul(h_encode1, W_mean) + b_mean
        z_var = tf.matmul(h_encode1, W_var) + b_var
        epsilon = tf.random_normal(tf.shape(z_var))
        std_var = tf.exp(0.5*z_var)
        z = z_mean + tf.multiply(std_var, epsilon)
        return z, z_mean, z_var

def generator(z, label):
        zy = tf.concat([z, label], 1)
        h_decode1 = tf.nn.relu(tf.matmul(zy, W_decode1) + b_decode1)
        y = tf.nn.relu(tf.matmul(h_decode1, W_decode2) + b_decode2)
        return y
        
source = tf.placeholder(tf.float32, shape = [None, dataSize])
label = tf.placeholder(tf.float32, shape = [None, speakerN])
target_spectral = tf.placeholder(tf.float32, shape = [None, dataSize])
latent_in = tf.placeholder(tf.float32, shape = [None, latentSize])

W_encode1 = weight_ini([257, 128])
W_mean = weight_ini([128, latentSize])
W_var = weight_ini([128, latentSize])
W_decode1 = weight_ini([latentSize + speakerN, 128])
W_decode2 = weight_ini([128, 257])

b_encode1 = bias_ini([128])
b_mean = bias_ini([latentSize])
b_var = bias_ini([latentSize])
b_decode1 = bias_ini([128])
b_decode2 = bias_ini([257])

dc_w1 = weight_ini([dataSize, 32])
dc_w2 = weight_ini([32, 1])

dc_b1 = bias_ini([32])
dc_b2 = bias_ini([1])

z, z_mean, z_var = encoder(source)
reconstruct_spectral = generator(z, label)
#### setting

batchSize = 128

Dz = discriminator(generator(latent_in, label))
Dx = discriminator(target_spectral)

MSE = tf.reduce_sum(tf.squared_difference(reconstruct_spectral, source), 1)
KL = -0.5*tf.reduce_sum((1 + z_var - tf.square(z_mean) - tf.exp(z_var)), 1)
loss = tf.reduce_mean(MSE)+tf.reduce_mean(KL)
Lg = 0.5*tf.reduce_mean(tf.square(Dz - 1))
Ld = 0.5*tf.reduce_mean(tf.square(Dx - 1)) + 0.5*tf.reduce_mean(tf.square(Dz))

VAE_train = tf.train.AdamOptimizer(0.001).minimize(loss)
Discriminator_train = tf.train.AdamOptimizer(0.0001).minimize(Ld)
Generator_train = tf.train.AdamOptimizer(0.0002).minimize(Lg)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


for i in range(2000):
        source_batch, Slab_batch = VAE_batch(sourceData, sourceLabel, batchSize)
        target_batch, Tlab_batch = VAE_batch(targetData, targetLabel, batchSize)
        VAE_train.run(feed_dict = {source: source_batch , label: Slab_batch})
        VAE_train.run(feed_dict = {source: target_batch , label: Tlab_batch})

for i in range(50):     
        for j in range(500):
                for k in range(5):
                        source_batch, target_batch, lab_batch = GAN_batch(sourceData, targetData, targetLabel, batchSize)
                        latent_z = sess.run([z], feed_dict = {source: source_batch})
##                        latent_z = np.array(latent_z[0])
                        Discriminator_train.run(feed_dict = {latent_in: latent_z[0], target_spectral: target_batch, label: lab_batch})
                Generator_train.run(feed_dict = {latent_in: latent_z[0], label: lab_batch})      
                source_batch, Slab_batch = VAE_batch(sourceData, sourceLabel, batchSize)
                target_batch, Tlab_batch = VAE_batch(targetData, targetLabel, batchSize)
                VAE_train.run(feed_dict = {source: source_batch , label: Slab_batch})
                VAE_train.run(feed_dict = {source: target_batch , label: Tlab_batch})
                
##        eva_source_batch, eva_target_batch, eva_lab_batch = GAN_batch(EvaData, EvaData, EvaLabel, batchSize)
        source_batch, Slab_batch = VAE_batch(sourceData, sourceLabel, batchSize)
        cost = sess.run([loss], feed_dict={source: source_batch, label: Slab_batch})
        
        source_batch, target_batch, lab_batch = GAN_batch(sourceData, targetData, targetLabel, batchSize)
        latent_z = sess.run([z], feed_dict = {source: source_batch})
##        latent_z = np.array(latent_z[0])
        dis_cost, gen_cost= sess.run([Ld, Lg], feed_dict={latent_in: latent_z[0], target_spectral: target_batch, label: lab_batch})
        
        print('epoch %d' %(i))
        print('loss: %f, Dis_loss: %f, Gen_loss: %f' %(np.mean(cost), dis_cost, gen_cost))
        
tar = np.zeros((1, speakerN))
tar[0][7] = 1
tar = np.tile(tar, (86, 1))

z = sess.run(z, feed_dict={source: sourceData[0:5000], label: sourceLabel[0:5000]})
recover = sess.run(reconstruct_spectral, feed_dict={source: sourceData[0:86], label: sourceLabel[0:86]})
transfer = sess.run(reconstruct_spectral, feed_dict={source: sourceData[0:86], label: tar})

recover = recover*sourcePhase[0:86, :]
transfer = transfer*sourcePhase[0:86, :]

recover = np.transpose(recover)
transfer = np.transpose(transfer)

sio.savemat('latent.mat', mdict={'latent': z})
sio.savemat('sp1.mat', mdict={'spec': recover})
sio.savemat('sp2.mat', mdict={'spec': transfer})

plt.scatter(z[:, 0], z[:, 1])
plt.show()
