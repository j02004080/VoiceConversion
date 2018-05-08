import tensorflow as tf
import numpy as np
import scipy.io as sio
import os
import time
import math
import pickle 
import random
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import matplotlib.pyplot as plt

SourceDataPath = 'C:\\Users\\Lab711_Jeff\\Desktop\\Voice Conversion\\vcc2016_mat\\WORLD_spectrogram_normalized\\source_512\\'
TargetDataPath = 'C:\\Users\\Lab711_Jeff\\Desktop\\Voice Conversion\\vcc2016_mat\\WORLD_spectrogram_normalized\\target_512\\'
EvaDataPath = 'C:\\Users\\Lab711_Jeff\\Desktop\\Voice Conversion\\vcc2016_mat\\WORLD_spectrogram_normalized\\evaluate_512\\'
dataSize = 513
latentSize = 64
speakerN = 10
lam = 0

## Load data

def loadData(path):
        Data = {}
        Label = {}
        folder = os.listdir(path)
        for foldername in folder:
                filepath = path+foldername
                filename = os.listdir(filepath)
                for i in range(len(filename)):
                        Data[foldername] = []
                              
                for name in filename:
                        data = sio.loadmat(filepath + '\\' + name)
                        Data[foldername].append(data['speaker'][0][0][0])
        return Data

sourceData = loadData(SourceDataPath)
targetData = loadData(TargetDataPath)

def conv(x, W, s):
    return tf.nn.conv2d(x, W, strides= s, padding= 'SAME')

def weight_ini(shape):
	initial = tf.truncated_normal(shape, stddev = 0.001)
	return tf.Variable(initial)

def bias_ini(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def VAE_batch():
        n = random.randint(0, 9)
        key = list(sourceData.keys()) + list(targetData.keys())
        if(n<5):
                data = sourceData[key[n]][random.randint(0, len(sourceData[key[n]])-1)]
                startpoint = random.randint(0, data.shape[1]-128)
                fragment = data[:, startpoint:startpoint+128]
        else:
                data = targetData[key[n]][random.randint(0, len(targetData[key[n]])-1)]
                startpoint = random.randint(0, data.shape[1]-128)
                fragment = data[:, startpoint:startpoint+128]
        tar = np.zeros((1, speakerN))
        tar[0][n] = 1
        tar = np.tile(tar, (128, 1))
        return  np.transpose(fragment), tar


def GAN_batch():
        n = random.randint(0, 4)
        key = list(sourceData.keys()) + list(targetData.keys())
        
        Sdata = sourceData[key[n]][random.randint(0, len(sourceData[key[n]])-1)]
        startpoint = random.randint(0, Sdata.shape[1]-128)
        Sfragment = Sdata[:, startpoint:startpoint+128]

        n = n+5
        Tdata = targetData[key[n]][random.randint(0, len(targetData[key[n]])-1)]
        startpoint = random.randint(0, Tdata.shape[1]-128)
        Tfragment = np.reshape(Tdata[:, startpoint:startpoint+128], [-1, dataSize, 128, 1])
        
        tar = np.zeros((1, speakerN))
        tar[0][n] = 1
        tar = np.tile(tar, (128, 1))
        return np.transpose(Sfragment), Tfragment, tar

def discriminator(spectral_x):
    dc_h1 = tf.nn.relu(conv(spectral_x, dc_w1, [1, 1, 2, 1]) + dc_b1)
    dc_h2 = tf.nn.relu(conv(dc_h1, dc_w2, [1, 2, 2, 1]) + dc_b2)
    dc_h3 = tf.nn.relu(conv(dc_h2, dc_w3, [1, 2, 2, 1]) + dc_b3)
    dc_h3_flat = tf.reshape(dc_h3, [1, 129*16*64])
    dc_h4 = tf.sigmoid(tf.matmul(dc_h3_flat, dc_w4) + dc_b4)
    return dc_h4

def encoder(x):
        h_encode_1 = tf.nn.relu(tf.matmul(x, W_encode_1) + b_encode_1)
        h_encode_2 = tf.nn.relu(tf.matmul(h_encode_1, W_encode_2) + b_encode_2)
        z_mean = tf.matmul(h_encode_2, W_Zmean) + b_Zmean
        z_var = tf.matmul(h_encode_2, W_Zvar) + b_Zvar
        epsilon = tf.random_normal(tf.shape(z_var))
        std_var = tf.exp(0.5*z_var)
        z = z_mean + tf.multiply(std_var, epsilon)
        return z, z_mean, z_var

def generator(z, label):
        zy = tf.concat([z, label], 1)
        h_decode_1 = tf.nn.relu(tf.matmul(zy, W_decode_1) + b_decode_1)
        h_decode_2 = tf.nn.relu(tf.matmul(h_decode_1, W_decode_2) + b_decode_2)
        X_mean = tf.matmul(h_decode_2, W_Xmean) + b_Xmean
        X_var = tf.matmul(h_decode_2, W_Xvar) + b_Xvar
        epsilon = tf.random_normal(tf.shape(X_var))
        std_var = tf.exp(0.5*X_var)
        X = X_mean + tf.multiply(std_var, epsilon)
        return X, X_mean, X_var
        
source = tf.placeholder(tf.float32, shape = [None, dataSize])
label = tf.placeholder(tf.float32, shape = [None, speakerN])
target_spectral = tf.placeholder(tf.float32, shape = [None, dataSize, 128, 1])
latent_in = tf.placeholder(tf.float32, shape = [None, latentSize])

W_encode_1 = weight_ini([dataSize, 300])
W_encode_2 = weight_ini([300, 100])
W_Zmean = weight_ini([100, latentSize])
W_Zvar = weight_ini([100, latentSize])
W_decode_1 = weight_ini([latentSize+speakerN, 100])
W_decode_2 = weight_ini([100, 300])
W_Xmean = weight_ini([300, dataSize])
W_Xvar = weight_ini([300, dataSize])

b_encode_1 = bias_ini([300])
b_encode_2 = bias_ini([100])
b_Zmean = bias_ini([latentSize])
b_Zvar = bias_ini([latentSize])
b_decode_1 = bias_ini([100])
b_decode_2 = bias_ini([300])
b_Xmean = bias_ini([dataSize])
b_Xvar = bias_ini([dataSize])

dc_w1 = weight_ini([5, 5, 1, 16])
dc_w2 = weight_ini([3, 3, 16, 32])
dc_w3 = weight_ini([3, 3, 32, 64])
dc_w4 = weight_ini([129*16*64, 1])

dc_b1 = bias_ini([16])
dc_b2 = weight_ini([32])
dc_b3 = bias_ini([64])
dc_b4 = bias_ini([1])

z, z_mean, z_var = encoder(source)
X, X_mean, X_var = generator(z, label)
#### setting

trans_spec = tf.reshape(generator(latent_in, label)[0], [-1, dataSize, 128, 1])
Dz = discriminator(trans_spec)
Dx = discriminator(target_spectral)

logP = 0.5*tf.reduce_sum((tf.log(2*math.pi) + X_var + tf.divide(tf.square(source-X_mean), tf.exp(X_var))), 1)
KL = -0.5*tf.reduce_sum((1 + z_var - tf.square(z_mean) - tf.exp(z_var)), 1)
loss = tf.reduce_mean(logP)+tf.reduce_mean(KL)
Lg = 0.5*tf.reduce_mean(tf.square(Dz - 1))
Ld = 0.5*tf.reduce_mean(tf.square(Dx - 1)) + 0.5*tf.reduce_mean(tf.square(Dz))

vae_lambda = 0.001
VAE_train = tf.train.AdamOptimizer(vae_lambda).minimize(loss)
Discriminator_train = tf.train.AdamOptimizer(0.00001).minimize(Ld)
Generator_train = tf.train.AdamOptimizer(0.00001).minimize(Lg)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


for i in range(2000):
        source_batch, lab_batch = VAE_batch()
        VAE_train.run(feed_dict = {source: source_batch , label: lab_batch})


vae_lambda = 0.0001
for i in range(50):     
        for j in range(1000):
                for k in range(2):
                        source_batch, target_batch, lab_batch = GAN_batch()
                        latent_z = sess.run([z], feed_dict = {source: source_batch})
                        Discriminator_train.run(feed_dict = {latent_in: latent_z[0], target_spectral: target_batch, label: lab_batch})
                Generator_train.run(feed_dict = {latent_in: latent_z[0], label: lab_batch})      
                source_batch, lab_batch = VAE_batch()
                VAE_train.run(feed_dict = {source: source_batch , label: lab_batch})

                
        source_batch, lab_batch = VAE_batch()
        cost = sess.run([loss], feed_dict={source: source_batch, label: lab_batch})
        
        source_batch, target_batch, lab_batch = GAN_batch()
        latent_z = sess.run([z], feed_dict = {source: source_batch})
        dis_cost, gen_cost= sess.run([Ld, Lg], feed_dict={latent_in: latent_z[0], target_spectral: target_batch, label: lab_batch})
        
        print('epoch %d' %(i))
        print('loss: %f, Dis_loss: %f, Gen_loss: %f' %(np.mean(cost), dis_cost, gen_cost))
        
tar = np.zeros((1, speakerN))
tar[0][7] = 1
tar = np.tile(tar, (216, 1))

z = sess.run(z, feed_dict={source: sourceData[0:5000], label: sourceLabel[0:5000]})
recover = sess.run(X, feed_dict={source: sourceData[0:216, :], label: sourceLabel[0:216]})
transfer = sess.run(X, feed_dict={source: sourceData[0:216, :], label: tar})

sio.savemat('latent.mat', mdict={'latent': z})
sio.savemat('spec_recover.mat', mdict={'spec_recover': np.transpose(recover)})
sio.savemat('spec_transfer.mat', mdict={'spec_transfer': np.transpose(transfer)})

plt.scatter(z[:, 0], z[:, 1])
plt.show()
