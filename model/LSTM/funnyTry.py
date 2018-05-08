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

dataPath = 'C:\\Users\\Lab711_Jeff\\Desktop\\Voice Conversion\\vcc2016_mat\\origin\\'
speakerN = 10
batch_size = 20
nodeNum = 4000

Data = {}
for dirNames in os.listdir(dataPath):
    Data[dirNames] = []
    dirPath = dataPath + dirNames
    for filename in os.listdir(dirPath):
        filePath = dirPath + '\\' + filename
        Data[dirNames].append(sio.loadmat(filePath)['x'])
        
def conv(x, W, s):
    return tf.nn.conv2d(x, W, strides= s, padding= 'SAME')

def deconv(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides= [1, 2, 1, 1], padding= 'SAME')

def weight_ini(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_ini(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)
def pool(x, k, s):
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME')

def nextbatch(dat):
    label = []
    data = []
    for i in range(batch_size):
        randInd = random.choice(list(dat.keys()))
        randfile = random.choice(dat[randInd])
        randind = random.randint(0,randfile.shape[0]-4000)
        data.append(randfile[randind:randind+4000])
        ind = list(dat.keys()).index(randInd)
        lab = np.zeros((1, speakerN))
        lab[0][ind] = 1
        label.append(lab)
    data = np.array(data)
    label = np.array(label)
    data = data.reshape([batch_size, nodeNum, 1, 1])
    return data, label

def prelu(x, alpha):
    pos = tf.nn.relu(x)
    neg = alpha*(x-abs(x))*0.5
    return pos + neg
    
    
x = tf.placeholder(tf.float32, shape = [None, nodeNum, 1, 1])
fex_W1 = weight_ini([32, 1, 1, 256])
fex_b1 = bias_ini([256])



frec_W1 = weight_ini([32, 1, 1, 256])
frec_b1 = bias_ini([1])


f_cnn_h1 = conv(x, fex_W1, [1, 2, 1, 1]) + fex_b1
fex_a1 = tf.get_variable('alpha1', f_cnn_h1.get_shape()[-1], initializer=tf.constant_initializer(0.25))    
feature1 = prelu(f_cnn_h1, fex_a1)

f_dcnn_h1 = deconv(feature1, frec_W1, [tf.shape(x)[0], 4000, 1, 1])
frec_a1 = tf.get_variable('alpha2', f_dcnn_h1.get_shape()[-1], initializer=tf.constant_initializer(0.25))    
rec_x = prelu(f_dcnn_h1, frec_a1)

mse = tf.reduce_mean(tf.pow(rec_x-x, 2))
optimizer = tf.train.AdamOptimizer(0.000001).minimize(mse)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


err = []
for epoch in range(2000):
    
    source, label = nextbatch(Data)
    optimizer.run(feed_dict={x:source})
    if(epoch%30 == 0):
        mseval = mse.eval(feed_dict={x:source})
        err.append(mseval)
        print('mse: %g' % (mseval))

input_x = Data['SF1'][0][0:4000].reshape([1, 4000, 1, 1])
spec = rec_x.eval(feed_dict={x:input_x})
input_x = Data['SF1'][0][4000:8000].reshape([1, 4000, 1, 1])
spec = np.concatenate((spec, rec_x.eval(feed_dict={x:input_x})), axis=2)
input_x = Data['SF1'][0][8000:12000].reshape([1, 4000, 1, 1])
spec = np.concatenate((spec, rec_x.eval(feed_dict={x:input_x})), axis=2)
input_x = Data['SF1'][0][12000:16000].reshape([1, 4000, 1, 1])
spec = np.concatenate((spec, rec_x.eval(feed_dict={x:input_x})), axis=2) 

sio.savemat('x.mat', mdict={'x': spec})
plt.plot(err)
plt.show()






