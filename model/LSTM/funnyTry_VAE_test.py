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
from function.layer import (conv2d, deconv2d, prelu)

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
    
x = tf.placeholder(tf.float32, shape = [None, nodeNum, 1, 1])

feature = conv2d(x, 512, [5, 1], [2, 1], prelu, 'ex_feature')

en_h1= conv2d(feature, 256, [5, 1], [2, 1], prelu, 'en_h1')
en_h2 = conv2d(en_h1, 64, [5, 1], [2, 1], prelu, 'en_h2')

de_h1 = deconv2d(en_h2, 256, [5, 1], [2, 1], prelu, 'de_h1')
de_h2 = deconv2d(de_h1, 512, [5, 1], [2, 1], prelu, 'de_h2')

rec_x = deconv2d(de_h2, 1, [5, 1], [2, 1], prelu, 'rec_x')

mse = tf.reduce_mean(tf.pow(rec_x-x, 2))
optimizer = tf.train.AdamOptimizer(0.000001).minimize(mse)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


err = []
for epoch in range(200):
    source, label = nextbatch(Data)
    optimizer.run(feed_dict={x:source})
    if(epoch%100 == 0):
        mseval = mse.eval(feed_dict={x:source})
        err.append(mseval)
        print('mse: %g' % (mseval))

input_x = Data['SF1'][0][0:4000].reshape([1, 4000, 1, 1])
spec = rec_x.eval(feed_dict={x:input_x})
input_x = Data['SF1'][0][4000:8000].reshape([1, 4000, 1, 1])
spec = np.concatenate((spec, rec_x.eval(feed_dict={x:input_x})), axis=1)
input_x = Data['SF1'][0][8000:12000].reshape([1, 4000, 1, 1])
spec = np.concatenate((spec, rec_x.eval(feed_dict={x:input_x})), axis=1)
input_x = Data['SF1'][0][12000:16000].reshape([1, 4000, 1, 1])
spec = np.concatenate((spec, rec_x.eval(feed_dict={x:input_x})), axis=1) 

sio.savemat('x.mat', mdict={'x': spec})
plt.plot(err)
plt.show()






