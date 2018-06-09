import tensorflow as tf
import numpy as np
import time
import scipy.io as sio
import matplotlib.pyplot as plt
from util import (pickOne, nextbatch)
import layer

TrainDataPath = '../../vcc2016/TFrecords/raw/Train/'
TestDataPath = '../../vcc2016/TFrecords/raw/Test/'

speakerN = 10
N = 256
length = 80
tstep = 100
hidNum = 1000
segmentLength = tstep*length

arch = {'channel' : [16, 32], 'kernel': [[1, 512], [1, 3]], 'stride': [[1,40], [1,2]]}

source = tf.placeholder(tf.float32, shape = [None, segmentLength])
label = tf.placeholder(tf.float32, shape = [None, speakerN])
x = tf.reshape(source, [-1, 1, segmentLength, 1])

tS = time.time()

for i in range(len(arch['channel'])):
        x = layer.conv2d(x, arch['channel'][i], arch['kernel'][i], arch['stride'][i], tf.nn.relu, name='cnn-L{}'.format(i))
flat = tf.reshape(x, [-1, 100*arch['channel'][1]])
y = tf.layers.dense(flat, 10, bias_initializer=tf.constant_initializer(0.1))

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=y))
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

batchSize = 128

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(20):
        for j in range(200):
                x_batch, y_batch = nextbatch(TrainDataPath, segmentLength, batchSize)
                train_step.run(feed_dict = {source: x_batch, label: y_batch})
        x_Evabatch, y_Evabatch = nextbatch(TestDataPath, segmentLength, batchSize)
        tr_acu = sess.run(accuracy, feed_dict={source: x_batch, label: y_batch})
        te_acu= sess.run(accuracy, feed_dict={source: x_Evabatch, label: y_Evabatch})

        print('epoch %d' %(i))
        print('Train accuracy: %f' %(tr_acu))
        print('Test accuracy: %f' % (te_acu))

tE = time.time()
print("training time: %f" % (tE-tS))
