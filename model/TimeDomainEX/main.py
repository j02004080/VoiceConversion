import tensorflow as tf
import numpy as np
import scipy.io.wavfile
from layer import conv2d
from util import (loadTF, nextbatch, pickOne)

Trainfilepath = '../../vcc2016/TFrecords/raw/Train/'
Testfilepath = '../../vcc2016/TFrecords/raw/Test/'
filename = 'SF1/100002'

tstep = 100
length = 80
sampleRate = 16000
segmentLength = tstep*length

source = tf.placeholder(tf.float32, [None, segmentLength])
label = tf.placeholder(tf.float32, [None, ])

ReshapeToCNN = tf.reshape(source, [-1, 1, segmentLength, 1])
cnn_op1 = conv2d(ReshapeToCNN, 128, [1, 512], [1, 250], tf.nn.relu, 'same', 'cnn_L1')
SquzToLSTM = tf.squeeze(cnn_op1, [1])

lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(250, forget_bias=1.0)
with tf.variable_scope('lstm_cell_1'):
    lstm_out, _ = tf.nn.dynamic_rnn(lstm_cell_1, SquzToLSTM, dtype=tf.float32)

recover = tf.reshape(lstm_out, [-1, segmentLength])

trainables = tf.trainable_variables()
loss = tf.nn.l2_loss(recover - source)
train_L2_loss = tf.train.AdamOptimizer(0.0001).minimize(loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


batchSize = 128
is_training = False
saver = tf.train.Saver()

if is_training:
    for epoch in range(20):
        for iter in range(100):
            x = nextbatch(Trainfilepath, segmentLength, batchSize)
            train_L2_loss.run(feed_dict={source: x})

        loss_value = sess.run(loss, feed_dict={source: x})
        print('epoch: %d' % (epoch))
        print('loss: %f' % (loss_value))

    saver.save(sess, 'ckpt/model.ckpt')
else:
    saver.restore(sess, tf.train.latest_checkpoint('ckpt/'))
    print('Model restored.')

voice, filename = pickOne(Testfilepath, 'SF1', segmentLength)
output = sess.run(recover, feed_dict={source: voice})
output = output.reshape([-1])
scipy.io.wavfile.write(filename + '.wav', sampleRate, output)

