import tensorflow as tf
import numpy as np
import time
import layer

N = 500
L = 80
tstep = 100
hidNum = 1000

class speakerReg():
        def __init__(self, arch):
                self.arch = arch
                self.regnition = tf.make_template('regnition', self.regnition)


        def regnition(self, input):
                net = self.arch['regnizer']
                c = net['channel']
                k = net['kernel']
                s = net['stride']
                latentSize = self.arch['z_dim']

                x = tf.reshape(input, [-1, latentSize, 1, 1])
                for i in range(len(c)):
                        x = layer.conv2d(x, c[i], k[i], s[i], tf.nn.relu, name='cnn-L{}'.format(i))
                flat = tf.layers.flatten(x)
                y = tf.layers.dense(flat, 10, bias_initializer=tf.constant_initializer(0.1))
                return y

        def loss(self, input, label):
                y = self.regnition(input)
                loss=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=y))
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                return loss, accuracy

