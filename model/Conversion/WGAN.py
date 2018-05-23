import tensorflow as tf
import numpy as np
import time
import layer

N = 500
L = 80
tstep = 100
hidNum = 64
batchSize = 128

class WGAN():
        def __init__(self, arch, name):
                self.arch = arch
                self.discriminator = tf.make_template('discriminator', self.discriminator)
                self.generator = tf.make_template('generator', self.generator)
                self.loss = tf.make_template('loss', self.loss)
                with tf.name_scope(name):
                    self.speaker_emb = self.ini_emb(self.arch['speaker_dim'], name + 'speaker_embedding')

        def ini_emb(self, n_speaker, scope_name):
                with tf.variable_scope(scope_name):
                        embeddings_trans = tf.get_variable(
                                name='y_emb',
                                shape=[n_speaker, hidNum, hidNum])
                return embeddings_trans

        def lrelu(self, x, leak=0.2, name="lrelu"):
            with tf.variable_scope(name):
                f1 = 0.5 * (1 + leak)
                f2 = 0.5 * (1 - leak)
                return f1 * x + f2 * abs(x)

        def generator(self, x, label):
                net = self.arch['gen']
                c = net['channel']
                k = net['kernel']
                s = net['stride']
                h = net['hidNum']
                y = tf.argmax(label, axis=1)
                y = tf.nn.embedding_lookup(self.speaker_emb, y)
                x = tf.layers.dense(x, h[0], activation=self.lrelu, bias_initializer=tf.constant_initializer(0.1))
                x = tf.layers.dense(x, h[1], activation=self.lrelu, bias_initializer=tf.constant_initializer(0.1))
                x = tf.reshape(x, [-1, tstep, hidNum])
                x = tf.matmul(x, y)
                x = tf.reshape(x, [-1, hidNum])
                x = tf.layers.dense(x, h[2], activation=self.lrelu, bias_initializer=tf.constant_initializer(0.1))
                output = tf.layers.dense(x, h[3], activation=tf.nn.tanh, bias_initializer=tf.constant_initializer(0.1))
                return output

        def discriminator(self, input):
                x = tf.reshape(input, [-1, 1, tstep*L, 1])
                net = self.arch['dis']
                c = net['channel']
                k = net['kernel']
                s = net['stride']
                for i in range(len(c)):
                        x = layer.conv2d(x, c[i], k[i], s[i], tf.nn.relu, 'SAME', name='dis-L{}'.format(i))
                flat = tf.contrib.layers.flatten(x)
                y = tf.layers.dense(flat, 1024, activation=tf.nn.relu, bias_initializer=tf.constant_initializer(0.1))
                y = tf.layers.dense(y, 10, bias_initializer=tf.constant_initializer(0.1))
                return y

        def loss(self, ori, trans, lamb = 10):
            Lg = tf.reduce_mean(self.discriminator(trans))
            Ld = tf.reduce_mean(self.discriminator(ori)) - tf.reduce_mean(self.discriminator(trans))

            ori = tf.reshape(ori, [batchSize, -1])
            trans = tf.reshape(trans, [batchSize, -1])
            epsilon = tf.random_uniform(shape=[batchSize, 1], minval=0, maxval=1)
            interpolate = epsilon * trans + (1 - epsilon) * ori
            interpolate = tf.reshape(interpolate, [batchSize, tstep*L])

            gradients = tf.reshape(tf.gradients(self.discriminator(interpolate), [interpolate])[0], [batchSize, -1])
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            Ld += lamb * gradient_penalty
            return Ld, Lg


