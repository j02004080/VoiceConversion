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
                self.latentSize = hidNum
                self.discriminator = tf.make_template('discriminator', self.discriminator)
                self.encoder = tf.make_template('encoder', self.encoder)
                self.generator = tf.make_template('generator', self.generator)
                self.loss = tf.make_template('loss', self.loss)
                with tf.name_scope(name):
                    self.speaker_emb = self.ini_emb(self.arch['speaker_dim'], name + 'speaker_embedding')
                # self.input_dim = input_dim

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

        def gateActivation(self, x, channel, kernel, stride, label, name):

            fx = layer.conv2d(x, channel, kernel, stride, activation=None, padding='SAME', name=name+'_gate-filter')
            fx = tf.reshape(fx, [-1, channel])
            fh = tf.layers.dense(label, channel, activation=None, bias_initializer=tf.constant_initializer(0.1))
            gx = layer.conv2d(x, channel, kernel, stride, activation=None, padding='SAME', name=name+'_gate-gate')
            gx = tf.reshape(gx, [-1, channel])
            gh = tf.layers.dense(label, channel, activation=None, bias_initializer=tf.constant_initializer(0.1))
            output = tf.multiply(tf.nn.tanh(fx+fh), tf.nn.sigmoid(gx+gh))

            return output

        def encoder(self, x):
            x = tf.reshape(x, [-1, 500, 1, 1])
            unit = self.arch['encoder']
            c = unit['channel']
            k = unit['kernel']
            s = unit['stride']
            for i in range(len(c)):
                x = layer.conv2d(x, c[i], k[i], s[i], tf.nn.relu, padding='SAME', name='encoder-L{}'.format(i))
            x = tf.layers.flatten(x)
            z_mu = tf.layers.dense(x, self.arch['z_dim'])
            z_var = tf.layers.dense(x, self.arch['z_dim'])
            return z_mu, z_var

        def generator(self, z, label):
                net = self.arch['generator']
                c = net['channel']
                k = net['kernel']
                s = net['stride']

                # x = tf.reshape(x, [-1, tstep, N, 1])

                label = tf.tile(tf.expand_dims(label, 1), [1, tstep, 1])
                label = tf.reshape(label, [-1, 10])
                x = tf.layers.dense(z, 25 * self.latentSize, activation=None,
                                    bias_initializer=tf.constant_initializer(0.1))
                h = tf.layers.dense(label, 25 * self.latentSize, activation=None,
                                    bias_initializer=tf.constant_initializer(0.1))
                z_emb = x + h
                x = z_emb
                x = tf.reshape(x, [-1, 25, 1, self.latentSize])
                for i in range(len(c)):
                    x = layer.deconv2d(x, c[i], k[i], s[i], tf.nn.relu, padding='SAME', name='generator-L{}'.format(i))
                x = tf.reshape(x, [-1, N])

                return x

        def discriminator(self, input):
                x = tf.reshape(input, [-1, 1, tstep*L, 1])
                net = self.arch['discriminator']
                c = net['channel']
                k = net['kernel']
                s = net['stride']
                for i in range(len(c)):
                        x = layer.conv2d(x, c[i], k[i], s[i], tf.nn.relu, 'SAME', name='dis-L{}'.format(i))
                flat = tf.contrib.layers.flatten(x)
                y = tf.layers.dense(flat, 32, activation=tf.nn.relu, bias_initializer=tf.constant_initializer(0.1))
                y = tf.layers.dense(y, 1, bias_initializer=tf.constant_initializer(0.1))
                return y

        def loss(self, ori, trans, lamb = 10):
            Lg = tf.reduce_mean(self.discriminator(trans))
            Ld = tf.reduce_mean(self.discriminator(ori)) - tf.reduce_mean(self.discriminator(trans))

            ori = tf.reshape(ori, [tf.shape(trans)[0], -1])
            trans = tf.reshape(trans, [tf.shape(trans)[0], -1])
            epsilon = tf.random_uniform(shape=[tf.shape(trans)[0], 1], minval=0, maxval=1)
            interpolate = epsilon * trans + (1 - epsilon) * ori
            interpolate = tf.reshape(interpolate, [tf.shape(trans)[0], tstep*L])

            gradients = tf.reshape(tf.gradients(self.discriminator(interpolate), [interpolate])[0], [tf.shape(trans)[0], -1])
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            Ld += lamb * gradient_penalty
            return Ld, Lg


