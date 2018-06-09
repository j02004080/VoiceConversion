import tensorflow as tf
import numpy as np
from function.layer import (conv2d, deconv2d, prelu)

class ConvVAE_WGAN():
    def __init__(self, arch):
        self.arch = arch
        with tf.name_scope('SpeakerRepre'):
            self.speaker_emb = self.ini_emb(self.arch['speaker_dim'], self.arch['z_dim'], 'speaker_embedding')
        self.featureSize = self.arch['featureSize']
        self.latentSize = self.arch['z_dim']
        self.discriminator = tf.make_template('discriminator', self.discriminator)
        self.encoder = tf.make_template('encoder', self.encoder)
        self.generator = tf.make_template('decoder', self.generator)

    def ini_emb(self, n_speaker, z_dim, scope_name):
        with tf.variable_scope(scope_name):
            embeddings = tf.get_variable(
                name = 'y_emb',
                shape = [n_speaker, z_dim])
        return embeddings
    
    def encoder(self, x):
        x = tf.reshape(x, [-1, self.featureSize, 1, 1])
        unit = self.arch['encoder']
        c = unit['channel']
        k = unit['kernel']
        s = unit['stride']
        for i in range(len(c)):
            x = conv2d(x, c[i], k[i], s[i], prelu, name='encoder-L{}'.format(i))
        x = tf.layers.flatten(x)
        z_mu = tf.layers.dense(x, self.arch['z_dim'])
        z_var = tf.layers.dense(x, self.arch['z_dim'])
        return z_mu, z_var
    
    def generator(self, z, y):
        unit = self.arch['generator']
        c = unit['channel']
        k = unit['kernel']
        s = unit['stride']
        # y = tf.nn.embedding_lookup(self.speaker_emb, y)
        x = tf.layers.dense(z, 19*self.latentSize, activation=None, bias_initializer=tf.constant_initializer(0.1))
        h = tf.layers.dense(y, 19*self.latentSize, activation=None, bias_initializer=tf.constant_initializer(0.1))
        z_emb = x + h
        x = z_emb
        x = tf.reshape(x, [-1, 19, 1, self.latentSize])
        for i in range(len(c)):
            x = deconv2d(x, c[i], k[i], s[i], prelu, name='generator-L{}'.format(i))
        x = tf.reshape(x, [-1, self.featureSize])
        return x, h

    def discriminator(self, x):
        x = tf.reshape(x, [-1, self.featureSize, 1, 1])
        unit = self.arch['discriminator']
        c = unit['channel']
        k = unit['kernel']
        s = unit['stride']       
        for i in range(len(c)):
            x = conv2d(x, c[i], k[i], s[i], prelu, name='discriminator-L{}'.format(i))
        x = tf.layers.flatten(x)
        y = tf.layers.dense(x, 1, bias_initializer=tf.constant_initializer(0.1))
        return y

    def loss(self, ori, trans, lamb = 0.01):
        ori = tf.reshape(ori, [-1, self.featureSize])
        trans = tf.reshape(trans, [-1, self.featureSize])
        epsilon = tf.random_uniform(shape=[tf.shape(trans)[0], 1], minval=0, maxval=1)
        interpolate = epsilon*trans + (1-epsilon)*ori

        gradients = tf.gradients(self.discriminator(interpolate), [interpolate])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=1))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        Lg = -tf.reduce_mean(self.discriminator(trans))
        Ld = -tf.reduce_mean(self.discriminator(ori)) + tf.reduce_mean(self.discriminator(trans)) + lamb*gradient_penalty

        return Ld, Lg
        
        
        
    
