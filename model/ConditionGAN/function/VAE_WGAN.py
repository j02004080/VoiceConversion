import tensorflow as tf
import numpy as np
from function.layer import (conv2d, deconv2d, prelu)

class ConvVAE_WGAN():
    def __init__(self, arch):
        self.arch = arch
        with tf.name_scope('SpeakerRepre'):
            self.speaker_emb = self.ini_emb(self.arch['speaker_dim'], self.arch['z_dim'], 'speaker_embedding')
        
        self.discriminator = tf.make_template('discriminator', self.discriminator)
        self.encoder = tf.make_template('encoder', self.encoder)
        self.generator = tf.make_template('decoder', self.generator)

    def ini_emb(self, n_speaker, z_dim, scope_name):
        with tf.variable_scope(scope_name):
            embeddings = tf.get_variable(
                name = 'y_emb',
                shape = [n_speaker, z_dim])
        return embeddings

    def merge(self, var_list, op_unit, l2_reg=1e-6):
        x = 0.
        with tf.contrib.slim.arg_scope([tf.contrib.slim.fully_connected],
                                       num_outputs = op_unit,
                                       weights_regularizer = tf.contrib.slim.l2_regularizer(l2_reg),
                                       normalizer_fn = None,
                                       activation_fn = None):
            for var in var_list:
                x = x + tf.contrib.slim.fully_connected(var)
        return x
    
    def encoder(self, x):
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
        y = tf.nn.embedding_lookup(self.speaker_emb, y)
        z_emb = self.merge([z, y], 19*64)
        x = z_emb
        x = tf.reshape(x, [-1, 19, 1, 64])
        for i in range(len(c)):
            x = deconv2d(x, c[i], k[i], s[i], prelu, name='generator-L{}'.format(i))
        return x

    def discriminator(self, x):
        unit = self.arch['discriminator']
        c = unit['channel']
        k = unit['kernel']
        s = unit['stride']       
        for i in range(len(c)):
            x = conv2d(x, c[i], k[i], s[i], prelu, name='discriminator-L{}'.format(i))
        x = tf.layers.flatten(x)
        y = tf.layers.dense(x, 1)
        return y

    def loss(self, x, y, lamb = 0.01):

        epsilon = tf.random_normal(tf.shape(1))
        interpolate = epsilon*x + (1-epsilon)*y
        gradients = tf.gradients(self.discriminator(interpolate), [interpolate])[0]
        
        Lg = -tf.reduce_mean(self.discriminator(x))
        Ld = -tf.reduce_mean(self.discriminator(y)) + tf.reduce_mean(self.discriminator(x)) + lamb*tf.reduce_mean(tf.pow(tf.norm(gradients) - 1, 2))
        loss = dict()
        loss['Ld'] = Ld
        loss['Lg'] = Lg
        return loss
        
        
        
    
