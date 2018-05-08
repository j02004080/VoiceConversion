import tensorflow as tf
import numpy as np
from function.layer import (conv2d, deconv2d, prelu, PS)

class ConvWGAN():
    def __init__(self, arch):
        self.arch = arch

        self.discriminator = tf.make_template('discriminator', self.discriminator)
        self.generator = tf.make_template('generator', self.generator)

    def generator(self, x):

        net = self.arch['generator']

        ## DownSample
        downS = net['downSample']
        c = downS['channel']
        k = downS['kernel']
        s = downS['stride']

        for i in range(len(c)):
            x = conv2d(x, c[i], k[i], s[i], prelu, name='downSample-L{}'.format(i))

        ## Residual
        resid = net['residual']
        c = resid['channel']
        k = resid['kernel']
        s = resid['stride']
        for i in range(len(c)//2):
            y = x
            x = conv2d(x, c[2*i], k[2*i], s[2*i], prelu, name='resid_conv1-L{}'.format(i))
            x = conv2d(x, c[2*i+1], k[2*i+1], s[2*i+1], prelu, name='resid_conv2-L{}'.format(i))
            x = y + x

        ## UpSample
        upS = net['UpSample']
        c = upS['channel']
        k = upS['kernel']
        s = upS['stride']
        r = upS['ratio']
        for i in range(len(c)):
            x = conv2d(x, c[i], k[i], s[i], prelu, name='UpSample-L{}'.format(i))
            if(i<len(r)):
                x = PS(x, r[i], color=True)

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

        
    
