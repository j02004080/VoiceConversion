import tensorflow as tf
import layer

class VAE():
    def __init__(self, arch):
        self.arch = arch
        self.encoder = tf.make_template('encoder', self.encoder)
        self.decoder = tf.make_template('decoder', self.decoder)

    def encoder(self, input):
        net = self.arch['encoder']
        c = net['channel']
        k = net['kernel']
        s = net['stride']
        unit_z = self.arch['z_dim']
        featureSize = self.arch['featureSize']

        x = tf.reshape(input, [-1, featureSize, 1, 1])
        for l in range(len(c)):
            x = layer.conv2d(x, c[l], k[l], s[l], activation = layer.prelu, name = 'encoder-L{}'.format(l))
        x = tf.layers.flatten(x)
        z_mu = tf.layers.dense(x, unit_z)
        z_var = tf.layers.dense(x, unit_z)
        return z_mu, z_var

    def decoder(self, input, speaker_id):
        net = self.arch['decoder']
        c = net['channel']
        k = net['kernel']
        s = net['stride']
        img_h, img_c = net['hc']
        featureSize = self.arch['featureSize']

        z_emb = tf.layers.dense(input, img_h*img_c, bias_initializer=tf.constant_initializer(0.1))
        speaker_emb = tf.layers.dense(speaker_id, img_h*img_c, bias_initializer=tf.constant_initializer(0.1))
        latent = z_emb + speaker_emb
        x = tf.reshape(latent, [-1, img_h, 1, img_c])
        for l in range(len(c)):
            x = layer.deconv2d(x, c[l], k[l], s[l], activation=layer.prelu, name='decoder-L{}'.format(l))
        x = tf.reshape(x, [-1, featureSize])
        return x

