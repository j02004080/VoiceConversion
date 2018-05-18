import tensorflow as tf
import numpy as np
import time
import layer

N = 500
L = 80
tstep = 100
hidNum = 1000

class speakerReg():
        def __init__(self, arch, name):
                self.arch = arch
                self.regnition = tf.make_template('regnition', self.regnition)
                self.emb_trans = tf.make_template('emb_trans', self.emb_trans)
                with tf.name_scope(name):
                        self.speaker_emb = self.ini_emb(self.arch['speaker_dim'], name+'speaker_embedding')

        def ini_emb(self, n_speaker, scope_name):
                with tf.variable_scope(scope_name):
                        embeddings_trans = tf.get_variable(
                                name='y_emb',
                                shape=[n_speaker, N, N])
                return embeddings_trans

        def emb_trans(self, x, label):
                y = tf.argmax(label, axis=1)
                trans_mat = tf.nn.embedding_lookup(self.speaker_emb, y)
                trans = tf.matmul(x, trans_mat)
                trans = tf.reshape(trans, [-1, 500])
                return trans

        def regnition(self, input):
                x = tf.reshape(input, [-1, 1, tstep*N, 1])
                c = self.arch['channel']
                k = self.arch['kernel']
                s = self.arch['stride']
                for i in range(len(c)):
                        x = layer.conv2d(x, c[i], k[i], s[i], tf.nn.relu, 'SAME', name='cnn-L{}'.format(i))
                flat = tf.reshape(x, [-1, 100*c[1]])
                y = tf.layers.dense(flat, 10, bias_initializer=tf.constant_initializer(0.1))
                return y

        def loss(self, y, label):
                loss=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=y))
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                return loss, accuracy

