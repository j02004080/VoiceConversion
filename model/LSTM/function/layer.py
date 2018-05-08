import tensorflow as tf
import numpy as np


def prelu(x, name):
    pos = tf.nn.relu(x)
    with tf.variable_scope(name):
        alpha = tf.get_variable(name, x.get_shape()[-1], initializer=tf.constant_initializer(0.25))
    neg = alpha*(x-abs(x))*0.5
    return pos + neg

def conv2d(x, c, k, s, activation, name):
    with tf.variable_scope(name):
        x = tf.layers.conv2d(x, c, k, s,
            padding = 'same',
            name = name)
        return activation(x, name)

def deconv2d(x, c, k, s, activation, name):
    with tf.variable_scope(name):
        x = tf.layers.conv2d_transpose(x, c, k, s,
            padding = 'same',
            name = name)
        return activation(x, name)
