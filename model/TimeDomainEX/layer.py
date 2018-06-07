import tensorflow as tf
import numpy as np

def weight_ini(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_ini(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def prelu(x, name):
    pos = tf.nn.relu(x)
    with tf.variable_scope(name):
        alpha = tf.get_variable(name, x.get_shape()[-1], initializer=tf.constant_initializer(0.25))
    neg = alpha*(x-abs(x))*0.5
    return pos + neg

def conv2d(x, c, k, s, activation, padding, name):
    with tf.variable_scope(name):
        x = tf.layers.conv2d(x, c, k, s,
            padding = padding,
            name = name)
        x = tf.contrib.layers.layer_norm(x)
        return activation(x, name)

def deconv2d(x, c, k, s, activation, padding, name):
    with tf.variable_scope(name):
        x = tf.layers.conv2d_transpose(x, c, k, s,
            padding = padding,
            name = name)
        x = tf.contrib.layers.layer_norm(x)
        return activation(x, name)

def gatedCNN(input, arch, name):
    c = arch['channel']
    k = arch['kernel']
    s = arch['stride']
    with tf.variable_scope(name):
        A = conv2d(input, c, k, s, tf.nn.relu, 'VALID', 'reluConv')
        B = conv2d(input, c, k, s, tf.sigmoid, 'VALID', 'sigmoidConv')
        output = tf.multiply(A, B)
        return output


