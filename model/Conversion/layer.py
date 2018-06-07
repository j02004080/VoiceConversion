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

def conv2d(x, c, k, s, activation, name):
    with tf.variable_scope(name):
        x = tf.layers.conv2d(x, c, k, s,
            padding = 'same',
            name = name)
        x = tf.contrib.layers.layer_norm(x)
        return activation(x, name)

def deconv2d(x, c, k, s, activation, name):
    with tf.variable_scope(name):
        x = tf.layers.conv2d_transpose(x, c, k, s,
            padding = 'same',
            name = name)
        x = tf.contrib.layers.layer_norm(x)
        return activation(x, name)

def gatedCNN(input, arch, name):
    c = arch['channel']
    k = arch['kernel']
    s = arch['stride']
    with tf.variable_scope(name):
        A = tf.layers.conv2d(input, c, k, s, activation=tf.nn.relu, padding = 'valid', name = 'reluConv')
        B = tf.layers.conv2d(input, c, k, s, activation=tf.nn.sigmoid, padding = 'valid', name = 'sigmoidConv')
        output = tf.multiply(A, B)
        return output

def IN_conv(x):
    Xtihw = tf.transpose(x, [0, 3, 1, 2])
    mu = tf.reduce_mean(tf.reduce_mean(Xtihw, 2), 2)
    mu_shape = mu.get_shape().as_list()
    x_shape = x.get_shape().as_list()
    mu = tf.tile(tf.reshape(mu, [-1, 1, 1, mu_shape[1]]), [1, x_shape[1], x_shape[2], 1])
    
    Xtihw_square = tf.transpose(tf.square(x-mu), [0, 3, 1, 2])
    var = tf.reduce_mean(tf.reduce_mean(Xtihw_square, 2), 2)
    epsilon = tf.random_normal(tf.shape(var))
    var = tf.tile(tf.reshape(var, [-1, 1, 1, mu_shape[1]]), [1, x_shape[1], x_shape[2], 1])
    epsilon = tf.tile(tf.reshape(epsilon, [-1, 1, 1, mu_shape[1]]), [1, x_shape[1], x_shape[2], 1])
    y = tf.divide((x-mu), tf.sqrt(var+epsilon))
    return y

def _phase_shift(I, r):
    # Helper function with main phase shift operation
    bsize, a, b, c = I.get_shape().as_list()
    X = tf.reshape(I, (-1, a, b, r[0], r[1]))
    X = tf.transpose(X, (0, 1, 2, 3, 4))  # bsize, a, b, 1, 1
    X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x, axis = 1) for x in X], 2)  # bsize, b, a*r, r
    X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x, axis = 1) for x in X], 2)  # bsize, a*r, b*r
    return tf.reshape(X, (-1, a*r[0], b*r[1], 1))

def PS(X, r, color=False):
  # Main OP that you can arbitrarily use in you tensorflow code
  channel = X.get_shape().as_list()[3]
  if color:
    Xc = tf.split(X, (channel//(r[0]*r[1])), 3)
    X = tf.concat([_phase_shift(x, r) for x in Xc], 3)
  else:
    X = _phase_shift(X, r)
  return X

