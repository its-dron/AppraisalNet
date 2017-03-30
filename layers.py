import tensorflow as tf
import numpy as np

'''
Helper class to make common layers and summaries.

Largely copied from https://www.tensorflow.org/get_started/summaries_and_tensorboard
'''

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def conv(bottom, k_w, k_h, n_in, n_out, name, add_summary=True):
    '''
    Create a conv2d layer consisting of a conv2d, bias, and relu.

    bottom - input tensor
    n_in - number of input channels
    n_out - number of output channels
    k_w - kernel width
    k_h - kernel height
    name - name of the layer
    add_summary - Whether to add summaries to layer variables.
    '''
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable('weights', shape=(k_h, k_w, n_in, n_out), dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases', shape=(n_out), dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(bottom, kernel, [1, 1, 1, 1], padding='SAME')
        preactivate = tf.nn.bias_add(conv, biases)
        out = tf.nn.relu(preactivate, name=scope)

        if add_summary:
            variable_summaries(kernel)
            variable_summaries(biases)
            tf.summary.histogram("activations", out)

    return out, kernel, biases

def fully_connected(bottom, fan_out, name):
    fan_in = num_tensor_params(bottom)
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=(fan_in, fan_out), dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases', shape=(fan_out), dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
        preactivate = tf.nn.bias_add(tf.matmul(bottom, weights), biases)
        fc = tf.nn.relu(preactivate)

    return fc, weights, biases

def num_tensor_params(tensor, ignore_batch=True):
    if ignore_batch:
        shape = tensor.get_shape()[1:]
    else:
        shape = tensor.get_shape()
    return int(np.prod(shape))
