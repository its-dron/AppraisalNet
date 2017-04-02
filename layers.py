import tensorflow as tf
import numpy as np

def variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).

    Example inputs: Layer weights, biases, kernel, etc
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def conv2d(name, input, shape,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        strides=[1,1,1,1],
        padding='SAME',
        activation='relu'):
    '''
    2D convolution Layer with smart variable reuse.
    '''

    def conv2d_helper(input, shape, dtype, initializer, strides, padding, activation):
        kernel = tf.get_variable('weights', shape=shape, dtype=dtype,
                initializer=initializer)
        biases = tf.get_variable('biases', shape=shape[-1], dtype=dtype,
                initializer=initializer)

        variable_summaries(kernel)
        variable_summaries(biases)

        conv = tf.nn.conv2d(input, kernel, strides, padding=padding)
        biased_conv = tf.nn.bias_add(conv, biases)
        if activation is None or \
                activation.lower()=='none' or \
                activation.lower()=='linear':
                    output = biased_conv
        elif activation.lower() == 'relu':
            output = tf.nn.relu(biased_conv, name='relu')
        else:
            print("Unknown Activation Type")
        return output, kernel, biases

    with tf.variable_scope(name) as scope:
        try:
            return conv2d_helper(input, shape, dtype, initializer,
                    strides, padding, activation)
        except ValueError:
            scope.reuse_variables()
            return conv2d_helper(input, shape, dtype, initializer,
                    strides, padding, activation)

def fc(name, input, units,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        activation='relu'):
    def fc_helper(input, units, initializer, activation):
        fan_in = int(np.prod(input.get_shape()[1:]))
        weights = tf.get_variable('weights', shape=(fan_in, units), dtype=dtype, initializer=initializer)
        biases = tf.get_variable('biases', shape=(units), dtype=dtype,
                initializer=initializer)

        variable_summaries(weights)
        variable_summaries(biases)

        input_flat = tf.reshape(input, [-1, fan_in])
        pre_activate = tf.nn.bias_add(tf.matmul(input_flat, weights), biases)
        if activation is None or \
                activation.lower()=='none' or \
                activation.lower()=='linear':
                    output = pre_activate
        elif activation.lower() == 'relu':
            output = tf.nn.relu(pre_activate, name='relu')
        else:
            print("Unknown Activation Type")
        return output, weights, biases

    with tf.variable_scope(name) as scope:
        try:
            return fc_helper(input, units, initializer, activation)
        except ValueError:
            scope.reuse_variables()
            return fc_helper(input, units, initializer, activation)

