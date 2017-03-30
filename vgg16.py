# Based on code writted by Davi Frossard, 2016

import tensorflow as tf
import numpy as np
import layers

FLAGS = tf.app.flags.FLAGS

class vgg16:
    '''
    VGG16 Model with ImageNet pretrained weight loader method

    Weights can be downloaded from:
    https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz
    '''

    def __init__(self, imgs, weights_path=None, sess=None):
        '''
        Sets up network enough to do a forward pass.
        '''

        """ init the model with hyper-parameters etc """

        # List used for loading weights from vgg16.npz (if necessary)
        self.parameters = []

        ########
        # Misc #
        ########
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.train.exponential_decay(
                FLAGS.initial_lr, # Base learning rate
                self.global_step * FLAGS.batch_size, # Current idx into the dataset
                FLAGS.decay_step, # Decay step (when to decrease LR)
                FLAGS.decay_rate, # Decay Rate
                staircase=False)

        ####################
        # I/O placeholders #
        ####################
        self.x = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.y = tf.placeholder(tf.float32, [None, 10])

        ###############
        # Main Layers #
        ###############
        self._convlayers()
        self._fc_layers()
        self.predictions = tf.nn.softmax(self.fc3l)

        ###################################################
        # Load pre-trained ImageNet weights if applicable #
        ###################################################
        if weights_path is not None and sess is not None:
            self._load_weights(weights_path, sess)

    def inference(self):
        '''
        Returns the output of a forward pass of the network (Tensor).
        '''
        return self.predictions

    def loss(self):
        '''
        Returns the loss output (Tensor).
        '''
        self.loss = tf.reduce_mean('blahblahblah')

        # Add a scalar summary for TensorBoard
        # Note: tf.scaler_summary is deprecated
        tf.summary.scalar('loss', self.loss)
        return self.loss

    def optimize(self):
        '''
        Returns the Training Operation (op).
        '''
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss, name='optimizer')
        return self.train_op

    # INCOMPLETE
    def evaluation(self, labels):
        '''
        Returns the count of correct classifications (Tensor).
        '''
        correct = tf.nn.in_top_k(self.predictions, labels, 1)
        n_correct = tf.reduce_sum(tf.cast(correct, tf.int32))

        accuracy = tf.reduce_mean(tf.cast(correct, tf.int32))
        return n_correct

    #####################
    # Private Functions #
    #####################
    def _convlayers(self):
        '''
        All conv and pooling layers of VGG16
        '''
        # zero-mean input; resizing has to be done beforehand for uniform tensor shape
        with tf.variable_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.x-mean

        # conv1_1
        self.conv1_1, kernel, biases = layers.conv(images, 3, 3, 3, 64, 'conv1_1')
        self.parameters += [kernel, biases]
        # conv1_2
        self.conv1_2, kernel, biases = layers.conv(self.conv1_1, 3, 3, 64, 64, 'conv1_2')
        self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name='pool1')

        # conv2_1
        self.conv2_1, kernel, biases = layers.conv(self.conv1_2, 3, 3, 64, 128, 'conv2_1')
        self.parameters += [kernel, biases]
        # conv2_2
        self.conv2_2, kernel, biases = layers.conv(self.conv2_1, 3, 3, 128, 128, 'conv2_2')
        self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name='pool2')

        # conv3_1
        self.conv3_1, kernel, biases = layers.conv(self.conv2_2, 3, 3, 128, 256, 'conv3_1')
        self.parameters += [kernel, biases]
        # conv3_2
        self.conv3_2, kernel, biases = layers.conv(self.conv3_1, 3, 3, 256, 256, 'conv3_2')
        self.parameters += [kernel, biases]
        # conv3_3
        self.conv3_3, kernel, biases = layers.conv(self.conv3_2, 3, 3, 256, 256, 'conv3_3')
        self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name='pool3')

        # conv4_1
        self.conv4_1, kernel, biases = layers.conv(self.conv3_3, 3, 3, 256, 512 , 'conv4_1')
        self.parameters += [kernel, biases]
        # conv4_2
        self.conv4_2, kernel, biases = layers.conv(self.conv4_1, 3, 3, 512, 512, 'conv4_2')
        self.parameters += [kernel, biases]
        # conv4_3
        self.conv4_3, kernel, biases = layers.conv(self.conv4_2, 3, 3, 512, 512, 'conv4_3')
        self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name='pool4')

        # conv5_1
        self.conv5_1, kernel, biases = layers.conv(self.conv4_3, 3, 3, 256, 512 , 'conv5_1')
        self.parameters += [kernel, biases]
        # conv5_2
        self.conv5_2, kernel, biases = layers.conv(self.conv5_1, 3, 3, 512, 512, 'conv5_2')
        self.parameters += [kernel, biases]
        # conv5_3
        self.conv5_3, kernel, biases = layers.conv(self.conv5_2, 3, 3, 512, 512, 'conv5_3')
        self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name='pool5')

    def _fc_layers(self):
        '''
        All FC layers of VGG16 (+custom layers)
        '''
        param_count = int(np.prod(self.pool5.get_shape()[1:]))
        pool5_flat = tf.reshape(self.pool5, [-1, param_count])

        # fc1
        self.fc1, weights, biases = layers.fully_connected(pool5_flat, 4096, 'fc1')
        self.parameters += [weights, biases]

        # fc2
        self.fc2, weights, biases = layers.fully_connected(self.fc1, 4096, 'fc2')
        self.parameters += [weights, biases]

        # fc3
        self.fc3l, weights, biases = layers.fully_connected(self.fc2, 1000, 'fc3')
        self.parameters += [weights, biases]

    def load_npz_weights(self, weight_file, sess):
        '''
        Load Pretrained VGG16 weights from .npz file
        (weights converted from Caffe)

        To only be used when no TensorFlow Snapshot is avaialable.
        '''
        print "Loading Imagenet Weights."

        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print i, k, np.shape(weights[k])
            try:
                sess.run(self.parameters[i].assign(weights[k]))
            except:
                print "%s layer not found." % k
                pass

