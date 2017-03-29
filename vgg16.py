# Based on code writted by Davi Frossard, 2016

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
#from imagenet_classes import class_names

class vgg16:
    def __init__(self, imgs, weights_path=None, sess=None):
        '''
        Sets up network enough to do a forward pass.
        '''

        """ init the model with hyper-parameters etc """
        # List used for loading weights from vgg16.npz
        self.parameters = []

        # Construct I/O placeholders
        self.x = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.y = tf.placeholder(tf.float32, [None, 10])

        # Declare rest of layers
        self._convlayers()
        self._fc_layers()
        self.predictions = tf.nn.softmax(self.fc3l)

        # Load pre-trained ImageNet weights if applicable
        if weights_path is not None and sess is not None:
            self._load_weights(weights_path, sess)

        return self.predictions

    def loss(self, batch_x, batch_y=None):
        y_predict = self.inference(batch_x)
        self.loss = tf.loss_function(y, y_predict, name="loss") # supervised

    def optimize(self):
        return tf.train.optimizer.minimize(self.loss, name="optimizer")

    def _convlayers(self):

        # zero-mean input; resizing has to be done beforehand for uniform tensor shape
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.x-mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.get_variable('weights', shape=(3,3,3,64), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable('biases', shape=(64), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.get_variable('weights', shape=(3,3,64,64), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable('biases', shape=(64), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.get_variable('weights', shape=(3,3,64,128), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable('biases', shape=(128), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.get_variable('weights', shape=(3,3,128,128), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable('biases', shape=(128), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.get_variable('weights', shape=(3,3,128,256), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable('biases', shape=(256), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.get_variable('weights', shape=(3,3,256,256), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable('biases', shape=(256), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.get_variable('weights', shape=(3,3,256,256), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable('biases', shape=(256), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.get_variable('weights', shape=(3,3,256,512), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable('biases', shape=(512), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.get_variable('weights', shape=(3,3,512,512), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable('biases', shape=(512), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.get_variable('weights', shape=(3,3,512,512), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable('biases', shape=(512), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.get_variable('weights', shape=(3,3,512,512), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable('biases', shape=(512), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.get_variable('weights', shape=(3,3,512,512), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable('biases', shape=(512), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.get_variable('weights', shape=(3,3,512,512), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable('biases', shape=(512), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name='pool4')

    def _fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            fan_in = int(np.prod(self.pool5.get_shape()[1:]))
            weights = tf.get_variable('weights', shape=(fan_in, 4096), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable('biases', shape=(4096), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            pool5_flat = tf.reshape(self.pool5, [-1, fan_in])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, weights), biases)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [weights, biases]

        # fc2
        with tf.name_scope('fc2') as scope:
            weights = tf.get_variable('weights', shape=(4096, 4096), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable('biases', shape=(4096), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, weights), biases)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [weights, biases]

        # fc3
        with tf.name_scope('fc3') as scope:
            weights = tf.get_variable('weights', shape=(4096, 1000), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable('biases', shape=(1000), dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, weights), biases)
            self.parameters += [weights, biases]

    def _load_npz_weights(self, weight_file, sess):
        '''
        Load Pretrained VGG16 weights from .npz file
        (weights converted from Caffe)

        To only be used when no TensorFlow Snapshot is avaialable.
        '''
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print i, k, np.shape(weights[k])
            sess.run(self.parameters[i].assign(weights[k]))

