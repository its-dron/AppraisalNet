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

    def __init__(self, x, y, phase):
        '''
        Sets up network enough to do a forward pass.
        '''

        """ init the model with hyper-parameters etc """

        # List used for loading weights from vgg16.npz (if necessary)
        self.parameters = []
        self.WEIGHT_FILE = '/appraisal_net/pretrained_weights/vgg16_weights.npz'
        self.CONV_ACTIVATION = 'relu'
        self.FC_ACTIVATION   = 'relu'

        ########
        # Misc #
        ########
        self.global_step = tf.get_variable('global_step', dtype=tf.int32, trainable=False,
                        initializer=0)
        self.learning_rate = FLAGS.initial_lr
        self.IM_SHAPE = [224, 224, 3]

        ####################
        # I/O placeholders #
        ####################
        self.x = x
        self.x.set_shape([None]+self.IM_SHAPE)
        self.y = tf.to_int32(y)

        ###############
        # Main Layers #
        ###############
        with tf.variable_scope('conv_layers'):
            self._convlayers()
        with tf.variable_scope('fc_layers'):
            self._fc_layers()

        ######################
        # Define Collections #
        ######################
        self.conv_trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                "conv_layers")
        self.fc_trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                "fc_layers")

    def get_global_step(self):
        return self.global_step

    def inference(self):
        '''
        Returns the output of a forward pass of the network (Tensor).
        '''
        return self.predictions

    def loss(self):
        '''
        Returns the loss output (Tensor).
        '''
        # Sparse takes in index rather than one_hot encoding
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.fc3, labels=self.y, name='cross_entropy')
        self.loss = tf.reduce_mean(cross_entropy, name='loss')
        return self.loss, self.y

    def optimize(self):
        '''
        Returns the Training Operation (op).
        '''
        # Adam
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # Apply Gradient Clipping
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5.0)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars),
                global_step=self.global_step)
        return self.train_op

    def evaluate(self):
        '''
        Returns the count of correct classifications (Tensor).
        '''
        # Bool Tensor where 1 is correct and 0 is incorrect
        correct = tf.nn.in_top_k(self.predictions, self.y, 1)
        # Average them to get accuracy.  Must cast to a float32
        self.accuracy = tf.reduce_mean(tf.to_float(correct))
        return self.accuracy

    #####################
    # Private Functions #
    #####################
    def _convlayers(self):
        '''
        All conv and pooling layers of VGG16
        '''
        # zero-mean input; resizing has to be done beforehand for uniform tensor shape
        with tf.variable_scope('preprocess'):
            mean = tf.constant([123.68, 116.779, 103.939],
                    dtype=tf.float32,
                    shape=[1, 1, 1, 3],
                    name='img_mean')
            images = self.x*255.0 - mean

        # conv1_1
        self.conv1_1, weights, biases = layers.conv2d(name='conv1_1',
                input=images,
                shape=(3,3,3,64),
                padding='SAME',
                strides = [1,1,1,1],
                activation=self.CONV_ACTIVATION)
        self.parameters += [weights, biases]

        layers.conv_visual_summary(weights, name='conv_1_1')

        # conv1_2
        self.conv1_2, weights, biases = layers.conv2d(name='conv1_2',
                input=self.conv1_1,
                shape=(3,3,64,64),
                padding='SAME',
                strides = [1,1,1,1],
                activation=self.CONV_ACTIVATION)
        self.parameters += [weights, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name='pool1')

        # conv2_1
        self.conv2_1, weights, biases = layers.conv2d(name='conv2_1',
                input=self.pool1,
                shape=(3,3,64,128),
                padding='SAME',
                strides = [1,1,1,1],
                activation=self.CONV_ACTIVATION)
        self.parameters += [weights, biases]

        # conv2_2
        self.conv2_2, weights, biases = layers.conv2d(name='conv2_2',
                input=self.conv2_1,
                shape=(3,3,128,128),
                padding='SAME',
                strides = [1,1,1,1],
                activation=self.CONV_ACTIVATION)
        self.parameters += [weights, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name='pool2')

        # conv3_1
        self.conv3_1, weights, biases = layers.conv2d(name='conv3_1',
                input=self.pool2,
                shape=(3,3,128,256),
                padding='SAME',
                strides = [1,1,1,1],
                activation=self.CONV_ACTIVATION)
        self.parameters += [weights, biases]

        # conv3_2
        self.conv3_2, weights, biases = layers.conv2d(name='conv3_2',
                input=self.conv3_1,
                shape=(3,3,256,256),
                padding='SAME',
                strides = [1,1,1,1],
                activation=self.CONV_ACTIVATION)
        self.parameters += [weights, biases]

        # conv3_3
        self.conv3_3, weights, biases = layers.conv2d(name='conv3_3',
                input=self.conv3_2,
                shape=(3,3,256,256),
                padding='SAME',
                strides = [1,1,1,1],
                activation=self.CONV_ACTIVATION)
        self.parameters += [weights, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name='pool3')

        # conv4_1
        self.conv4_1, weights, biases = layers.conv2d(name='conv4_1',
                input=self.pool3,
                shape=(3,3,256,512),
                padding='SAME',
                strides = [1,1,1,1],
                activation=self.CONV_ACTIVATION)
        self.parameters += [weights, biases]

        # conv4_2
        self.conv4_2, weights, biases = layers.conv2d(name='conv4_2',
                input=self.conv4_1,
                shape=(3,3,512,512),
                padding='SAME',
                strides = [1,1,1,1],
                activation=self.CONV_ACTIVATION)
        self.parameters += [weights, biases]

        # conv4_3
        self.conv4_3, weights, biases = layers.conv2d(name='conv4_3',
                input=self.conv4_2,
                shape=(3,3,512,512),
                padding='SAME',
                strides = [1,1,1,1],
                activation=self.CONV_ACTIVATION)
        self.parameters += [weights, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name='pool4')

        # conv5_1
        self.conv5_1, weights, biases = layers.conv2d(name='conv5_1',
                input=self.pool4,
                shape=(3,3,512,512),
                padding='SAME',
                strides = [1,1,1,1],
                activation=self.CONV_ACTIVATION)
        self.parameters += [weights, biases]

        # conv5_2
        self.conv5_2, weights, biases = layers.conv2d(name='conv5_2',
                input=self.conv5_1,
                shape=(3,3,512,512),
                padding='SAME',
                strides = [1,1,1,1],
                activation=self.CONV_ACTIVATION)
        self.parameters += [weights, biases]

        # conv5_3
        self.conv5_3, weights, biases = layers.conv2d(name='conv5_3',
                input=self.conv5_2,
                shape=(3,3,512,512),
                padding='SAME',
                strides = [1,1,1,1],
                activation=self.CONV_ACTIVATION)
        self.parameters += [weights, biases]

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
        # fc1
        self.fc1, weights, biases = layers.fc(name='fc1',
                input=self.pool5,
                units=4096,
                activation=self.FC_ACTIVATION)
        self.parameters += [weights, biases]

        # fc2
        self.fc2, weights, biases = layers.fc(name='fc2',
                input=self.fc1,
                units=4096,
                activation=self.FC_ACTIVATION)
        self.parameters += [weights, biases]

        # fc3
        self.fc3, weights, biases = layers.fc(name='fc3',
                input=self.fc2,
                units=FLAGS.num_classes,
                activation='linear')

        # Softmax
        self.predictions = tf.nn.softmax(self.fc3)

    def load_pretrained_weights(self, sess):
        '''
        Load Pretrained VGG16 weights from .npz file
        (weights converted from Caffe)

        To only be used when no TensorFlow Snapshot is avaialable.

        Assumes layers are properly added to self.parameters.
        '''
        print "Loading Imagenet Weights."

        weights = np.load(self.WEIGHT_FILE)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print i, k, np.shape(weights[k])
            try:
                sess.run(self.parameters[i].assign(weights[k]))
            except:
                print "%s layer not found." % k
                pass

