# Based on code writted by Davi Frossard, 2016

import tensorflow as tf
import layers

FLAGS = tf.app.flags.FLAGS

class shallow:
    '''
    Weights can be downloaded from:
    https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz
    '''

    def __init__(self, x, y):
        '''
        Sets up network enough to do a forward pass.
        '''

        ########
        # Misc #
        ########
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = FLAGS.initial_lr

        ####################
        # I/O placeholders #
        ####################
        self.x = x
        self.y = tf.to_int32(y)
        #self.am_training = tf.placeholder(dtype=bool, shape=())
        #self.x = tf.cond(self.am_training, lambda:train_x, lambda:validate_x)
        #self.y = tf.to_int32(tf.cond(self.am_training, lambda:validate_x, lambda:validate_y))

        ###############
        # Main Layers #
        ###############
        self._convlayers()
        self._fc_layers()

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
                logits=self.fc2, labels=self.y, name='cross_entropy')
        self.loss = tf.reduce_mean(cross_entropy, name='loss')
        return self.loss

    def optimize(self):
        '''
        Returns the Training Operation (op).
        '''
        # SGD
        #optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        #self.train_op = optimizer.minimize(self.loss, name='optimizer')

        # Adam
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss, name='optimizer')
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
            images = tf.reduce_mean(self.x, axis=3, keep_dims=True) / 255.0
            #images = self.x[:,:,:,0] / 255.0

        # conv1_1
        self.conv1_1, weights, biases = layers.conv2d(name='conv1_1',
                input=images,
                shape=(5,5,1,32),
                padding='SAME',
                strides = [1,1,1,1],
                activation='relu')

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_1,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name='pool1')

        # conv2_1
        self.conv2_1, weights, biases = layers.conv2d(name='conv2_1',
                input=self.pool1,
                shape=(5,5,32,64),
                padding='SAME',
                strides = [1,1,1,1],
                activation='relu')

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_1,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name='pool2')

    def _fc_layers(self):
        '''
        All FC layers of VGG16 (+custom layers)
        '''
        self.pool2_flat = tf.reshape(self.pool2,[-1, 7*7*64])

        # fc1
        self.fc1, weights, biases = layers.fc(name='fc1',
                input=self.pool2_flat,
                units=1024,
                activation='relu')

        # fc2
        self.fc2, weights, biases = layers.fc(name='fc2',
                input=self.fc1,
                units=10,
                activation='linear')

        # Softmax
        self.predictions = tf.nn.softmax(self.fc2)

    def load_npz_weights(self, weight_file, sess):
        pass
