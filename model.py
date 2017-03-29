import tensorflow as tf

class appraisal_vgg16(object):
    '''
    Modeled after the practices layed out in:
    https://github.com/aicodes/tf-bestpractice
    '''

    def __init__(self, n_clases=1000):
        """ init the model with hyper-parameters etc """

        ##############
        # I/O Layers #
        ##############
        self.x = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.y = tf.placeholder(tf.float32, shape=[])


        ################
        # INNER LAYERS #
        ################
        pass

    def load_imagenet_weights(self, weight_file, sess):
        pass
    def inference(self, x):
        """ This is the forward calculation from x to y """
        return some_op(x, name="inference")

    def loss(self, batch_x, batch_y=None):
        y_predict = self.inference(batch_x)
        self.loss = tf.loss_function(y, y_predict, name="loss") # supervised
        # loss = tf.loss_function(x, y_predicted) # unsupervised

    def optimize(self, batch_x, batch_y):
        return tf.train.optimizer.minimize(self.loss, name="optimizer")
