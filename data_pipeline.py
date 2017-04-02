import tensorflow as tf
import numpy as np
import csv
import os
import random
import pdb
import image_utils

FLAGS = tf.app.flags.FLAGS

class DataPipeline:
    '''
    Setup Queue and data preprocessor
    '''
    def __init__(self):
        self.IM_SHAPE = [224, 224, 3]
        self._setup_data_pipeline()

    def encode_labels(self, labels):
        '''
        Do preprocessing on labels (such as one-hot encoding)
        Input: list of labels (each element is a string)
        Returns: list of formatted labels
        '''
        numerical = np.asarray( [float(x) for x in labels] )

        # Should never have a negative price
        numerical[numerical<0] = 0

        # Log-space grouping of prices into classes
        # the minus 1 is because we also consider all below and
        # all above as bins.
        # edges between 1 and 1000
        edges = np.logspace(0,3, num=FLAGS.num_classes-1)
        edges = np.insert(edges, 0, 0) # Minimum price
        edges = np.append(edges, 10000000) # arbitrarily large

        # Bin the pricing data
        # Warning, np.digitize is 1-indexed.  E.g. if something has the value one,
        # the original price was between edges[0] and edges[1]
        # Note: The "-1" makes digitized 0-indexed.
        digitized = np.digitize(numerical, bins=edges) - 1

        # One-hot encode
        #one_hot_labels = tf.one_hot(indices=digitized,
        #                            depth=FLAGS.num_classes,
        #                            on_value=1.0,
        #                            off_value=0.0,
        #                            axis=-1)
        return digitized

    def read_csv(self, csv_filename):
        '''
        Reads CSV file with the following format:
        Col_0           Col_1
        image_name,     price

        Returns 2 lists:
        - A list is list of image paths.
        - A list of prices
        '''
        # Get image names
        with open(csv_filename, 'rb') as f:
            im_names = [str(row['id']) for row in csv.DictReader(f)]
            im_paths = [os.path.join(FLAGS.image_dir, x + '.jpg')
                        for x in im_names]
        # Get prices
        with open(csv_filename, 'rb') as f:
            prices = [row['price'] for row in csv.DictReader(f)]

        return im_paths, self.encode_labels(prices)

    def augment_image(self, image):
        '''
        Apply data augmentations to image (like flip L/R)
        '''
        image = image_utils.random_crop_and_resize_proper(image, self.IM_SHAPE[0:2])
        image = image_utils.random_color_augmentation(image)
        image = tf.image.random_flip_left_right(image)
        return image

    def get_data_from_queue(self, input_queue):
        '''
        Reads image from disk
        Consumes a single filename and label.
        Resizes image as necessary.
        Returns an image tensor and a label.
        '''
        try:
            file_content = tf.read_file(input_queue[0])
            label = input_queue[1]
            # Get Tensor (image) of type uint8
            image = tf.image.decode_jpeg(file_content, channels=3)
            # Convert to [0,1]
            image = tf.to_float(image) / 255.0
            return image, label
        except:
            return None, None

    def train_batch_ops(self):
        return self.train_image_batch, self.train_label_batch

    def validate_batch_ops(self):
        return self.validate_image_batch, self.validate_label_batch

    def test_batch_ops(self):
        return self.test_image_batch, self.test_label_batch

    def _setup_data_pipeline(self):
        '''
        Partitions data and sets up data queues
        Based off of code written:
        http://ischlag.github.io/2016/06/19/tensorflow-input-pipeline-example/
        '''
        #################
        # Read CSV File #
        #################
        # Read in labels and image filenames
        im_paths, prices = self.read_csv(FLAGS.input_data)
        # convert into tensors op, dtype is inferred from input data
        all_images = tf.convert_to_tensor(im_paths)
        all_labels = tf.convert_to_tensor(prices)

        #####################
        # Partition Dataset #
        #####################
        # Generate assignment list (partition list)
        n_exemplars = len(im_paths)
        self.validate_set_size = int(FLAGS.validate_percentage * n_exemplars)
        self.test_set_size = int(FLAGS.test_percentage * n_exemplars)

        # TODO: stratified partition
        partitions = [0]*n_exemplars
        partitions[:self.validate_set_size] = [1] * self.validate_set_size
        partitions[self.validate_set_size:(self.validate_set_size + self.test_set_size)] \
                = [2] * self.test_set_size
        random.shuffle(partitions)

        # Actually partition the data
        train_images, validate_images, test_images = \
                tf.dynamic_partition(all_images, partitions, 3, name='image_partition')
        train_labels, validate_labels, test_labels = \
                tf.dynamic_partition(all_labels, partitions, 3, name='label_partition')

        #################
        # Create Queues #
        #################
        train_input_queue = tf.train.slice_input_producer(
                [train_images, train_labels],
                shuffle=True,
                name='train_producer',
                capacity=FLAGS.batch_size*2)
        validate_input_queue = tf.train.slice_input_producer(
                [validate_images, validate_labels],
                shuffle=False,
                name='validate_producer',
                capacity=FLAGS.batch_size*2)
        test_input_queue = tf.train.slice_input_producer(
                [test_images, test_labels],
                shuffle=False,
                name='test_producer',
                capacity=FLAGS.batch_size*2)

        ############################
        # Define Data Retrieval Op #
        ############################
        # these input queues automatically dequeue 1 slice
        train_image, train_label       = self.get_data_from_queue(train_input_queue)
        validate_image, validate_label = self.get_data_from_queue(validate_input_queue)
        test_image, test_label         = self.get_data_from_queue(test_input_queue)

        #####################
        # Data Augmentation #
        #####################
        # Network expects [0,255]
        train_image    = self.augment_image(train_image) * 255
        validate_image = image_utils.crop_and_resize_proper(validate_image, self.IM_SHAPE) * 255
        test_image     = image_utils.crop_and_resize_proper(test_image, self.IM_SHAPE) * 255

        ################
        # Minibatching #
        ################
        self.train_image_batch, self.train_label_batch = tf.train.batch(
                [train_image, train_label],
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.data_threads
                )
        tf.summary.image('post-aug', self.train_image_batch, max_outputs=6)
        self.validate_image_batch, self.validate_label_batch = tf.train.batch(
                [validate_image, validate_label],
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.data_threads
                )
        self.test_image_batch, self.test_label_batch = tf.train.batch(
                [test_image, test_label],
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.data_threads
                )
        # Data Pipeline is ready to go!
