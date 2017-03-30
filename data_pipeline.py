import numpy as np
import tensorflow as tf
import csv
import os
import random

FLAGS = tf.app.flags.FLAGS

class DataPipeline:
    '''
    Setup Queue and data preprocessor
    '''
    def __init__(self):
        self.setup_data_pipeline()

    def encode_labels(self, labels):
        '''
        Do preprocessing on labels (such as one-hot encoding)
        Input: list of labels (each element is a string)
        Returns: list of formatted labels
        '''
        numerical = [ float(x) for x in labels]
        return numerical

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
        return image

    def get_data_from_queue(self, input_queue):
        '''
        Reads image from disk
        Consumes a single filename and label.
        Resizes image as necessary.
        Returns an image tensor and a label.
        '''
        IM_SHAPE = [224, 224, 3]
        file_content = tf.read_file(input_queue[0])
        image = tf.image.decode_jpeg(file_content, channels=IM_SHAPE[2])
        resized_image = tf.image.resize_images(image, IM_SHAPE[0:2])
        label = input_queue[1]
        return resized_image, label

    def train_batch_ops(self):
        return self.train_image_batch, self.train_label_batch

    def validate_batch_ops(self):
        return self.validate_image_batch, self.validate_label_batch

    def test_batch_ops(self):
        return self.test_image_batch, self.test_label_batch

    def setup_data_pipeline(self):
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
        validate_set_size = np.ceil(FLAGS.validate_percentage * n_exemplars)
        test_set_size = np.ceil(FLAGS.test_percentage * n_exemplars)

        partitions = [0]*n_exemplars
        partitions[:validate_set_size] = [1] * validate_set_size
        partitions[validate_set_size:(validate_set_size + test_set_size)] = [2] * test_set_size
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
                name='train_producer')
        validate_input_queue = tf.train.slice_input_producer(
                [validate_images, validate_labels],
                shuffle=False,
                name='validate_producer')
        test_input_queue = tf.train.slice_input_producer(
                [test_images, test_labels],
                shuffle=False,
                name='test_producer')

        ############################
        # Define Data Retrieval Op #
        ############################
        train_image, train_label = self.get_data_from_queue(train_input_queue)
        validate_image, validate_label = self.get_data_from_queue(validate_input_queue)
        test_image, test_label = self.get_data_from_queue(test_input_queue)

        #####################
        # Data Augmentation #
        #####################
        train_image = self.augment_image(train_image)

        ################
        # Minibatching #
        ################
        self.train_image_batch, self.train_label_batch = tf.train.batch(
                [train_image, train_label],
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.data_threads
                )
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
