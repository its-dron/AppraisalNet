# Python 3 compatability imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange

import numpy as np
import tensorflow as tf
import numpy as np
import os
import sys
import argparse
import threading
from time import time

from vgg16 import vgg16
from utils import CustomRunner

# Basic model parameters as external flags.
FLAGS = None

# FIX THIS LATER
def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
    """
    Runs one evaluation against the full epoch of data.
    Args:
      sess: The session in which the model has been trained.
      eval_correct: The Tensor that returns the number of correct predictions.
      images_placeholder: The images placeholder.
      labels_placeholder: The labels placeholder.
      data_set: The set of images and labels to evaluate, from
                input_data.read_data_sets().
    """
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   images_placeholder,
                                   labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))

def run_training():
    '''
    Run Training Loop
    '''
    # Declare graph
    model = vgg16()
    predictions = model.inference()
    loss = model.loss()
    train_op = model.optimize()

    # Collect summaries for TensorBoard
    summary = tf.summary.merge_all()

    # Create variable initializer op
    init = tf.global_variables_initializer()

    # Create checkpoint saver
    saver = tf.train.Saver()

    # Begin TensorFlow Session
    with tf.Session() as sess:
        # Instantiate a summary writer to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        # Run the Variable Initializer Op
        sess.run(init)

        # Load ImageNet pretrained weights if given
        if FLAGS.vgg_init:
            model.load_npz_weights()

        # Actually begin the training process
        for step in xrange(FLAGS.max_steps):
            start_time = time()

            #??????????????????#
            # SOMEHOW GET DATA #
            #??????????????????#

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value = sess.run([train_op, loss],
                    feed_dict=feed_dict)

            duration_time = time() - start_time

            # Write the summaries and print an overview
            if step % 100 == 0:
                # Print progress to stdout
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration_time))

                # Update the summary file
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # Evaluate Model
            if (step+1)%FLAGS.validation_freq==0 or (step+1)==Flags.max_steps:
                print('Training Data Eval:')
                # FINISH THIS AFTER DATA THREADS IS COMPLETE

            # Save Checkpoint
            if (step+1)%FLAGS.checkpoint_freq==0 or (step+1)==Flags.max_steps:
                checkpoint_filename = 'model_%i.ckpt' % step
                checkpoint_path = os.path.join(FLAGS.log_dir, checkpoint_filename)
                saver.save(sess, checkpoint_path, global_step=step)


def main(_):
    # Delete logs if they exist
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    if FLAGS.mode.lower() == 'train':
        run_training()
    elif FLAGS.mode.lower() == 'valid':
        # Probably get rid of validation mode
        pass
    elif FLAGS.mode.lower() == 'test':
        pass
    else:
        print("Invalid MODE: %s." % FLAGS.model.lower())
        return

if __name__ == "__main__":
    '''
    Parse command line inputs and store them into tf.FLAGS and run main()
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        type=string,
        default=0.01,
        help='Set network operation. {TRAIN, VALID, TEST}.'
    )
    parser.add_argument(
        '--initial_lr',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--decay_rate',
        type=float,
        default=0.95,
        help='Learning rate decay rate.'
    )
    parser.add_argument(
        '--decay_step',
        type=float,
        default=10000,
        help='Number of training exemplars before decreasing learning rate.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=200000,
        help='Number of mini-batch iterations to run trainer.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='/tmp/tensorflow/mnist/logs/fully_connected_feed',
        help='Directory to put the log data.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--input_data',
        type=str,
        default='/tmp/tensorflow/mnist/input_data',
        help='input data (format TBD).'
    )
    parser.add_argument(
        '--vgg_init',
        type=str,
        default=None,
        help='Path to npz file containing pretrained VGG16 weights.'
    )
    parser.add_argument(
        '--validation_freq',
        type=int,
        default=100,
        help='Minibatch Frequency to test and report validation score.'
    )
    parser.add_argument(
        '--checkpoint_freq',
        type=int,
        default=100,
        help='Minibatch Frequency to save a checkpoint file.'
    )




    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv[sys.arv[0]] + unparsed)
