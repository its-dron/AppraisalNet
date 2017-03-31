# Python 3 compatability imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange

import tensorflow as tf
import os
import threading
from time import time

from vgg16 import vgg16
from data_pipeline import DataPipeline

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS

def run_training():
    '''
    Run Training Loop
    '''

    #####################
    # Setup Data Queues #
    #####################
    data_pipeline = DataPipeline()
    train_x, train_y = data_pipeline.train_batch_ops()
    validate_x, validate_y = data_pipeline.validate_batch_ops()
    test_x, test_y = data_pipeline.test_batch_ops()

    #######################
    # Declare train graph #
    #######################
    train_model = vgg16(train_x, train_y)
    predictions = train_model.inference()
    train_loss = train_model.loss()
    train_op = train_model.optimize()
    tf.summary.scalar('train_loss', train_loss)

    ##########################
    # Declare validate graph #
    ##########################
    validate_model = vgg16(validate_x, validate_y)
    predictions = validate_model.inference()
    validate_loss = validate_model.loss()
    validate_acc = validate_model.evaluate()
    tf.summary.scalar('validate_loss', validate_loss)
    tf.summary.scalar('validate_acc', validate_acc)

    ##########################
    # Declare test graph #
    ##########################
    test_model = vgg16(test_x, test_y)
    predictions = test_model.inference()

    #############################
    # Setup Summaries and Saver #
    #############################

    # Collect summaries for TensorBoard
    summary = tf.summary.merge_all()
    # Create variable initializer op
    init = tf.global_variables_initializer()
    # Create checkpoint saver
    saver = tf.train.Saver()
    # Begin TensorFlow Session
    with tf.Session() as sess:
        # Run the Variable Initializer Op
        sess.run(init)

        # Coordinator hands data fetching threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Instantiate a summary writer to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        # Load ImageNet pretrained weights if given
        if FLAGS.vgg_init:
            train_model.load_npz_weights()

        # Actually begin the training process
        try:
            for step in xrange(FLAGS.max_steps):
                if coord.should_stop():
                    break
                start_time = time()

                # Run one step of the model.  The return values are the activations
                # from the `train_op` (which is discarded) and the `loss` Op.  To
                # inspect the values of your Ops or variables, you may include them
                # in the list passed to sess.run() and the value tensors will be
                # returned in the tuple from the call.
                _, loss_value = sess.run([train_op, train_loss])

                duration_time = time() - start_time

                # Write the summaries and display progress
                if step % 100 == 0:
                    # Print progress to stdout
                    print('Step %d: loss = %.2f (%.3f sec)' %
                            (step, loss_value, duration_time))

                    # Update the summary file
                    summary_str = sess.run(summary)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                # Evaluate Model
                if (step+1)%FLAGS.validation_freq==0 or (step+1)==FLAGS.max_steps:
                    print('Training Data Eval:')
                    validate_loss_value, validate_acc_value = \
                            sess.run(validate_loss, validate_acc)
                    print('  Validation loss = %.2f   acc = %.2f' %
                            (validate_loss_value, validate_acc_value))

                # Save Checkpoint
                if (step+1)%FLAGS.checkpoint_freq==0 or (step+1)==FLAGS.max_steps:
                    checkpoint_filename = 'model_%i.ckpt' % step
                    checkpoint_path = os.path.join(FLAGS.log_dir, checkpoint_filename)
                    saver.save(sess, checkpoint_path, global_step=step)
        except tf.errors.OutOfRangeError:
            print('Done Training -- Epoch limit reached.')
        except Exception as e:
            print("Exception encountered: ", e)
        finally:
            coord.request_stop()

        # TODO
        # Run Test Dataset

        # Stop Queueing data, we're done!
        coord.request_stop()
        coord.join(threads)

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
    flags.DEFINE_string(
        'mode',
        'train',
        'Set network operation. {TRAIN, VALID, TEST}.'
    )
    flags.DEFINE_integer(
        'num_classes',
        20,
        'Number of classes'
    )
    flags.DEFINE_float(
        'initial_lr',
        0.01,
        'Initial learning rate.'
    )
    flags.DEFINE_float(
        'decay_rate',
        0.95,
        'Learning rate decay rate.'
    )
    flags.DEFINE_integer(
        'decay_step',
        10000,
        'Number of training exemplars before decreasing learning rate.'
    )
    flags.DEFINE_integer(
        'max_steps',
        200000,
        'Number of mini-batch iterations to run trainer.'
    )
    flags.DEFINE_string(
        'log_dir',
        '/tmp/tensorflow/logs/appraisalnet',
        'Directory to put the log data.'
    )
    flags.DEFINE_integer(
        'batch_size',
        100,
        'Batch size.  Must divide evenly into the dataset sizes.'
    )
    flags.DEFINE_string(
        'image_dir',
        'images',
        'Batch size.  Must divide evenly into the dataset sizes.'
    )
    flags.DEFINE_string(
        'input_data',
        '/tmp/tensorflow/mnist/input_data',
        'input data (format TBD). Currently a csv for training'
    )
    flags.DEFINE_string(
        'vgg_init',
        None,
        'Path to npz file containing pretrained VGG16 weights.'
    )
    flags.DEFINE_integer(
        'validation_freq',
        100,
        'Minibatch Frequency to test and report validation score.'
    )
    flags.DEFINE_integer(
        'checkpoint_freq',
        100,
        'Minibatch Frequency to save a checkpoint file.'
    )
    flags.DEFINE_integer(
        'data_threads',
        1,
        'Number of QueueRunner Threads.'
    )
    flags.DEFINE_float(
        'validate_percentage',
        0.1,
        'Percentage of dataset to use for validation.'
    )

    tf.app.run()
