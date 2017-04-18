# Python 3 compatability imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange

import tensorflow as tf
from tensorflow.python.client import timeline
import os
import sys
from time import time
import pdb

from vgg16 import vgg16 as model
#from shallow import shallow as vgg16
from data_pipeline import DataPipeline

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS

def run_training():
    '''
    Run Training Loop
    '''
    # GPU/CPU Flag
    if FLAGS.gpu is not None:
        compute_string = '/gpu:' + str(FLAGS.gpu)
    else:
        compute_string ='/cpu:0'

    #####################
    # Setup Data Queues #
    #####################
    with tf.device("/cpu:0"):
        data_pipeline = DataPipeline(augment=True)
        train_x, train_y = data_pipeline.batch_ops()

    #######################
    # Declare train graph #
    #######################
    with tf.device(compute_string):
        train_model = model(train_x, train_y)
        train_predictions = train_model.inference()
        train_acc = train_model.evaluate()
        train_loss = train_model.loss()
        train_op = train_model.optimize()
        tf.summary.scalar('train_loss', train_loss)
        tf.summary.scalar('train_acc', train_acc)

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
    session_config = tf.ConfigProto(
            log_device_placement=True, # Record Node Devices
            allow_soft_placement=True
            )
    with tf.Session(config=session_config) as sess:
        # Resume training or
        # Run the Variable Initializer Op
        if FLAGS.resume is None:
            sess.run(init)
            # Load ImageNet pretrained weights if given
            if FLAGS.vgg_init:
                try:
                    train_model.load_npz_weights(FLAGS.vgg_init, sess)
                except:
                    print('Failed to load pretrained weights.')
        else: # Try and Resume specific, fall back to latest
            saver = tf.train.import_meta_graph('model.meta')
            try:
                saver.restore(sess, FLAGS.resume)
            except:
                saver.restore(sess, tf.train.latest_checkpoint(FLAGS.log_dir))

        # Coordinator hands data fetching threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Instantiate a summary writer to output summaries and the Graph.
        train_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, "train"), sess.graph)

        # Actually begin the training process
        try:
            for step in xrange(FLAGS.max_steps):
                if coord.should_stop():
                    break
                start_time = time()

                # Run one step of the model.
                _, loss_value, acc = sess.run([train_op, train_loss, train_acc])
                duration_time = time() - start_time

                # debug profiler on step 3
                # open timeline.json in chrome://tracing/
                if FLAGS.profile and step == 3:
                    run_metadata = tf.RunMetadata()
                    _, loss, acc = sess.run([train_op, train_loss, train_acc],
                            options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                            run_metadata=run_metadata)
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open('timeline.json', 'w') as f:
                        f.write(ctf)

                # Display progress
                if step % 1 == 0:
                    # Print progress to stdout
                    print('Step %d: loss = %.2f, acc = %.2f (%.3f sec)' %
                            (step, loss_value, acc, duration_time))
                    sys.stdout.flush()

                # Write the summaries
                if step % 20 == 0:
                    # Update the summary file
                    summary_str = sess.run(summary)
                    train_writer.add_summary(summary_str, step)
                    train_writer.flush()

                # Save Model Checkpoint
                if (step+1)%FLAGS.checkpoint_freq==0 or (step+1)==FLAGS.max_steps:
                    checkpoint_path = os.path.join(FLAGS.log_dir, 'model')
                    saver.save(sess, checkpoint_path, global_step=step)
                #loop_time = time() - start_time
                #print('Total Loop Time: %.3f' % loop_time)
        except tf.errors.OutOfRangeError:
            print('Done Training -- Epoch limit reached.')
        except Exception as e:
            print("Exception encountered: ", e)

        # Stop Queueing data, we're done!
        coord.request_stop()
        coord.join(threads)

def run_test():
    #ToDo:
    # Deployment code
    # Should Run on an image
    # or an entire directory

    # GPU/CPU Flag
    if FLAGS.gpu is not None:
        compute_string = '/gpu:' + str(FLAGS.gpu)
    else:
        compute_string ='/cpu:0'

def main(_):
    # Delete logs if they exist
    if FLAGS.resume is None:
        if tf.gfile.Exists(FLAGS.log_dir):
            tf.gfile.DeleteRecursively(FLAGS.log_dir)
        tf.gfile.MakeDirs(FLAGS.log_dir)

    if FLAGS.mode.lower() == 'train':
        run_training()
    elif FLAGS.mode.lower() == 'test':
        run_test()
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
        'Set network operation. {TRAIN, TEST}.'
    )
    flags.DEFINE_integer(
        'num_classes',
        10,
        'Number of classes'
    )
    flags.DEFINE_float(
        'initial_lr',
        0.0001,
        'Initial learning rate.'
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
        'checkpoint_freq',
        100,
        'Minibatch Frequency to save a checkpoint file.'
    )
    flags.DEFINE_integer(
        'data_threads',
        1,
        'Number of QueueRunner Threads.'
    )
    flags.DEFINE_integer(
        'gpu',
        None,
        'Which GPU device to use'
    )
    flags.DEFINE_boolean(
        'profile',
        False,
        'Whether or not to run profiler on iter 5.'
    )
    flags.DEFINE_string(
        'resume',
        None,
        'Resume Checkpoint'
    )

    tf.app.run()
