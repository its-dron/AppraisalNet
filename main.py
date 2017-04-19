# Python 3 compatability imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange

# Import scipy after tensorflow causes issues for some reason
from scipy.io import savemat

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Vary TF Verbosity

import tensorflow as tf
import numpy as np
from tensorflow.python.client import timeline
import os
import sys
from time import time
import re
import pdb
#from scipy.io import savemat

from vgg16 import vgg16 as model
#from shallow import shallow as vgg16
from data_pipeline import DataPipeline

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS

def sort_nicely(l):
    """
    Sort the given list in the way that humans expect.
    From Ned Batchelder
    https://nedbatchelder.com/blog/200712/human_sorting.html
    """
    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        def tryint(s):
            try:
                return int(s)
            except:
                return s
        return [ tryint(c) for c in re.split('([0-9]+)', s) ]
    l.sort(key=alphanum_key)
    return l

def get_checkpoints(checkpoint_dir):
    '''
    Finds all checkpoints in a directory and returns them in order
    from least iterations to most iterations
    '''
    meta_list=[]
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.meta'):
            meta_list.append(os.path.join(checkpoint_dir, file[:-5]))
    meta_list = sort_nicely(meta_list)
    return meta_list

def optimistic_restore(session, save_file):
    '''
    A Caffe-style restore that loads in variables
    if they exist in both the checkpoint file and the current graph.
    Call this after running the global init op.
    By DanielGordon10 on December 27, 2016
    https://github.com/tensorflow/tensorflow/issues/312
    With RalphMao tweak.
    '''
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x:x.name.split(':')[0],
            tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            try:
                curr_var = name2var[saved_var_name]
                var_shape = curr_var.get_shape().as_list()
                if var_shape == saved_shapes[saved_var_name]:
                    restore_vars.append(curr_var)
            except:
                print("{} couldn't be loaded.".format(saved_var_name))
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)

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
        with tf.variable_scope('train'):
            data_pipeline = DataPipeline(augment=True)
            train_x, train_y = data_pipeline.batch_ops()

    #######################
    # Declare train graph #
    #######################
    with tf.device(compute_string):
        phase = tf.placeholder(tf.bool, name='phase')
        train_model = model(train_x, train_y, phase)
        train_predictions = train_model.inference()
        train_acc = train_model.evaluate()
        train_loss, gt_y = train_model.loss()
        train_op = train_model.optimize()
        global_step = train_model.get_global_step()
        tf.summary.scalar('train_loss', train_loss)
        tf.summary.scalar('train_acc', train_acc)

    #############################
    # Setup Summaries and Saver #
    #############################

    # Collect summaries for TensorBoard
    summary = tf.summary.merge_all()
    # Create variable initializer op
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # Create checkpoint saver
    saver = tf.train.Saver(max_to_keep=100)

    # Begin TensorFlow Session
    session_config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=session_config) as sess:
        # Resume training or
        # Run the Variable Initializer Op
        sess.run(init)
        if FLAGS.resume==True:
            try:
                meta_list = get_checkpoints(FLAGS.log_dir)
                optimistic_restore(sess, meta_list[-1])
                resume_status = True
            except:
                print('Checkpoint Load Failed')
                print('Training from scratch')
                resume_status = False
        if not resume_status:
            try:
                train_model.load_pretrained_weights(sess)
            except:
                print('Failed to load pretrained weights.')
                print('Training from scratch')
                sys.stdout.flush()

        # Coordinator hands data fetching threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Instantiate a summary writer to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        # Actually begin the training process
        try:
            for step in xrange(FLAGS.max_steps):
                if coord.should_stop():
                    break
                start_time = time()

                # Run one step of the model.
                _, loss_value, acc = sess.run([train_op, train_loss, train_acc],
                        feed_dict={phase:True})
                global_step_value = global_step.eval()
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
                if global_step_value % 1 == 0:
                    # Print progress to stdout
                    print('Step %d: loss = %.2f, acc = %.2f (%.3f sec)' %
                            (global_step_value, loss_value, acc, duration_time))
                    sys.stdout.flush()

                # Write the summaries
                if global_step_value % 20 == 0:
                    # Update the summary file
                    summary_str = sess.run(summary,
                            feed_dict={phase:False})
                    summary_writer.add_summary(summary_str, global_step_value)
                    summary_writer.flush()

                # Save Model Checkpoint
                if (global_step_value)%FLAGS.checkpoint_freq==0 or \
                        (global_step_value+1)==FLAGS.max_steps:
                    checkpoint_path = os.path.join(FLAGS.log_dir, 'model')
                    saver.save(sess, checkpoint_path,
                            global_step=global_step)
                #loop_time = time() - start_time
                #print('Total Loop Time: %.3f' % loop_time)
        except tf.errors.OutOfRangeError:
            print('Done Training -- Epoch limit reached.')
            sys.stdout.flush()
        except Exception as e:
            print("Exception encountered: ", e)
            sys.stdout.flush()

        # Stop Queueing data, we're done!
        coord.request_stop()
        coord.join(threads)

def run_validate():
    # Get all ckpt names in log dir (without meta ext)
    meta_list = get_checkpoints(FLAGS.log_dir)

    # GPU/CPU Flag
    if FLAGS.gpu is not None:
        compute_string = '/gpu:' + str(FLAGS.gpu)
    else:
        compute_string ='/cpu:0'

    # Iterate through the checkpoints
    val_loss = []
    val_acc = []
    val_itr = []
    for ckpt_path in meta_list:
        tf.reset_default_graph()

        ####################
        # Setup Data Queue #
        ####################
        with tf.device("/cpu:0"):
            with tf.variable_scope('validate') as scope:
                data_pipeline = DataPipeline(augment=False, num_epochs=1)
                validate_x, validate_y = data_pipeline.batch_ops()

        with tf.device(compute_string):
            ##########################
            # Declare Validate Graph #
            ##########################
            # Sets train/test mode; currently only used for BatchNormalization
            # True: Train   False: Test
            phase = tf.placeholder(tf.bool, name='phase')
            validate_model = model(validate_x, validate_y, phase)

            # Delete extraneous info when done debugging
            validate_pred = validate_model.inference()
            validate_acc = validate_model.evaluate()
            validate_loss, gt_y = validate_model.loss()
            global_step = validate_model.get_global_step()

        init = tf.group(tf.global_variables_initializer(),
                tf.local_variables_initializer())

        session_config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=session_config) as sess:
            sess.run(init)

            # Coordinator hands data fetching threads
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            optimistic_restore(sess, ckpt_path)
            global_step_value = global_step.eval()
            try:
                step = 0
                cum_loss = 0
                cum_acc = 0
                cum_time = 0
                while True:
                    if coord.should_stop():
                        break
                    step += 1
                    start_time = time()
                    loss_value, acc_value, prediction_value, gt_value = sess.run(
                            [validate_loss, validate_acc, validate_pred, gt_y],
                            feed_dict={phase:False})
                    duration_time = time() - start_time

                    cum_loss += loss_value
                    cum_acc += acc_value
                    cum_time += duration_time

                    if step % 1 == 0:
                        # Print progress to stdout
                        if FLAGS.print_pred:
                            print('Step %d: loss = %.4f acc = %.4f (%.3f sec)' %
                                    (step, loss_value, acc_value, duration_time))
                            print('Prediction:{}'.format(prediction_value))
                            print('GT:{}'.format(gt_value))
                        sys.stdout.flush()

            except tf.errors.OutOfRangeError:
                step -= 1
            except Exception as e:
                step -= 1

            # Stop Queueing data, we're done!
            coord.request_stop()
            coord.join(threads)

        avg_loss = cum_loss / step
        avg_acc = cum_acc / step
        avg_time = cum_time / step

        val_loss.append(float(avg_loss))
        val_acc.append(float(avg_acc))
        val_itr.append(int(global_step_value))

        print('Results For Load File: %s' % ckpt_path)
        print('Average_Loss = %.4f' % avg_loss)
        print('Average_Acc = %.4f' % avg_acc)
        print('Run Time: %.2f' % cum_time)
        sys.stdout.flush()

    val_loss = np.asarray(val_loss)
    val_acc = np.asarray(val_acc)
    val_itr = np.asarray(val_itr)

    best_loss = np.amin(val_loss)
    best_acc  = np.amax(val_acc)
    best_itr  = val_itr[np.argmax(val_acc)]

    print('Overall Results')
    print('Minimum Loss: %.4f' % best_loss)
    print('Maximum Acc: %.4f' % best_acc)
    print('Best Checkpoint: %d' % best_itr)

    save_path = os.path.join(FLAGS.log_dir, 'validation_results.mat')
    save_dict = {
            'val_loss':val_loss,
            'val_acc':val_acc,
            'val_itr':val_itr,
            }
    savemat(save_path, save_dict, appendmat=False)

def run_test():
    pass

def main(_):
    if FLAGS.mode.lower() == 'train':
        # Delete logs if they exist
        if FLAGS.resume == False:
            if tf.gfile.Exists(FLAGS.log_dir):
                tf.gfile.DeleteRecursively(FLAGS.log_dir)
            tf.gfile.MakeDirs(FLAGS.log_dir)
        run_training()
    elif FLAGS.mode.lower() == 'validate':
        run_validate()
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
    flags.DEFINE_boolean(
        'resume',
        True,
        'Resume training from latest checkpoint.'
    )
    flags.DEFINE_boolean(
        'print_pred',
        False,
        'Print Predictions and Ground Truths.'
    )
    tf.app.run()
