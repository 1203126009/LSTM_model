#!/usr/bin/python
# -*- coding:utf-8 -*-


import tensorflow as tf
from rcnn_model import RCNN
import time
import os
import sys
import datetime
from data_lstm_helper import load_data_and_labels, batch_iter
import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Data loading params
tf.flags.DEFINE_string("train_data_file", "./train_data.json", "Data source for the train data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 27, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("filter_nums", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_integer("sequence_length", 100, "every sequences length")
tf.flags.DEFINE_integer("num_classes", 7, "the number of classes")
tf.flags.DEFINE_integer("hidden_dim", 128, "the hidden unit number")
tf.flags.DEFINE_float("learning_rate", 1e-3, "learning rate")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("epochs", 1000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 100, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
print("Parameters:")
for key, value in FLAGS.flag_values_dict().items():
    print("{}={}".format(key.upper(), value))

print("Load data\n")
x_train, y_train, x_test, y_test = load_data_and_labels(
    FLAGS.train_data_file, FLAGS.embedding_dim, FLAGS.sequence_length)

# train

with tf.Graph().as_default():
    session = tf.Session()
    with session.as_default():
        rcnn = RCNN(
            embedding_dim=FLAGS.embedding_dim,
            sequence_length=FLAGS.sequence_length,
            hidden_dim=FLAGS.hidden_dim,
            num_classes=FLAGS.num_classes,
            dropout_keep_prob=FLAGS.dropout_keep_prob,
            filter_nums=FLAGS.filter_nums,
            learning_rate=FLAGS.learning_rate,
            batch_size=FLAGS.batch_size,
            epochs=FLAGS.epochs
        )

        # train produce
        global_step = tf.Variable(0, trainable=False, name='global_step')
        optimizer = tf.train.AdamOptimizer(rcnn.learning_rate)
        train_op = optimizer.apply_gradients(optimizer.compute_gradients(rcnn.loss), global_step=global_step)

        # output for model and summaries
        timestamp = str(int(time.time()))
        outputdir = os.path.abspath(os.path.join(os.path.curdir, 'runs', timestamp))
        print("Writing to dir: {}".format(outputdir))

        # Summary for loss and accuracy
        loss_summary = tf.summary.scalar('loss', rcnn.loss)
        acc_summary = tf.summary.scalar('accuracy', rcnn.accuracy)

        # train summary
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(outputdir, 'summary', 'train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, session.graph)

        # test summary
        test_summary_op = tf.summary.merge([loss_summary, acc_summary])
        test_summary_dir = os.path.join(outputdir, 'summary', 'test')
        test_summary_writer = tf.summary.FileWriter(test_summary_dir, session.graph)

        # checkpoint file
        checkpoint_dir = os.path.join(outputdir, 'checkpoint', 'model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        # 只保存最近的max_to_keep个检查点文件
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
        # saver.restore(session,"./runs/1563943313/checkpoint/model-80500")
        # initialize all variables
        session.run(tf.global_variables_initializer())


        def train_step(x_batch, y_batch):
            """
            A simple train step
            :param x_batch:
            :param y_batch:
            :return:
            """
            feed_dict = {
                rcnn.input_x: x_batch,
                rcnn.input_y: y_batch,
                rcnn.keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summary, loss, accuracy = session.run(
                [train_op, global_step, train_summary_op, rcnn.loss, rcnn.accuracy], feed_dict)

            time_str = datetime.datetime.now().isoformat()
            print("{}---step:{}, loss:{}, accuracy:{}".format(time_str, step, loss, accuracy))

            train_summary_writer.add_summary(summary, step)


        def test_step(x_batch, y_batch):
            """
            A simple test step
            :param x_batch:
            :param y_batch:
            :return:
            """
            feed_dict = {
                rcnn.input_x: x_batch,
                rcnn.input_y: y_batch,
                rcnn.keep_prob: 1.0
            }

            step, summary, loss, accuracy = session.run(
                [global_step, test_summary_op, rcnn.loss, rcnn.accuracy], feed_dict=feed_dict
            )
            time_str = datetime.datetime.now().isoformat()
            print("{}---step:{}, loss:{}, accuracy:{}".format(time_str, step, loss, accuracy))

            test_summary_writer.add_summary(summary, step)
            return loss


        # Generate batch
        batches = batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.epochs)

        # Statistic the number of parameters
        print("the number of parameters is: {}".format(np.sum([np.prod(v.get_shape().as_list())
                                                               for v in tf.trainable_variables()])))
        # Train loop
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(session, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                current_loss = test_step(x_test, y_test)

            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(session, checkpoint_dir, global_step=global_step)
                print('Saved model checkpoint to {}'.format(path))