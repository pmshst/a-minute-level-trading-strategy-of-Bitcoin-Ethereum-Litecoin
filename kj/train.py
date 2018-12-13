import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from input_helper import InputHelper
from lstm_network import LSTM

def init_parameters():
    tf.app.flags.DEFINE_string("input_files", "../data/data_format2_20180916_20180923.h5", "")
    #tf.app.flags.DEFINE_string("input_files", "../data/data_format2_201808.h5", "")

    tf.app.flags.DEFINE_integer("bar_length", 30, "")
    tf.app.flags.DEFINE_string("assets", "0", "")
    tf.app.flags.DEFINE_integer("percent_dev", 10, "")
    tf.app.flags.DEFINE_boolean("regression", False, "")

    tf.app.flags.DEFINE_integer("num_epoches", 200, "")
    tf.app.flags.DEFINE_integer("batch_size", 32, "")
    tf.app.flags.DEFINE_float("learning_rate", 0.008, "")

    tf.app.flags.DEFINE_integer("time_steps", 12, "")
    tf.app.flags.DEFINE_integer("num_layers", 2, "")
    tf.app.flags.DEFINE_integer("hidden_units", 32, "")
    tf.app.flags.DEFINE_float("l2_reg_lambda", 0.1, "")
    tf.app.flags.DEFINE_float("dropout_keep_prob", 1.0, "")

init_parameters()

def main(argv):
    FLAGS = tf.app.flags.FLAGS
    #print("Parameters:")
    #for attr, value in sorted(FLAGS.flag_values_dict().items()):
    #    print("{} = {}".format(attr, value))

    assets = list(map(int, FLAGS.assets.split(",")))
    input_files = FLAGS.input_files.split(",")
    input_helper = InputHelper()
    # train_set: (X_train, Y_train), where X_train's shape is [length, time_steps, len(assets) * 3], and Y_train's shape is [length, len(assets) * 3]
    train_set, dev_set = input_helper.get_dataset(input_files, FLAGS.bar_length, assets, FLAGS.time_steps, FLAGS.percent_dev, shuffle=True)
    print(train_set[0].shape, train_set[1].shape)
    print(dev_set[0].shape, dev_set[1].shape)

    #train_set = train_set[0][:100], train_set[1][:100]
    #dev_set = dev_set[0][:1], dev_set[1][:1]
    batch_size = train_set[0].shape[0]
    batch_size = 16
    # Training
    print("starting graph def")
    with tf.Graph().as_default():
        sess = tf.Session()
        #print("session started")

        with sess.as_default():
            model = LSTM(batch_size, FLAGS.time_steps, len(assets), FLAGS.num_layers, FLAGS.hidden_units, FLAGS.regression)
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            #optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
            
            #print("model object initialized")

        train_op = optimizer.minimize(model.loss)
        #print("training operation defined")

        sess.run(tf.global_variables_initializer())
        #print("variables initialized")

        def train_step(x_batch, y_batch):
            _, loss, accuracy = sess.run([train_op, model.loss, model.accuracy], 
                    feed_dict={model.input_x: x_batch, model.input_y: y_batch, model.dropout_keep_prob: FLAGS.dropout_keep_prob})
            return loss, accuracy

        def dev_step(x_batch, y_batch):
            _, loss, accuracy = sess.run([train_op, model.loss, model.accuracy], 
                    feed_dict={model.input_x: x_batch, model.input_y: y_batch, model.dropout_keep_prob: 1.0})
            return loss, accuracy
    
        num_batches_per_epoch = train_set[0].shape[0] // batch_size 
        if train_set[0].shape[0] % batch_size != 0:
            num_batches_per_epoch += 1
        print("num_batches_per_epoch = {}".format(num_batches_per_epoch))
        batches = input_helper.batch_iter(list(zip(train_set[0], train_set[1])), batch_size, FLAGS.num_epoches, shuffle=True)
        #print("Training started")
        train_loss_history = []
        train_accuracy_history = []
        test_accuracy_history = []
        for i in range(FLAGS.num_epoches * num_batches_per_epoch):
            batch = next(batches)
            if len(batch) < 1:
                continue
            x_batch, y_batch = zip(*batch)
            train_loss, train_accuracy = train_step(x_batch, y_batch)
            train_loss_history.append(train_loss)
            train_accuracy_history.append(train_accuracy)
            
            dev_loss, dev_accuracy = dev_step(dev_set[0], dev_set[1])
            test_accuracy_history.append(dev_accuracy)
            
            if i == 0 or (i + 1) % 50 == 0:
                print("Step {}: training loss={}, training accuracy={}, testing accuracy={}".format(i + 1, train_loss, train_accuracy, dev_accuracy))
            #if train_loss < 0.01:
            #    break
        print("Training finished")
        
        plt.plot(range(len(train_loss_history)), train_loss_history)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.show()

        #plt.plot(range(len(train_accuracy_history)), train_accuracy_history)
        #plt.xlabel("epoch")
        #plt.ylabel("train accuracy")
        #plt.show()

        #plt.plot(range(len(test_accuracy_history)), test_accuracy_history)
        #plt.xlabel("epoch")
        #plt.ylabel("test accuracy")
        #plt.show()

        saver = tf.train.Saver()
        saver.save(sess, "./save/" + "".join(map(str, assets)) + "/" + "".join(map(str, assets)))

if __name__ == "__main__":
    tf.app.run()
