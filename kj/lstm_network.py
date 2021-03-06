import tensorflow as tf
import numpy as np

class LSTM:
    def __init__(self, batch_size, time_steps, input_size, num_layers, hidden_units, regression=False):
        self.input_x = tf.placeholder(tf.float32, [None, time_steps, input_size * 5], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, input_size], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.pred_val = self.model(self.input_x, time_steps, input_size, num_layers, hidden_units, self.dropout_keep_prob)
        if regression:
            self.accuracy = tf.reduce_mean(tf.maximum(.0, tf.sign(tf.multiply(self.pred_val, self.input_y))))
            self.loss = self.mse_loss(self.pred_val, self.input_y)
        else:
            self.pred_prob = tf.nn.sigmoid(self.pred_val, name="pred_prob")
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.pred_prob), tf.reshape(self.input_y, [-1])), tf.float32))
            self.loss = self.cross_entropy_loss(self.pred_val, tf.reshape(self.input_y, [-1]))

    # x's shape is [batch_size, time_steps, input_size * 3]
    def model(self, x, time_steps, input_size, num_layers, hidden_units, dropout_keep_prob):
        x = tf.unstack(x, axis=1)
        lstm_cells = []
        for _ in range(num_layers):
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_units, forget_bias=1.0, state_is_tuple=True)
            dropout_lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=dropout_keep_prob)
            lstm_cells.append(dropout_lstm_cell)
        stacked_lstm_cells = tf.nn.rnn_cell.MultiRNNCell(cells=lstm_cells, state_is_tuple=True)
        outputs, state = tf.nn.static_rnn(stacked_lstm_cells, x, dtype=tf.float32)
        W = self.weights(hidden_units, input_size)
        b = self.bias(input_size)
        return tf.reshape(tf.add(tf.matmul(outputs[-1], W), b), shape=[-1], name="pred_val")

    # y's shape is [batch_size, input_size]
    def mse_loss(self, pred_val, y):
        return tf.losses.mean_squared_error(tf.reshape(pred_val, [-1]), tf.reshape(y, [-1]))

    def cross_entropy_loss(self, pred_val, y):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=pred_val))

    def weights(self, input_size, output_size):
        W = tf.Variable(tf.random_normal(shape=[input_size, output_size]) * 0.01, name="weights")
        return W

    def bias(self, output_size):
        b = tf.Variable(tf.constant(0.1, shape=[output_size]), name="bias")
        return b
