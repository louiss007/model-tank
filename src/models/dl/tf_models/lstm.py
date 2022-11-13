"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-9-16 下午8:59
# @FileName: lstm.py
# @Email   : quant_master2000@163.com
======================
"""
from src.models.dl.tf_models.rnn import RnnModel
import tensorflow as tf


class LstmModel(RnnModel):
    """Long Short Term Memory"""
    def __init__(self, args, task_type=None):
        RnnModel.__init__(self, args, task_type)
        self.time_steps = None
        self.init_net()
        if self.task_type == 'regression':
            self.nclass = 1
            self.loss, self.train_op = self.build_model()
        else:
            self.loss, self.train_op, self.accuracy = self.build_model()

    def init_net(self):
        self.time_steps = 28
        self.X = tf.placeholder(tf.float32, [None, self.time_steps, self.layers[0]], name='input_x')
        self.Y = tf.placeholder(tf.float32, [None, self.nclass], name='input_y')
        self.weights = {
            'out': tf.Variable(tf.random_normal([self.layers[1], self.nclass]))
        }
        self.biases = {
            'out': tf.Variable(tf.random_normal([self.nclass]))
        }

    def forward(self, x):
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, timesteps, n_input)
        # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, self.time_steps, 1)

        # Define a lstm cell with tensorflow
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.layers[1], forget_bias=1.0, reuse=tf.AUTO_REUSE)

        # Get lstm cell output
        outputs, states = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)  # ?*128

        # Linear activation, using rnn inner loop last output
        output = tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']
        return output
