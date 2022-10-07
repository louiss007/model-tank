"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-9-16 下午9:00
# @FileName: gru.py
# @Email   : quant_master2000@163.com
======================
"""
from rnn import RnnModel
import tensorflow as tf


class GruModel(RnnModel):
    """Gated Recurrent Unit"""
    def __init__(self, args, task_type=None):
        RnnModel.__init__(self, args, task_type)
        self.time_steps = None
        self.init_net()

    def init_net(self):
        self.time_steps = 28
        self.X = tf.placeholder(tf.float32, [None, self.time_steps, self.layers[0]])
        self.Y = tf.placeholder(tf.float32, [None, self.nclass])
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
        gru_cell = tf.nn.rnn_cell.GRUCell(self.layers[1], reuse=tf.AUTO_REUSE)

        # Get gru cell output
        outputs, state = tf.nn.static_rnn(gru_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        output = tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']
        return output
