"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-9-16 下午8:58
# @FileName: rnn.py
# @Email   : quant_master2000@163.com
======================
"""
from src.models.dl.tf_models.model import Model
import tensorflow as tf


class RnnModel(Model):
    """Recurrent Neural Network"""
    def __init__(self, args, task_type=None):
        Model.__init__(self, args, task_type)
        self.time_steps = None
        self.init_net()
        if self.task_type == 'regression':
            self.nclass = 1
            self.loss, self.train_op = self.build_model()
        else:
            self.loss, self.train_op, self.accuracy = self.build_model()

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
        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, self.time_steps, 1)

        # Define a rnn cell with tensorflow
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.layers[1], reuse=tf.AUTO_REUSE)

        # Get rnn cell output
        outputs, states = tf.nn.static_rnn(rnn_cell, x, dtype=tf.float32)  # ?*128

        # Linear activation, using rnn inner loop last output
        output = tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']
        return output

    def build_model(self):
        """
        构建模型，损失函数，优化器，学习算子等
        Optimizer is different from super class
        :return:
        """
        y_hat = self.forward(self.X)
        if self.task_type is None or self.task_type == 'classification':
            self.out = tf.nn.softmax(logits=y_hat)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=self.Y))
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            train_op = optimizer.minimize(loss, global_step=self.global_step)
            corr_pred = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.Y, 1))
            accuracy = tf.reduce_mean(tf.cast(corr_pred, tf.float32))
            return loss, train_op, accuracy

        if self.task_type == 'regression':
            loss = tf.reduce_mean(tf.square(y_hat - self.Y))
            # loss = tf.reduce_mean(tf.square(y_hat - self.Y), keep_dims=False)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            train_op = optimizer.minimize(loss, global_step=self.global_step)
            self.out = y_hat
            return loss, train_op

    def parse_tfrecord(self, tfrecord, record_type='mnist'):
        example = tf.parse_single_example(tfrecord, features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
            'num1': tf.FixedLenFeature([], tf.float32),
            'num2': tf.FixedLenFeature([], tf.int64)
        })
        image = tf.decode_raw(example['image'], tf.float32)
        label = tf.decode_raw(example['label'], tf.float32)
        image = tf.reshape(image, shape=[self.time_steps, self.layers[0]])
        # image = tf.reshape(image, shape=[self.layers[0]])
        label = tf.reshape(label, shape=[self.nclass])
        return image, label
