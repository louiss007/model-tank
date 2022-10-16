"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-9-16 下午8:57
# @FileName: fnn.py
# @Email   : quant_master2000@163.com
======================
"""
from src.models.dl.tf_models.model import Model
import tensorflow as tf
import numpy as np


class FnnModel(Model):
    """Feedforward Neural Network"""
    def __init__(self, args, task_type=None):
        Model.__init__(self, args, task_type)
        self.init_net()
        if self.task_type == 'regression':
            self.nclass = 1
            self.loss, self.train_op = self.build_model()
        else:
            self.loss, self.train_op, self.accuracy = self.build_model()

    def init_net(self):
        self.X = tf.placeholder(tf.float32, [None, self.layers[0]])
        self.Y = tf.placeholder(tf.float32, [None, self.nclass])
        if len(self.layers) != 1:
            for i in range(1, len(self.layers)):
                init_method = np.sqrt(2.0 / (self.layers[i - 1] + self.layers[i]))
                self.weights['h' + str(i)] = tf.Variable(np.random.normal(
                    loc=0, scale=init_method, size=(self.layers[i - 1], self.layers[i])),
                    dtype=np.float32)
                self.biases['b' + str(i)] = tf.Variable(
                    np.random.normal(loc=0, scale=init_method, size=(1, self.layers[i])),
                    dtype=np.float32)
            self.weights['out'] = tf.Variable(tf.random_normal([self.layers[-1], self.nclass]))
            self.biases['out'] = tf.Variable(tf.random_normal([self.nclass]))

    def forward(self, x):
        for i in range(1, len(self.layers)):
            x = tf.add(tf.matmul(x, self.weights['h'+str(i)]), self.biases['b'+str(i)])
        output = tf.matmul(x, self.weights['out']) + self.biases['out']
        return output
