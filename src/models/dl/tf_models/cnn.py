"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-9-16 下午8:58
# @FileName: cnn.py
# @Email   : quant_master2000@163.com
======================
"""
from model import Model
import tensorflow as tf


class CnnModel(Model):
    """Convolution Neural Network"""
    def __init__(self, args, task_type=None):
        Model.__init__(self, args, task_type)
        self.height = 28
        self.width = 28
        self.channels = 1
        self.init_net()

    def init_net(self):
        self.X = tf.placeholder(tf.float32, [None, self.layers[0]])
        self.Y = tf.placeholder(tf.float32, [None, self.nclass])
        # Store layers weight & bias
        self.weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([1024, self.nclass]))
        }

        self.biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([self.num_classes]))
        }

        # TODO modify to for loop
        # do something

    def forward(self, x):
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution Layer
        conv1 = self.conv2d(x, self.weights['wc1'], self.biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = self.maxpool2d(conv1, k=2)

        # Convolution Layer
        conv2 = self.conv2d(conv1, self.weights['wc2'], self.biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = self.maxpool2d(conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, self.weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, self.dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out'])
        return out

    def parse_tfrecord(self, tfrecord):
        """

        :param tfrecord:
        :return:
        """
        example = tf.parse_single_example(tfrecord, features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
            'num1': tf.FixedLenFeature([], tf.float32),
            'num2': tf.FixedLenFeature([], tf.int64)
        })
        image = tf.decode_raw(example['image'], tf.float32)
        label = tf.decode_raw(example['label'], tf.float32)
        # num1 = example['num1']
        # num2 = example['num2']
        image = tf.reshape(image, shape=[self.height, self.width, self.channels])
        image = tf.reshape(image, shape=[self.layers[0]])
        label = tf.reshape(label, shape=[self.num_classes])
        return image, label
