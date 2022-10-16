"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-9-19 下午10:38
# @FileName: alex_net.py
# @Email   : quant_master2000@163.com
======================
"""
from src.models.dl.tf_models.cnn import CnnModel
import tensorflow as tf


class AlexNet(CnnModel):
    """AlexNet Convolution Neural Network"""
    def __init__(self, args, task_type=None):
        CnnModel.__init__(self, args, task_type)
        self.init_net()
        if self.task_type == 'regression':
            self.loss, self.train_op = self.build_model()
        else:
            self.loss, self.train_op, self.accuracy = self.build_model()

    def init_net(self):
        self.X = tf.placeholder(tf.float32, [None, self.input_size])
        self.Y = tf.placeholder(tf.float32, [None, self.nclass])
        # Store layers weight & bias
        self.weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.truncated_normal([3, 3, 1, 64], dtype=tf.float32, stddev=0.1)),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=0.1)),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc3': tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32, stddev=0.1)),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'fc1': tf.Variable(tf.truncated_normal([4 * 4 * 256, 1024], dtype=tf.float32, stddev=0.1)),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'fc2': tf.Variable(tf.truncated_normal([1024, 1024], dtype=tf.float32, stddev=0.1)),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.Variable(tf.truncated_normal([1024, self.nclass], dtype=tf.float32, stddev=0.1))
        }

        # self.biases = {
        #     'bc1': tf.Variable(tf.random_normal([64])),
        #     'bc2': tf.Variable(tf.random_normal([128])),
        #     'bc3': tf.Variable(tf.random_normal([256])),
        #     'fb1': tf.Variable(tf.random_normal([1024])),
        #     'fb2': tf.Variable(tf.random_normal([1024])),
        #     'out': tf.Variable(tf.random_normal([self.num_classes]))
        # }

        # better than above bias initial method
        self.biases = {
            'bc1': tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True),
            'bc2': tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True),
            'bc3': tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True),
            'fb1': tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32), trainable=True),
            'fb2': tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32), trainable=True),
            'out': tf.Variable(tf.constant(0.0, shape=[self.nclass], dtype=tf.float32), trainable=True)
        }

    def forward(self, x):
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        conv1 = self.conv2d(x, self.weights['wc1'], self.biases['bc1'])
        lrn1 = tf.nn.lrn(
            conv1, depth_radius=4, bias=1, alpha=0.001 / 9, beta=0.75
        )
        pool1 = tf.nn.max_pool(
            lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME'
        )

        conv2 = self.conv2d(pool1, self.weights['wc2'], self.biases['bc2'])
        lrn2 = tf.nn.lrn(
            conv2, depth_radius=4, bias=1, alpha=0.001 / 9, beta=0.75
        )
        pool2 = tf.nn.max_pool(
            lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME'
        )

        conv5 = self.conv2d(pool2, self.weights['wc3'], self.biases['bc3'])
        pool5 = tf.nn.max_pool(
            conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME'
        )

        flat = tf.reshape(pool5, [-1, 4 * 4 * 256])
        fc1 = tf.nn.relu(tf.matmul(flat, self.weights['fc1']) + self.biases['fb1'])
        fc1 = tf.nn.dropout(fc1, keep_prob=self.dropout)
        fc2 = tf.nn.relu(tf.matmul(fc1, self.weights['fc2']) + self.biases['fb1'])
        fc2 = tf.nn.dropout(fc2, keep_prob=self.dropout)
        out = tf.nn.xw_plus_b(fc2, self.weights['out'], self.biases['out'])
        return out

    # def forward(self, x):
    #     x = tf.reshape(x, shape=[-1, 28, 28, 1])
    #     with tf.name_scope('conv1') as scope:
    #         kernel = tf.Variable(
    #             tf.truncated_normal([3, 3, 1, 64], dtype=tf.float32, stddev=0.1), name='weights'
    #         )
    #         biases = tf.Variable(
    #             tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases'
    #         )
    #         conv1 = self.conv2d(x, kernel, biases, name=scope)
    #         lrn1 = tf.nn.lrn(
    #             conv1, depth_radius=4, bias=1, alpha=0.001/9, beta=0.75, name='lrn1'
    #         )
    #         pool1 = tf.nn.max_pool(
    #             lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1'
    #         )
    #         # pool1 = self.maxpool2d(lrn1, 2, name='pool1')
    #     # self.check_shape(pool1)
    #
    #     with tf.name_scope('conv2') as scope:
    #         kernel = tf.Variable(
    #             tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=0.1), name='weights'
    #         )
    #         biases = tf.Variable(
    #             tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases'
    #         )
    #         conv2 = self.conv2d(pool1, kernel, biases, name=scope)
    #         lrn2 = tf.nn.lrn(
    #             conv2, depth_radius=4, bias=1, alpha=0.001 / 9, beta=0.75, name='lrn2'
    #         )
    #         pool2 = tf.nn.max_pool(
    #             lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2'
    #         )
    #         # pool2 = self.maxpool2d(lrn2, 2, name='pool2')
    #         # self.check_shape(pool2)
    #
    #     # with tf.name_scope('conv3') as scope:
    #     #     kernel = tf.Variable(
    #     #         tf.truncated_normal([], dtype=tf.float32, stddev=0.1), name='weights'
    #     #     )
    #     #     conv = tf.nn.conv2d(pool2, kernel, [], padding='SAME')
    #     #     biases = tf.Variable(
    #     #         tf.constant(0.0, shape=[], dtype=tf.float32), trainable=True, name='biases'
    #     #     )
    #     #     bias = tf.nn.bias_add(conv, biases)
    #     #     conv3 = tf.nn.relu(bias, name=scope)
    #     #
    #     # with tf.name_scope('conv4') as scope:
    #     #     kernel = tf.Variable(
    #     #         tf.truncated_normal([], dtype=tf.float32, stddev=0.1), name='weights'
    #     #     )
    #     #     conv = tf.nn.conv2d(conv3, kernel, [], padding='SAME')
    #     #     biases = tf.Variable(
    #     #         tf.constant(0.0, shape=[], dtype=tf.float32), trainable=True, name='biases'
    #     #     )
    #     #     bias = tf.nn.bias_add(conv, biases)
    #     #     conv4 = tf.nn.relu(bias, name=scope)
    #
    #     with tf.name_scope('conv5') as scope:
    #         kernel = tf.Variable(
    #             tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32, stddev=0.1), name='weights'
    #         )
    #         biases = tf.Variable(
    #             tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases'
    #         )
    #         conv5 = self.conv2d(pool2, kernel, biases, name=scope)
    #         pool5 = tf.nn.max_pool(
    #             conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5'
    #         )
    #         # pool5 = self.maxpool2d(conv5, 2, name='pool5')
    #         # self.check_shape(pool5)
    #
    #     with tf.name_scope('fc6') as scope:
    #         kernel = tf.Variable(
    #             tf.truncated_normal([4*4*256, 1024], dtype=tf.float32, stddev=0.1), name='weights'
    #         )
    #         biases = tf.Variable(
    #             tf.constant(0.0, shape=[1024], dtype=tf.float32), trainable=True, name='biases'
    #         )
    #         flat = tf.reshape(pool5, [-1, 4*4*256])
    #         fc = tf.nn.relu(tf.matmul(flat, kernel)+biases, name=scope)
    #         fc6 = tf.nn.dropout(fc, keep_prob=self.dropout)
    #         # self.check_shape(fc6)
    #
    #     with tf.name_scope('fc7') as scope:
    #         kernel = tf.Variable(
    #             tf.truncated_normal([1024, 1024], dtype=tf.float32, stddev=0.1), name='weights'
    #         )
    #         biases = tf.Variable(
    #             tf.constant(0.0, shape=[1024], dtype=tf.float32), trainable=True, name='biases'
    #         )
    #
    #         fc = tf.nn.relu(tf.matmul(fc6, kernel)+biases, name=scope)
    #         fc7 = tf.nn.dropout(fc, keep_prob=self.dropout)
    #         # self.check_shape(fc7)
    #
    #     with tf.name_scope('fc8') as scope:
    #         kernel = tf.Variable(
    #             tf.truncated_normal([1024, 10], dtype=tf.float32, stddev=0.1), name='weights'
    #         )
    #         biases = tf.Variable(
    #             tf.constant(0.0, shape=[10], dtype=tf.float32), trainable=True, name='biases'
    #         )
    #
    #         out = tf.nn.xw_plus_b(fc7, kernel, biases, name=scope)
    #         # self.check_shape(out)
    #     return out
