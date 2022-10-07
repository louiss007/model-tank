"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-9-16 下午8:49
# @FileName: model.py
# @Email   : quant_master2000@163.com
======================
"""
import tensorflow as tf
import abc
import os


class Model(object):
    """Base Class of NN Models"""
    def __init__(self, args, task_type=None):
        """
        模型构建初始化
        :param args:
        :param task_type:
        """
        self.weights = {}
        self.biases = {}
        self.X = None
        self.Y = None
        self.out = None
        self.task_type = task_type

        # hyper paras of model
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.dropout = args.dropout
        self.learning_rate = args.learning_rate

        # model paras
        self.input_size = args.input_size
        self.layers = args.layers
        self.nclass = args.nclass

        # io paras
        self.input = args.input
        self.output = args.output

        # state record during model learning
        self.global_step = tf.Variable(0, trainable=False)
        if self.task_type == 'regression':
            self.nclass = 1
            self.loss, self.train_op = self.build_model()
        else:
            self.loss, self.train_op, self.accuracy = self.build_model()

    @abc.abstractmethod
    def init_net(self):
        """与下面的网络结构相对应，是下面网络结构中的权重矩阵定义与数据输入定义，
        修改下面网络结构时，此函数中对应的权重矩阵也要对应的修改"""
        pass

    @abc.abstractmethod
    def forward(self, x):
        """
        网络结构，正向传播输出，可以替换为其他任意构造的网络结构
        :param x: input tensor
        :return:
        """
        pass

    def build_model(self):
        """构建模型，损失函数，优化器，学习算子等"""
        y_hat = self.forward(self.X)
        if self.task_type is None or self.task_type == 'classification':
            self.out = tf.nn.softmax(logits=y_hat)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=self.Y))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            train_op = optimizer.minimize(loss, global_step=self.global_step)
            corr_pred = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.Y, 1))
            accuracy = tf.reduce_mean(tf.cast(corr_pred, tf.float32))
            return loss, train_op, accuracy

        if self.task_type == 'regression':
            loss = tf.reduce_mean(tf.square(y_hat - self.Y))
            # loss = tf.reduce_mean(tf.square(y_hat - self.Y), keep_dims=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            train_op = optimizer.minimize(loss, global_step=self.global_step)
            self.out = y_hat
            return loss, train_op

    def fit(self, sess, batch_x, batch_y):
        """
        learn model, 训练模型
        :param sess:
        :param batch_x:
        :param batch_y:
        :return:
        """
        if self.task_type is None or self.task_type == 'classification':
            loss, acc, _, step = sess.run(
                [self.loss, self.accuracy, self.train_op, self.global_step], feed_dict={
                    self.X: batch_x,
                    self.Y: batch_y
                })
            return loss, acc, step
        if self.task_type == 'regression':
            loss, _, step = sess.run(
                [self.loss, self.train_op, self.global_step], feed_dict={
                    self.X: batch_x,
                    self.Y: batch_y
                })
            return loss, step

    def predict(self, sess, x, y):
        """
        模型预测
        :param sess:
        :param x: 特征
        :param y: y is None
        :return: y_hat
        """
        result = sess.run([self.out], feed_dict={
            self.X: x,
            self.Y: y
        })
        return result

    def eval(self):
        """模型评估"""
        pass

    def save_model(self, sess):
        """
        模型保存
        :param sess:
        """
        saver = tf.train.Saver()
        saver.save(
            sess,
            save_path=os.path.join(self.input, self.__name__) + '.ckpt',
            global_step=self.global_step
        )

    def restore_model(self, sess):
        """
        模型恢复
        :param sess:
        """
        saver = tf.train.Saver()
        saver.restore(sess, save_path=self.output)

    def parse_tfrecord(self, tfrecord):
        """

        :param tfrecord:
        :return:
        """
        feature_columns = tfrecord.feature_column
        target_column = tfrecord.target
        features = {}
        for col in feature_columns:
            features.setdefault(col, tf.FixedLenFeature([], tf.float32))
        features.setdefault(target_column, tf.FixedLenFeature([], tf.float32))
        example = tf.parse_single_example(tfrecord, features=features)
        x = [example[col] for col in feature_columns]
        y = [example[target_column]]
        return x, y

    def make_train_batch(self, tfrecord_files):
        """

        :param tfrecord_files:
        :return:
        """
        dataset = tf.data.TFRecordDataset(tfrecord_files)
        dataset = dataset.map(self.parse_tfrecord).batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        batch_x, batch_y = iterator.get_next()
        return batch_x, batch_y

    def make_test_batch(self, tfrecord_files, size=256):
        """

        :param tfrecord_files:
        :param size:
        :return:
        """
        dataset = tf.data.TFRecordDataset(tfrecord_files)
        if size is None or size == 0:
            dataset = dataset.map(self.parse_tfrecord)
        else:
            dataset = dataset.map(self.parse_tfrecord).batch(size)
        iterator = dataset.make_one_shot_iterator()
        batch_x, batch_y = iterator.get_next()
        return batch_x, batch_y

    @staticmethod
    def check_shape(input_tensor):
        """

        :param input_tensor:
        :return:
        """
        print('============tensor shape==============')
        print(input_tensor.op.name, ' ', input_tensor.get_shape().as_list())
        print('============tensor shape==============')
