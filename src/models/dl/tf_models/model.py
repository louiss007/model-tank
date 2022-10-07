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


class Model(object):
    """Base Class of NN Models"""
    def __init__(self, args, task_type=None):
        """
        模型构建初始化
        :param args:
        :param task_type:
        """
        self.task_type = task_type
        # hyper paras of model
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.dropout = args.dropout
        self.learning_rate = args.learning_rate

        # model paras
        self.input_size = args.input_size
        self.hidden_layers = args.hidden_layers
        self.nclass = args.nclass
        if self.task_type == 'regression':
            self.nclass = 1

        # other paras
        self.global_step = tf.Variable(0, trainable=False)

    def init_net(self):
        """与下面的网络结构相对应，是下面网络结构中的权重矩阵定义与数据输入定义，
        修改下面网络结构时，此函数中对应的权重矩阵也要对应的修改"""
        pass

    def forward(self, x):
        """
        网络结构，正向传播输出，可以替换为其他任意构造的网络结构
        :param x: input tensor
        :return:
        """
        pass

    def build_model(self):
        """构建模型，损失函数，优化器，学习算子等"""
        pass

    def fit(self, sess, batch_x, batch_y):
        """
        learn model, 训练模型
        :param sess:
        :param batch_x:
        :param batch_y:
        :return:
        """
        pass

    def predict(self, sess, x, y):
        """
        模型预测
        :param sess:
        :param x: 特征
        :param y: y is None
        :return: y_hat
        """
        pass

    def eval(self):
        pass

    def save_model(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path + '.ckpt', global_step=self.global_step)

    def restore_model(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)

    # ***********
    def parse_tfrecord(self, tfrecord):
        features = {}
        for col in self.feat_cols:
            features.setdefault(col, tf.FixedLenFeature([], tf.float32))
        features.setdefault(self.target, tf.FixedLenFeature([], tf.float32))
        example = tf.parse_single_example(tfrecord, features=features)
        x = [example[col] for col in self.feat_cols]
        y = [example[self.target]]
        return x, y

    def make_one_batch(self, tfrecord_files):
        dataset = tf.data.TFRecordDataset(tfrecord_files)
        dataset = dataset.map(self.parse_tfrecord).batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        batch_x, batch_y = iterator.get_next()
        return batch_x, batch_y

    def make_batch(self, tfrecord_files):
        dataset = tf.data.TFRecordDataset(tfrecord_files)
        dataset = dataset.map(self.parse_tfrecord).batch(256)
        iterator = dataset.make_one_shot_iterator()
        batch_x, batch_y = iterator.get_next()
        return batch_x, batch_y
