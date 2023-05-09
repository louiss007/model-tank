"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-11-19 下午11:10
# @FileName: rs_model.py
# @Email   : quant_master2000@163.com
======================
"""
import tensorflow as tf
import abc
import pandas as pd
import numpy as np
import os
"""
rs model build step frequently as follows:
1. input_fn
    parse_example
2. get feature_column embedding
3. embedding concat
4. model_fn
5. export model
"""


class Model(object):

    def __init__(self, args, task_type):
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
        self.loss = None
        self.train_op = None
        self.accuracy = None
        self.task_type = task_type

        # io paras
        self.input = args.input
        self.eval_file = args.eval_file
        self.output = args.output
        self.embedding_size = args.embedding_size

        # hyper paras of model
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.dropout = args.dropout
        self.lr = args.lr
        # self.train_steps = self.count_tfrecord(self.input) // self.batch_size

        # model paras
        self.layers = [int(i) for i in args.layers.split(',')]
        self.nclass = args.nclass
        self.input_size = int(self.layers[0])

        # state record during model learning
        self.global_step = tf.Variable(0, trainable=False)
        self.display_step = args.display_step

    @abc.abstractmethod
    def model_fn(self, features, labels, mode, params=None):
        pass

    def input_fn(self, data_file, batch_size, is_train=True):  # 定义估算器输入函数
        """估算器的输入函数."""
        train = pd.read_csv(data_file).dropna()
        # train, test = train_test_split(df, test_size=test_size, random_state=1)
        train_y = train.pop("label")
        train_y = train_y.astype(np.float32)

        if is_train:
            # from_tensor_slices 从内存引入数据
            dataset = tf.data.Dataset.from_tensor_slices((train.to_dict(orient='list'), train_y))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(train.to_dict(orient='list'))

        if is_train:  # 对数据进行乱序操作
            dataset = dataset.shuffle(buffer_size=batch_size)  # 越大shuffle程度越大
        dataset = dataset.batch(batch_size).repeat(self.epochs).prefetch(1)  # 预取数据,buffer_size=1 在多数情况下就足够了
        return dataset

    def export_tf_model(self, model, features, model_dir):
        example_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features)
        model.export_savedmodel(model_dir, example_input_fn, as_text=True)
        print("export tf model to %s!" % model_dir)

    @staticmethod
    def check_shape(input_tensor):
        """
        检查输入数据形状
        :param input_tensor: 输入张量
        :return:
        """
        print('============tensor shape==============')
        print(input_tensor.op.name, ' ', input_tensor.get_shape().as_list())
        print('============tensor shape==============')

    @staticmethod
    def display_tfrecord(tfrecord_file):
        item = next(tf.io.tf_record_iterator(tfrecord_file))
        print(tf.train.Example.FromString(item))

    @staticmethod
    def count_tfrecord(tfrecord_file):
        count = 0
        for _ in tf.io.tf_record_iterator(tfrecord_file):
            count += 1
        print("数据{} 的样本条数为\t{}".format(tfrecord_file, count))
        return count
