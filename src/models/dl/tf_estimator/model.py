"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-10-25 下午10:41
# @FileName: model.py
# @Email   : quant_master2000@163.com
======================
"""
import tensorflow as tf
import numpy as np
import abc
import os


class Model(object):
    """ Base Class of NN model built with tf.Estimator """
    def __init__(self, args, task_type=None):
        """
        模型构建初始化
        :param args:
        :param task_type:
        """
        self.task_type = task_type

        # io paras
        self.input = args.input
        self.output = args.output

        # hyper paras of model
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.dropout = args.dropout
        self.lr = args.lr
        self.train_steps = self.count_tfrecord(
            os.path.join(self.input, 'train.tfrecord')
        ) // self.batch_size

        # model paras
        self.layers = [int(i) for i in args.layers.split(',')]
        self.nclass = args.nclass
        self.input_size = int(self.layers[0])

        # state record during model learning
        self.global_step = tf.Variable(0, trainable=False)
        self.display_step = args.display_step

    @abc.abstractmethod
    def model_fn(self, features, labels, mode):
        pass

    def export_tf_model(self, classifier, feed_dict):
        feature_map = dict()
        for key, value in feed_dict:
            feature_map[key] = tf.placeholder(
                dtype=tf.int64, shape=[None, value], name=key
            )
        recevier_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_map)
        model_dir = classifier.export_saved_model(self.output, recevier_fn)
        print("export tf model to %s!" % model_dir)

    def load_model(self):
        pass

    def load_data(self, tfrecord_files, record_type='mnist', is_train=True):
        """
        TensorFlow训练数据持久化格式，通过加载tfrecord格式文件，return np.array
        :param tfrecord_files:
        :return: np.array
        """
        dataset = tf.data.TFRecordDataset(tfrecord_files)
        if is_train:
            dataset = dataset.map(
                lambda x: self.parse_tfrecord(x, record_type)
            ).shuffle(self.batch_size*2).batch(self.batch_size)
            dataset = dataset.repeat(self.epochs)
        else:
            dataset = dataset.map(
                lambda x: self.parse_tfrecord(x, record_type)
            ).batch(self.batch_size)
        return dataset

    def parse_tfrecord(self, tfrecord, record_type='mnist'):
        """
        通过解析TFrecord格式样本，返回x和y
        :param tfrecord: tfrecord格式样本
        :param record_type: cv/nlp/rs/mnist
        :return:
        """
        if record_type == 'mnist':
            # features = {
            #     'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.string])),
            #     'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.string])),
            #     'num1': tf.train.Feature(float_list=tf.train.FloatList(value=[tf.float32])),
            #     'num2': tf.train.Feature(int64_list=tf.train.Int64List(value=[tf.int64]))
            # }
            features = {
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string),
                'num1': tf.FixedLenFeature([], tf.float32),
                'num2': tf.FixedLenFeature([], tf.int64)
            }
            example = tf.parse_single_example(tfrecord, features=features)
            image = tf.decode_raw(example['image'], tf.float32)
            label = tf.decode_raw(example['label'], tf.float32)
            # num1 = example['num1']
            # num2 = example['num2']
            image = tf.reshape(image, shape=[28, 28, 1])
            image = tf.reshape(image, shape=[784])
            label = tf.reshape(label, shape=[self.nclass])
            return image, label
        elif record_type == 'cv':
            pass
        elif record_type == 'nlp':
            pass
        elif record_type == 'rs':
            features = {}
            for col in self.feat_cols:
                features.setdefault(col, tf.FixedLenFeature([], tf.float32))
            features.setdefault(self.target, tf.FixedLenFeature([], tf.float32))
            example = tf.parse_single_example(tfrecord, features=features)
            x = [example[col] for col in self.feat_cols]
            y = [example[self.target]]
            return x, y
        else:
            return None, None

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
    def count_tfrecord(tfrecord_file):
        count = 0
        for _ in tf.io.tf_record_iterator(tfrecord_file):
            count += 1
        print("数据{} 的样本条数为\t{}".format(tfrecord_file, count))
        return count
