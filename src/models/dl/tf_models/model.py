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
        self.loss = None
        self.train_op = None
        self.accuracy = None
        self.task_type = task_type

        # io paras
        self.input = args.input
        self.output = args.output

        # hyper paras of model
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.dropout = args.dropout
        self.lr = args.lr
        self.train_steps = self.count_tfrecord(self.input) // self.batch_size

        # model paras
        self.layers = [int(i) for i in args.layers.split(',')]
        self.nclass = args.nclass
        self.input_size = int(self.layers[0])

        # state record during model learning
        self.global_step = tf.Variable(0, trainable=False)
        self.display_step = args.display_step
        # self.init_net()
        # if self.task_type == 'regression':
        #     self.nclass = 1
        #     self.loss, self.train_op = self.build_model()
        # else:
        #     self.loss, self.train_op, self.accuracy = self.build_model()

    @abc.abstractmethod
    def init_net(self):
        """与下面的网络结构相对应，是下面网络结构中的权重矩阵定义与数据输入定义，
        修改下面网络结构时，此函数中对应的权重矩阵也要对应的修改"""
        pass

    @abc.abstractmethod
    def forward(self, x):
        """
        网络结构，qian向传播，可以替换为其他任意构造的网络结构
        :param x: input tensor
        :return:
        """
        pass

    def build_model(self):
        """构建模型，定义损失函数，模型优化器，模型度量等算子"""
        y_hat = self.forward(self.X)
        if self.task_type is None or self.task_type == 'classification':
            self.out = tf.nn.softmax(logits=y_hat, name='y_sm')
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=self.Y))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            corr_pred = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(corr_pred, tf.float32))
            return self.loss, self.train_op, self.accuracy

        if self.task_type == 'regression':
            self.loss = tf.reduce_mean(tf.square(y_hat - self.Y))
            # loss = tf.reduce_mean(tf.square(y_hat - self.Y), keep_dims=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            self.out = y_hat
            return self.loss, self.train_op

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

    def predict(self, sess, x):     # TODO
        """
        模型预测
        :param sess:
        :param x: 特征
        :return: y_hat
        """
        import numpy as np
        result = sess.run([self.out], feed_dict={
            self.X: x
        })
        return np.array(np.reshape(result, [-1, self.nclass]))

    def eval(self, sess, x, y):
        """
        模型评估, don't execute self.train_op in sess.run()
        :param sess:
        :param x:
        :param y:
        :return:
        """
        if self.task_type is None or self.task_type == 'classification':
            loss, acc = sess.run(
                [self.loss, self.accuracy], feed_dict={
                    self.X: x,
                    self.Y: y
                })
            return loss, acc
        if self.task_type == 'regression':
            loss = sess.run(
                [self.loss], feed_dict={
                    self.X: x,
                    self.Y: y
                })
            return loss

    def save_model(self, sess):
        """
        模型保存, used by tf.train.Saver()
        :param sess:
        """
        saver = tf.train.Saver()
        saver.save(
            sess,
            save_path=os.path.join(self.output, self.__class__.__name__.lower()) + '.ckpt',
            # global_step=self.global_step
        )

    def save_model_pb(self, sess, input_feed, output_feed):
        """
        model save, used by tf.saved_model
        :param sess:
        :param input_feed: a map, such as {"input": x, 'keep_prob': keep_prob}
        :param output_feed: a map, such as  {"output": y_conv}
        :return:
        """
        if tf.gfile.Exists(self.output+'/saved_model'):
            tf.gfile.DeleteRecursively(self.output+'/saved_model')
        tf.compat.v1.saved_model.simple_save(
            sess,
            self.output+'/saved_model',
            inputs=input_feed,
            outputs=output_feed
        )

    def restore_model(self, sess):
        """
        模型恢复
        :param sess:
        """
        saver = tf.train.Saver()
        saver.restore(sess, save_path=self.output)

    def load_model_pb(self, sess, is_saved_model=False):
        """
        load ckpt_to_pb model with tf.GraphDef(), saved_model pb with tf.saved_model
        :param sess:
        :param model_file:
        :param model_format:
        :return:
        """
        if not is_saved_model:
            model_file = os.path.join(
                self.output, '{}model.pb'.format(self.__class__.__name__.lower())
            )
            with tf.gfile.FastGFile(model_file, 'rb') as fd:
                # 导入图
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(fd.read())
                sess.graph.as_default()
                tf.import_graph_def(graph_def, name='')
                # return graph_def  # make no sense
        else:
            tf.compat.v1.saved_model.loader.load(
                sess, ["serve"], os.path.join(self.output, 'saved_model')
            )

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

    def make_train_batch(self, tfrecord_files, record_type='mnist'):
        """
        TensorFlow训练数据持久化格式，通过加载tfrecord格式文件，将训练样本分批处理与训练
        :param tfrecord_files:
        :return:
        """
        dataset = tf.data.TFRecordDataset(tfrecord_files)
        dataset = dataset.map(lambda x: self.parse_tfrecord(x, record_type)).batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        batch_x, batch_y = iterator.get_next()
        return batch_x, batch_y

    def make_test_batch(self, tfrecord_files, record_type='mnist', size=256):
        """
        TensorFlow训练数据持久化格式，通过加载tfrecord格式文件，方便模型测试
        :param tfrecord_files:
        :param size:
        :return:
        """
        dataset = tf.data.TFRecordDataset(tfrecord_files)
        if size is None or size == 0:
            dataset = dataset.map(lambda x: self.parse_tfrecord(x, record_type))
        else:
            dataset = dataset.map(lambda x: self.parse_tfrecord(x, record_type)).batch(size)
        iterator = dataset.make_one_shot_iterator()
        batch_x, batch_y = iterator.get_next()
        return batch_x, batch_y

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
