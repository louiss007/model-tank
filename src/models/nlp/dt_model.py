"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-10-21 下午11:29
# @FileName: dt_model.py
# @Email   : quant_master2000@163.com
======================
"""
import tensorflow as tf
import abc
import random


class DtModel(object):
    """ Base Class of Double Tower Models """
    def __init__(self, args, task_type):
        self.task_type = task_type
        self.is_train = args.is_train

        self.loss = None
        self.train_op = None
        self.accuracy = None

        # config para for distributed training
        self.is_cluster = args.is_cluster
        self.ps_master = args.ps_master
        self.ps_worker = args.ps_worker
        self.job_name = args.job_name
        self.task_index = args.task_index

        # io paras
        self.input = args.input
        self.output = args.output

        # hyper paras of model
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        # self.dropout = args.dropout
        self.lr = args.lr
        self.train_steps = self.count_tfrecord(self.input) // self.batch_size

        # model paras
        self.layers = [int(i) for i in args.layers.split(',')]
        # self.nclass = args.nclass
        # self.input_size = int(self.layers[0])

        # length is token-based
        self.query_max_length = args.query_max_length
        self.doc_max_length = args.doc_max_length
        self.token_vocab_size = args.token_vocab_size
        self.token_embedding_size = args.token_embedding_size
        self.neg_num = args.neg_num

        # state record during model learning
        self.global_step = tf.Variable(0, trainable=False)
        self.display_steps = args.display_steps

    def query_encoder(self, features):
        # 输入shape: [batch_size, sentence_size]
        query_input = tf.reshape(features["query"], [-1, self.query_max_length])

        query_embedding = self.word_embedding(
            query_input,
            "char_embedding"
        )

        query_encode = self.text_embedding(
            query_embedding,
            self.query_max_length,  # 60
            "text_embedding",
            None,
        )

        query_encode = tf.layers.dense(
            query_encode,
            units=self.layers[-1],
            activation=tf.nn.tanh,
            name="query_encode"
        )

        query_encode = tf.nn.l2_normalize(query_encode)
        return query_encode

    def doc_encoder(self, features):
        # 输入shape: [batch_size, sentence_size]
        doc_input = tf.reshape(features["doc"], [-1, self.doc_max_length])

        doc_embedding = self.word_embedding(
            doc_input,
            "char_embedding",
            True
        )

        doc_encode = self.text_embedding(
            doc_embedding,
            self.doc_max_length,
            "text_embedding",
            True if tf.estimator.ModeKeys.TRAIN == 'train' else tf.AUTO_REUSE
        )

        doc_encode = tf.layers.dense(
            doc_encode,
            units=self.layers[-1],
            activation=tf.nn.tanh,
            name="doc_encode"
        )

        doc_encode = tf.nn.l2_normalize(doc_encode)
        return doc_encode

    def text_encoder(self, features, feat_name, text_max_length, reuse=None):
        """integrate query encoder and doc encoder into one func implement"""
        # 输入shape: [batch_size, sentence_size]
        print("======features======", features)
        doc_input = tf.reshape(features[feat_name], [-1, text_max_length])

        doc_embedding = self.word_embedding(    # 512*60*128
            doc_input,
            "char_embedding",
            reuse
        )

        doc_encode = self.text_embedding(       # 512*512
            doc_embedding,
            text_max_length,
            "text_embedding",
            reuse
        )

        doc_encode = tf.layers.dense(
            doc_encode,
            units=self.layers[-1],
            activation=tf.nn.tanh,
            name="{}_encode".format(feat_name)
        )

        doc_encode = tf.nn.l2_normalize(doc_encode)
        return doc_encode

    def word_embedding(self, token_seq, scope_name, reuse=None):
        with tf.variable_scope(scope_name, reuse=reuse):
            embedding_matrix = tf.Variable(
                tf.truncated_normal((self.token_vocab_size, self.token_embedding_size))
            )
            embedding = tf.nn.embedding_lookup(embedding_matrix, token_seq, name=scope_name)
            embedding = tf.nn.tanh(embedding)
            return embedding

    @abc.abstractmethod
    def text_embedding(self, text, text_length, scope_name, reuse=None):
        pass

    def model_fn(self, features, labels, mode, params):
        print("============{} task===========".format(mode))
        # 512*64 batch_size*rnn_hidden_size
        query_encoder = self.text_encoder(features, 'query', self.query_max_length)
        doc_encoder = self.text_encoder(features, 'doc', self.doc_max_length, True)

        # Predict
        if mode == tf.estimator.ModeKeys.PREDICT:
            pred = {"doc_encode": doc_encoder}
            ex_outputs = {"pred": tf.estimator.export.PredictOutput(outputs=pred)}
            predict_est_spec = tf.estimator.EstimatorSpec(
                mode,
                predictions=pred,
                export_outputs=ex_outputs
            )
            return predict_est_spec

        with tf.name_scope("fd-rotate"):
            query_encoder_fd = tf.tile(query_encoder, [self.neg_num + 1, 1])  # [26112, 64]
            tmp = tf.tile(doc_encoder, [1, 1])
            doc_encoder_fd = doc_encoder
            for i in range(self.neg_num):
                rand = random.randint(1, self.batch_size + i) % self.batch_size
                # rand = int((random.random() + i) * self.batch_size / self.neg_num)
                s1 = tf.slice(tmp, [rand, 0], [self.batch_size - rand, -1])  # wrong in eval stage
                s2 = tf.slice(tmp, [0, 0], [rand, -1])
                doc_encoder_fd = tf.concat([doc_encoder_fd, s1, s2], axis=0)

        with tf.name_scope("cosine_sim"):
            query_norm = tf.tile(   # [26112, 1]
                tf.sqrt(tf.reduce_sum(tf.square(query_encoder), axis=1, keepdims=True)),
                [self.neg_num + 1, 1]
            )
            doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_encoder_fd), axis=1, keepdims=True))  # [26112, 1]

            prod = tf.reduce_sum(tf.multiply(query_encoder_fd, doc_encoder_fd), axis=1, keepdims=True)
            norm_prod = tf.multiply(query_norm, doc_norm)

            cos_sim = tf.truediv(prod, norm_prod)
            cos_sim_t = tf.transpose(tf.reshape(tf.transpose(cos_sim), [self.neg_num + 1, -1])) * 20  # [512, 51]

        with tf.name_scope("loss"):
            prob = tf.nn.softmax(cos_sim_t)     # [512, 51]
            hit_prob = tf.slice(prob, [0, 0], [-1, 1])
            loss = -tf.reduce_mean(tf.log(hit_prob))    # []
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            # train_op = optimizer.minimize(loss, global_step=self.global_step) # Wrong in tf.Estimator
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())  # OK

        # Train
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_est_spec = tf.estimator.EstimatorSpec(
                mode,
                loss=loss,
                train_op=train_op
            )
            return train_est_spec

        corr_pred = tf.cast(tf.equal(tf.argmax(prob, 1), 0), tf.float32)
        accuracy = tf.reduce_mean(corr_pred)

        # Eval
        eval_est_spec = tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            eval_metric_ops={}
        )
        return eval_est_spec

    def export_tf_model(self, model, feed_dict, model_dir):
        feature_map = dict()
        for key, value in feed_dict.items():
            feature_map[key] = tf.placeholder(
                dtype=tf.int64, shape=[None, value], name=key
            )
        recevier_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_map)
        model_dir = model.export_saved_model(model_dir, recevier_fn)
        print("export tf model to %s!" % model_dir)

    def parse_example(self, tfrecord):
        """
        text preprocessing
        :param tfrecord:
        :return:
        """
        features_map = dict()
        features_map["label"] = tf.io.FixedLenFeature([1], tf.int64)
        features_map["query"] = tf.io.FixedLenFeature([self.query_max_length], tf.int64)
        features_map["doc"] = tf.io.FixedLenFeature([self.doc_max_length], tf.int64)
        features = tf.io.parse_single_example(tfrecord, features=features_map)
        label = features.pop("label")
        return features, label

    def make_train_dataset(self,
                           file_path=None,
                           batch_size=128):
        all_train_files = tf.gfile.Glob(file_path)
        if self.is_cluster:
            # 集群上训练需要切分数据
            train_worker_num = len(self.ps_worker.split(","))
            hash_id = self.task_index if self.job_name == "worker" else train_worker_num - 1
            file_shards = [
                train_file for i, train_file in enumerate(all_train_files)
                if i % train_worker_num == hash_id
            ]
            dataset = tf.data.TFRecordDataset(file_shards)
        else:
            dataset = tf.data.TFRecordDataset(all_train_files)

        dataset = dataset.shuffle(batch_size * 10)
        dataset = dataset.map(self.parse_example, num_parallel_calls=4)
        dataset = dataset.batch(batch_size, drop_remainder=True).repeat(self.epochs).prefetch(1)
        return dataset

    def make_test_dataset(self,
                          file_path=None,
                          batch_size=128):
        test_files = tf.gfile.Glob(file_path)
        dataset = tf.data.TFRecordDataset(test_files)
        dataset = dataset.map(self.parse_example, num_parallel_calls=4)
        dataset = dataset.batch(batch_size).repeat()
        return dataset

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
