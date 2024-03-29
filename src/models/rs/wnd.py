"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-9-19 下午11:20
# @FileName: wnd.py
# @Email   : quant_master2000@163.com
======================
"""
import tensorflow as tf
import json
import itertools
import pandas as pd
from src.models.rs.dnn_linear_combined import Net
from src.models.rs.rs_model import Model


"""
TODO:
how does train wnd model with two learning algorithms ftrl and adam in tf.Estimator?
"""


class WideAndDeep(Model):

    def __init__(self, args, task_type):
        Model.__init__(self, args, task_type)
        self.model_type = None
        self.warm_start_from = None
        self.fe_name2fe_size = {}
        self.input_size = 0
        self.wide_columns, self.deep_columns = self.get_feature_columns()
        self.net = Net(self.fe_name2fe_size, self.layers, self.nclass, mode=None)

    def get_feature_columns(self, config_file=None):
        feat_type_map = {
            'int': tf.int32,
            'int64': tf.int64,
            'float': tf.float32,
            'float64': tf.float64,
            'string': tf.string
        }

        if config_file is None:
            feat_list = json.load(open('../src/models/rs/feature_columns.conf'))
        else:
            feat_list = json.load(open(config_file))

        base_columns = []
        tmp_cross_columns = []
        deep_columns = []
        for item in feat_list:
            col_type = item.get('col_type', 'dense')
            fe_name = item.get('fe_name')
            fe_type = feat_type_map.get(item.get('fe_type'), tf.string)
            fe_tran = item.get('fe_tran')
            if fe_tran == 'numeric_column':
                # numeric feat
                feat_column = tf.feature_column.numeric_column(fe_name, shape=(1, ), dtype=fe_type)
                size = 1
            elif fe_tran == 'categorical_column_with_identity':
                # category feat with encoded, feat value has been int
                num_buckets = item.get('num_buckets')
                d_value = item.get('default_value', -1)
                print('d_type_identity:{}, {}, {}'.format(fe_name, d_value, type(d_value)))
                feat_column = tf.feature_column.categorical_column_with_identity(
                    fe_name, num_buckets=num_buckets, default_value=d_value
                )

                if col_type == 'sparse-onehot':
                    tmp_cross_columns.append(feat_column)

                feat_column = tf.feature_column.indicator_column(feat_column)   # trans to one-hot
                size = num_buckets
            elif fe_tran == 'categorical_column_with_vocabulary_list':
                # category feat with finite feat values, which values can't too much
                vocab_list = item.get('vocab_list')
                feat_column = tf.feature_column.categorical_column_with_vocabulary_list(fe_name, vocab_list, fe_type)

                if col_type == 'sparse-onehot':
                    tmp_cross_columns.append(feat_column)

                feat_column = tf.feature_column.indicator_column(feat_column)   # trans to one-hot
                size = len(vocab_list)
            else:
                # fe_tran == 'categorical_column_with_hash_bucket'
                # category feat with too many feat values
                bucket_size = item.get('hash_bucket_size', 100000)
                feat_column = tf.feature_column.categorical_column_with_hash_bucket(fe_name, bucket_size, fe_type)
                # trans to embedding
                feat_column = tf.feature_column.embedding_column(feat_column, dimension=self.embedding_size)
                size = self.embedding_size
            # self.feat_map.setdefault(fe_name, feat_column)
            self.fe_name2fe_size.setdefault(fe_name, size)
            self.input_size += size

            if col_type == 'sparse-onehot':
                base_columns.append(feat_column)
            deep_columns.append(feat_column)

        cross_feats = list(itertools.combinations(tmp_cross_columns, 2))
        cross_columns = []
        for kv in cross_feats:
            feat_1 = kv[0]
            feat_2 = kv[1]
            feat_1_size = self.fe_name2fe_size.get(feat_1.name)
            feat_2_size = self.fe_name2fe_size.get(feat_2.name)

            cross_feat = tf.feature_column.crossed_column([feat_1, feat_2], feat_1_size * feat_2_size)

            self.fe_name2fe_size.setdefault(cross_feat.name, feat_1_size * feat_2_size)

            cross_feat = tf.feature_column.indicator_column(cross_feat)
            cross_columns.append(cross_feat)
        wide_columns = base_columns + cross_columns
        return wide_columns, deep_columns

    # sparse
    def get_feat_embedding(self):
        pass

    # one-hot
    def get_feat_onehot(self):
        pass

    # numeric
    def get_feat_numeric(self):
        pass

    def input_fn(self, data_file, batch_size, is_train=True):
        """估算器的输入函数."""
        train = pd.read_csv(data_file).dropna()
        # train, test = train_test_split(df, test_size=test_size, random_state=1)
        train_y = train.pop("label")
        # train_to_dict = train.to_dict(orient='list')
        # train_x = {key: train_to_dict.get(key) for key in self.feat_map}
        if is_train:
            # from_tensor_slices 从内存引入数据
            dataset = tf.data.Dataset.from_tensor_slices((train.to_dict(orient='list'), train_y))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(train.to_dict(orient='list'))

        if is_train:  # 对数据进行乱序操作
            dataset = dataset.shuffle(buffer_size=batch_size)  # 越大shuffle程度越大
        dataset = dataset.batch(batch_size).prefetch(1)  # 预取数据,buffer_size=1 在多数情况下就足够了
        return dataset

    def model_fn(self, features, labels, mode, params=None):
        wide_logit_fn = self.build_wide_logit_fn(self.wide_columns, mode)
        wide_logits = wide_logit_fn(features)
        print('wide_logits: {}'.format(wide_logits))

        deep_logit_fn = self.build_deep_logit_fn(self.deep_columns, mode)
        deep_logits = deep_logit_fn(features)

        if self.model_type == 'wide':
            logits = wide_logits
        elif self.model_type == 'deep':
            logits = deep_logits
        else:
            # default model is wide and deep
            logits = wide_logits + deep_logits

        if mode == tf.estimator.ModeKeys.PREDICT:
            if self.nclass == 2:
                prob = tf.nn.sigmoid(logits, name='y')
            else:
                prob = tf.nn.softmax(logits, name='y_sm')
            predictions = {
                'prob': prob
            }
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions
            )

        if mode == tf.estimator.ModeKeys.TRAIN:
            if self.nclass == 2:
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
                train_op = optimizer.minimize(loss, global_step=self.global_step)
            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
                train_op = optimizer.minimize(loss, global_step=self.global_step)
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op
            )
        if self.nclass == 2:
            preds = tf.nn.sigmoid(logits)
            auc = tf.metrics.auc(labels, tf.sigmoid(preds))
            metrics = {'auc': auc}
        else:
            preds = tf.nn.softmax(logits)
            p = tf.metrics.precision(labels, preds)
            r = tf.metrics.recall(labels, preds)
            f1 = 2 * p * r / (p + r)
            metrics = {'f1': f1}
        
        return tf.estimator.EstimatorSpec(
            mode=mode,
            eval_metric_ops=metrics
        )

    def build_wide_logit_fn(self, feature_columns, mode):

        def wide_logit_fn(features):
            wide_logits = self.net.wide_net(features, feature_columns)
            return wide_logits

        return wide_logit_fn

    def build_deep_logit_fn(self, feature_columns, mode):

        def deep_logit_fn(features):
            deep_logits = self.net.deep_net(features, feature_columns)
            return deep_logits

        return deep_logit_fn
