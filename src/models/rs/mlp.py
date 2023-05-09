"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-12-11 下午5:11
# @FileName: mlp.py
# @Email   : quant_master2000@163.com
======================
"""
import tensorflow as tf
import json
from src.models.rs.rs_model import Model


class MLP(Model):
    def __init__(self, args, task_type):
        Model.__init__(self, args, task_type)
        self.model_type = None
        self.warm_start_from = None
        self.fe_name2fe_size = {}
        self.input_size = 0
        self.deep_columns = self.get_feature_columns()

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
                feat_column = tf.feature_column.categorical_column_with_identity(
                    fe_name, num_buckets=num_buckets, default_value=d_value
                )
                feat_column = tf.feature_column.indicator_column(feat_column)   # trans to one-hot
                size = num_buckets
            elif fe_tran == 'categorical_column_with_vocabulary_list':
                # category feat with finite feat values, which values can't too much
                vocab_list = item.get('vocab_list')
                feat_column = tf.feature_column.categorical_column_with_vocabulary_list(fe_name, vocab_list, fe_type)
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
            deep_columns.append(feat_column)
        return deep_columns

    def model_fn(self, features, labels, mode, params=None):
        labels = tf.reshape(labels, shape=[-1, 1])
        input_layer = tf.feature_column.input_layer(features, self.deep_columns)
        for unit in self.layers:
            input_layer = tf.layers.dense(
                input_layer,
                unit,
                activation=tf.nn.relu
            )
            input_layer = tf.layers.dropout(
                input_layer,
                rate=self.dropout,
                training=mode == tf.estimator.ModeKeys.TRAIN
            )
        if self.nclass == 2:
            logits = tf.layers.dense(
                input_layer,
                self.nclass-1,
                activation=None,
                bias_initializer=None
            )
        else:
            # n > 2 classification
            logits = tf.layers.dense(
                input_layer,
                self.nclass,
                activation=None,
                bias_initializer=None
            )

        # predict
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

        # train
        if mode == tf.estimator.ModeKeys.TRAIN:
            if self.nclass == 2:
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
                train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
                train_op = optimizer.minimize(loss, global_step=self.global_step)
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op
            )

        # evaluate
        if self.nclass == 2:
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
            preds = tf.nn.sigmoid(logits)
            auc = tf.metrics.auc(labels, tf.sigmoid(preds))
            metrics = {'auc': auc}
        else:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
            preds = tf.nn.softmax(logits)
            p = tf.metrics.precision(labels, preds)
            r = tf.metrics.recall(labels, preds)
            f1 = 2 * p * r / (p + r)
            metrics = {'f1': f1}

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=metrics
        )

    # def input_fn(self, data_file, batch_size, is_train=True):
    #     """估算器的输入函数."""
    #     train = pd.read_csv(data_file).dropna()
    #     # train, test = train_test_split(df, test_size=test_size, random_state=1)
    #     train_y = train.pop("label")
    #     train_y = train_y.astype(np.float32)
    #     # train_to_dict = train.to_dict(orient='list')
    #     # train_x = {key: train_to_dict.get(key) for key in self.feat_map}
    #     if is_train:
    #         # from_tensor_slices 从内存引入数据
    #         dataset = tf.data.Dataset.from_tensor_slices((train.to_dict(orient='list'), train_y))
    #     else:
    #         dataset = tf.data.Dataset.from_tensor_slices(train.to_dict(orient='list'))
    #
    #     if is_train:  # 对数据进行乱序操作
    #         dataset = dataset.shuffle(buffer_size=batch_size)  # 越大shuffle程度越大
    #     dataset = dataset.batch(batch_size).repeat(self.epochs).prefetch(1)  # 预取数据,buffer_size=1 在多数情况下就足够了
    #     return dataset
