"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-12-14 下午10:50
# @FileName: deepfm_v2.py
# @Email   : quant_master2000@163.com
======================
"""
import tensorflow as tf
from src.models.rs.rs_model import Model


class DeepFm(Model):

    def __init__(self, args, task_type):
        Model.__init__(self, args, task_type)
        self.model_type = None
        self.warm_start_from = None
        self.optimizer = 'Adam'
        self.batch_norm_decay = 0.9
        self.is_batch_norm = True
        self.field_size = 22
        self.feature_size = 10000000    # maximal feature index
        self.l2 = 0.001

    def model_fn(self, features, labels, mode, params=None):
        # init bias, wi, vi and vj
        fm_b = tf.get_variable(name='fm_bias', shape=[1], initializer=tf.constant_initializer(0.0))
        fm_w = tf.get_variable(name='fm_w', shape=[self.feature_size], initializer=tf.glorot_normal_initializer())
        fm_v = tf.get_variable(
            name='fm_v', shape=[self.feature_size, self.embedding_size], initializer=tf.glorot_normal_initializer()
        )

        #
        feat_ids = features['feat_ids']
        feat_ids = tf.reshape(feat_ids, shape=[-1, self.field_size])
        feat_vals = features['feat_vals']
        feat_vals = tf.reshape(feat_vals, shape=[-1, self.field_size])

        # build fm part
        with tf.variable_scope("fm-1st-order"):
            feat_wts = tf.nn.embedding_lookup(fm_w, feat_ids)
            y_wx = tf.reduce_sum(tf.multiply(feat_wts, feat_vals), 1)

        with tf.variable_scope("fm-2nd-order"):
            embeddings = tf.nn.embedding_lookup(fm_v, feat_ids)
            feat_vals = tf.reshape(feat_vals, shape=[-1, self.field_size, 1])
            embeddings = tf.multiply(embeddings, feat_vals)  # vi*xi
            sum_square = tf.square(tf.reduce_sum(embeddings, 1))    # (v1x1 + v2x2 + ... + vnxn)^2
            square_sum = tf.reduce_sum(tf.square(embeddings), 1)    # (v1x1)^2 + (v2x2)^2 + (vnxn)^2
            y_wxx = 0.5 * tf.reduce_sum(tf.subtract(sum_square, square_sum), 1)

        # build deep part
        with tf.variable_scope("deep-part"):
            input_layer = tf.reshape(embeddings, [-1, self.field_size * self.embedding_size])
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
                    self.nclass - 1,
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
            y_d = tf.reshape(logits, [-1])

        with tf.variable_scope("deepfm-out"):
            y_bias = fm_b * tf.ones_like(y_d, dtype=tf.float32)  # None * 1
            y = y_bias + y_wx + y_wxx + y_d
            if self.nclass == 2:
                preds = tf.nn.sigmoid(y, name='y')
            else:
                preds = tf.nn.softmax(y, name='y_sm')

        # predict
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'prob': preds
            }
            export_outputs = {
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    tf.estimator.export.PredictOutput(predictions)
            }
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs
            )


        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels)
        ) + self.l2 * tf.nn.l2_loss(fm_w) + self.l2 * tf.nn.l2_loss(fm_v)

        # eval
        if mode == tf.estimator.ModeKeys.EVAL:
            if self.nclass == 2:
                eval_metric_ops = {
                    'auc': tf.metrics.auc(labels, preds)
                }
            else:
                p = tf.metrics.precision(labels, preds)
                r = tf.metrics.recall(labels, preds)
                f1 = 2 * p * r / (p + r)
                eval_metric_ops = {'f1': f1}
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=eval_metric_ops
            )

        # train
        if self.optimizer == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
        elif self.optimizer == 'Adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr, initial_accumulator_value=1e-8)
        elif self.optimizer == 'Momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.95)
        elif self.optimizer == 'ftrl':
            optimizer = tf.train.FtrlOptimizer(self.lr)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999, epsilon=1e-8)

        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op
        )

    def input_fn(self, data_file, batch_size, is_train=True):
        """
        svm format, but need the same feat column cnt per sample, otherwise maybe wrong
        :param data_file:
        :param batch_size:
        :param is_train:
        :return:
        """
        def parse_libsvm(line):
            columns = tf.string_split([line], ' ')
            labels = tf.string_to_number(columns.values[0], out_type=tf.float32)
            # labels = tf.reshape(labels, [-1])
            feat_kvs = tf.string_split(columns.values[1:], ':')  # ipinyou dataset, index is 2
            id_vals = tf.reshape(feat_kvs.values, feat_kvs.dense_shape)
            feat_ids, feat_vals = tf.split(id_vals, num_or_size_splits=2, axis=1)
            feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
            feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
            # sparse_feature = tf.SparseTensor(
            #     indices=feat_ids,     # feat_ids-1 is wrong, feat_ids is right
            #     values=tf.reshape(feat_vals, [-1]),
            #     dense_shape=[26 * 50]
            # )
            # dense_feature = tf.sparse.to_dense(sparse_feature, validate_indices=False)
            # return {"feat_ids": sparse_feature.indices, "feat_vals": sparse_feature.values}, labels
            return {"feat_ids": feat_ids, "feat_vals": feat_vals}, labels

        # Extract lines from input files using the Dataset API, can pass one filename or filename list
        dataset = tf.data.TextLineDataset(data_file).map(
            parse_libsvm, num_parallel_calls=10
        ).prefetch(500000)  # multi-thread pre-process then prefetch

        # Randomizes input using a window of 256 elements (read into memory)
        if is_train:
            dataset = dataset.shuffle(buffer_size=256)

        # epochs from blending together.
        dataset = dataset.repeat(self.epochs)
        dataset = dataset.batch(batch_size)  # Batch size to use

        iterator = dataset.make_one_shot_iterator()
        batch_features, batch_labels = iterator.get_next()
        # return tf.reshape(batch_ids,shape=[-1,field_size]), tf.reshape(batch_vals,shape=[-1,field_size]), batch_labels
        return batch_features, batch_labels

    def batch_norm(self):
        pass
