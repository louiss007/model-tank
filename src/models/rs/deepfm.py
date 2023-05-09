"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-9-19 下午11:20
# @FileName: deepfm.py
# @Email   : quant_master2000@163.com
======================
"""
import tensorflow as tf
from src.models.rs.rs_model import Model


class DeepFm(Model):
    """libsvm train file for deepfm model with tf.Estimator"""

    def __init__(self, args, task_type):
        Model.__init__(self, args, task_type)
        self.model_type = None
        self.warm_start_from = None
        self.optimizer = 'Adam'
        self.batch_norm_decay = 0.9
        self.is_batch_norm = True

    def model_fn(self, features, labels, mode, params=None):
        """Bulid Model function f(x) for Estimator."""
        # ------hyperparameters----
        field_size = params["field_size"]
        feature_size = params["feature_size"]
        embedding_size = params["embedding_size"]
        l2_reg = params["l2_reg"]
        # learning_rate = params["learning_rate"]
        # batch_norm_decay = params["batch_norm_decay"]
        # optimizer = params["optimizer"]
        # layers = map(int, params["deep_layers"].split(','))
        # dropout = map(float, params["dropout"].split(','))

        # ------bulid weights------
        FM_B = tf.get_variable(name='fm_bias', shape=[1], initializer=tf.constant_initializer(0.0))
        FM_W = tf.get_variable(name='fm_w', shape=[feature_size], initializer=tf.glorot_normal_initializer())
        FM_V = tf.get_variable(name='fm_v', shape=[feature_size, embedding_size],
                               initializer=tf.glorot_normal_initializer())

        # ------build feaure-------
        feat_ids = features['feat_ids']
        feat_ids = tf.reshape(feat_ids, shape=[-1, field_size])
        feat_vals = features['feat_vals']
        feat_vals = tf.reshape(feat_vals, shape=[-1, field_size])

        # ------build f(x)------
        with tf.variable_scope("First-order"):
            feat_wgts = tf.nn.embedding_lookup(FM_W, feat_ids)  # None * F * 1
            y_w = tf.reduce_sum(tf.multiply(feat_wgts, feat_vals), 1)

        with tf.variable_scope("Second-order"):
            embeddings = tf.nn.embedding_lookup(FM_V, feat_ids)  # None * F * K
            feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1])
            embeddings = tf.multiply(embeddings, feat_vals)  # vij*xi
            sum_square = tf.square(tf.reduce_sum(embeddings, 1))
            square_sum = tf.reduce_sum(tf.square(embeddings), 1)
            y_v = 0.5 * tf.reduce_sum(tf.subtract(sum_square, square_sum), 1)  # None * 1

        with tf.variable_scope("Deep-part"):
            if self.is_batch_norm:
                # normalizer_fn = tf.contrib.layers.batch_norm
                # normalizer_fn = tf.layers.batch_normalization
                if mode == tf.estimator.ModeKeys.TRAIN:
                    train_phase = True
                    # normalizer_params = {'decay': batch_norm_decay, 'center': True, 'scale': True, 'updates_collections': None, 'is_training': True, 'reuse': None}
                else:
                    train_phase = False
                    # normalizer_params = {'decay': batch_norm_decay, 'center': True, 'scale': True, 'updates_collections': None, 'is_training': False, 'reuse': True}
            else:
                normalizer_fn = None
                normalizer_params = None

            deep_inputs = tf.reshape(embeddings, shape=[-1, field_size * embedding_size])  # None * (F*K)
            for i in range(len(self.layers)):
                # if FLAGS.batch_norm:
                #    deep_inputs = batch_norm_layer(deep_inputs, train_phase=train_phase, scope_bn='bn_%d' %i)
                # normalizer_params.update({'scope': 'bn_%d' %i})
                deep_inputs = tf.contrib.layers.fully_connected(
                    inputs=deep_inputs,
                    num_outputs=self.nclass,
                    # normalizer_fn=normalizer_fn, normalizer_params=normalizer_params, \
                    weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                    scope='mlp%d' % i
                )
                if self.is_batch_norm:
                    deep_inputs = self.batch_norm(
                        deep_inputs,
                        train_phase=train_phase,
                        scope_bn='bn_%d' % i
                    )   # 放在RELU之后 https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md#bn----before-or-after-relu
                if mode == tf.estimator.ModeKeys.TRAIN:
                    deep_inputs = tf.nn.dropout(
                        deep_inputs, keep_prob=self.dropout)  # Apply Dropout after all BN layers and set dropout=0.8(drop_ratio=0.2)
                    # deep_inputs = tf.layers.dropout(inputs=deep_inputs, rate=dropout[i], training=mode == tf.estimator.ModeKeys.TRAIN)

            y_deep = tf.contrib.layers.fully_connected(
                inputs=deep_inputs,
                num_outputs=1,
                activation_fn=tf.identity,
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                scope='deep_out'
            )
            y_d = tf.reshape(y_deep, shape=[-1])
            # sig_wgts = tf.get_variable(name='sigmoid_weights', shape=[layers[-1]], initializer=tf.glorot_normal_initializer())
            # sig_bias = tf.get_variable(name='sigmoid_bias', shape=[1], initializer=tf.constant_initializer(0.0))
            # deep_out = tf.nn.xw_plus_b(deep_inputs,sig_wgts,sig_bias,name='deep_out')

        with tf.variable_scope("DeepFM-out"):
            # y_bias = FM_B * tf.ones_like(labels, dtype=tf.float32)  # None * 1  warning;这里不能用label，否则调用predict/export函数会出错，train/evaluate正常；初步判断estimator做了优化，用不到label时不传
            y_bias = FM_B * tf.ones_like(y_d, dtype=tf.float32)  # None * 1
            y = y_bias + y_w + y_v + y_d
            pred = tf.sigmoid(y)

        predictions = {"prob": pred}
        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
                predictions)}
        # Provide an estimator spec for `ModeKeys.PREDICT`
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs
            )

        # ------bulid loss------
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels)
        ) + l2_reg * tf.nn.l2_loss(FM_W) + l2_reg * tf.nn.l2_loss(FM_V)

        # Provide an estimator spec for `ModeKeys.EVAL`
        eval_metric_ops = {
            "auc": tf.metrics.auc(labels, pred)
        }
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                eval_metric_ops=eval_metric_ops
            )

        # ------select optimizer------
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

        # Provide an estimator spec for `ModeKeys.TRAIN` modes
        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
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
            #     indices=feat_ids,
            #     values=tf.reshape(feat_vals, [-1]),
            #     dense_shape=[26 * 50]
            # )
            # feat_ids-1 is wrong, feat_ids is right
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

    def batch_norm(self, x, train_phase, scope_bn):
        bn_train = tf.contrib.layers.batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True,
                                                updates_collections=None, is_training=True, reuse=None, scope=scope_bn)
        bn_infer = tf.contrib.layers.batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True,
                                                updates_collections=None, is_training=False, reuse=True, scope=scope_bn)
        z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)
        return z

