"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-11-22 下午10:51
# @FileName: dnn_linear_combined.py
# @Email   : quant_master2000@163.com
======================
"""
import tensorflow as tf
import numpy as np


class Net(object):

    def __init__(self, wide_feat_map, layers, nclass, mode):
        self.wide_feat_map = wide_feat_map
        self.layers = layers
        self.nclass = nclass
        self.mode = mode

    def wide_net(self, features, feature_columns):
        """
        directly product between w and input
        :param features: input data
        :param feature_columns: valid feature columns
        :return: model logits
        """
        # Sparse Tensor
        input_layer = tf.feature_column.input_layer(features, feature_columns)
        feat_size = 0
        for key in self.wide_feat_map:
            feat_size += self.wide_feat_map.get(key)
        with tf.variable_scope('wide_model', values=(input_layer,)) as wide_scope:
            w_matrix = tf.Variable(tf.truncated_normal(
                (feat_size,),
                mean=0.0,
                stddev=1.0,
                dtype=tf.float32,
                seed=None,
                name='wide_init_weighted_matrix'
            ))
            bias = tf.Variable(tf.random_normal((1,)))
            print('input_layer:{}'.format(input_layer))
            print('w_matrix:{}'.format(w_matrix))
            wide_logits = tf.matmul(input_layer, w_matrix) + bias
        return tf.reshape(wide_logits, [-1, 1])

    def wide_net_acc(self, features, feature_columns):
        """
        accelerate product between w and input
        :param features: input data
        :param feature_columns: valid feature columns
        :return: model logits
        """
        # Sparse Tensor
        input_layer = tf.feature_column.input_layer(features, feature_columns)
        indices, vals = self.dense_to_sparse(input_layer)
        # print('indices:{}'.format(indices))
        # print('vals:{}'.format(vals))
        dense_shape = tf.shape(input_layer, out_type=tf.int64)
        feat_size = 0
        for key in self.wide_feat_map:
            feat_size += self.wide_feat_map.get(key)
        with tf.variable_scope('wide_model', values=(input_layer,)) as wide_scope:
            w_matrix = tf.Variable(tf.truncated_normal(
                (feat_size,),
                mean=0.0,
                stddev=1.0,
                dtype=tf.float32,
                seed=None,
                name='wide_init_weighted_matrix'
            ))
            bias = tf.Variable(tf.random_normal((1,)))
            sp_ids = tf.SparseTensor(indices=indices, values=vals, dense_shape=dense_shape)
            print('sp_ids:{}'.format(sp_ids))
            wide_logits = tf.nn.embedding_lookup_sparse(
                params=w_matrix,
                sp_ids=sp_ids,
                sp_weights=None,
                combiner='sum'
            ) + bias
        return tf.reshape(wide_logits, [-1, 1])

    def deep_net(self, features, feature_columns):
        dense_tensor = tf.feature_column.input_layer(features, feature_columns)
        with tf.variable_scope('deep_model', values=(dense_tensor,)) as deep_scope:
            for unit in self.layers:
                dense_tensor = tf.compat.v1.layers.dense(dense_tensor, unit, tf.nn.relu)
            deep_logits = tf.compat.v1.layers.dense(dense_tensor, 1)
            # for i in range(1, len(self.tmp_layers)):
            #     input_layer = tf.add(
            #         tf.matmul(
            #             input_layer, self.weights['h' + str(i)]
            #         ),
            #         self.biases['b' + str(i)]
            #     )
            # deep_logits = tf.matmul(input_layer, self.weights['out']) + self.biases['out']
            return deep_logits

    @staticmethod
    def dense_to_sparse(dense_tensor):
        zero = tf.constant(0, dtype=tf.float32)
        where = tf.not_equal(dense_tensor, zero)
        indices = tf.where(where)
        vals = tf.gather_nd(dense_tensor, indices)
        return indices, vals
