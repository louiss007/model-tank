"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-9-19 下午11:22
# @FileName: dssm.py
# @Email   : quant_master2000@163.com
======================
"""
from src.models.nlp.dt_model import DtModel
import tensorflow as tf


class DssmModel(DtModel):
    """ Deep Structured Semantic Model, used to Semantic Match """
    def __init__(self, args, task_type=None):
        DtModel.__init__(self, args, task_type)
        self.rnn_hidden_size = 256
        self.test_file_path = args.test_file_path

    def text_embedding(self, text, text_length, scope_name, reuse=None):
        with tf.variable_scope(scope_name, reuse=reuse):
            embedding = tf.reshape(text, [-1, text_length, self.token_embedding_size])
            token_seq_encoder = self.rnn_encoder(embedding, scope_name, tf.AUTO_REUSE)
            text_encoder = self.attn_layer(
                token_seq_encoder,
                scope_name=scope_name,
                out_name=scope_name,
                reuse=tf.AUTO_REUSE
            )
            return text_encoder

    def rnn_encoder(self, input, scope_name, reuse):
        with tf.variable_scope(scope_name, reuse=reuse):
            gru_cell_fw = tf.contrib.rnn.GRUCell(self.rnn_hidden_size)
            gru_cell_bw = tf.contrib.rnn.GRUCell(self.rnn_hidden_size)
            ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=gru_cell_fw,
                cell_bw=gru_cell_bw,
                inputs=input,
                sequence_length=self.compute_seq_length(input),
                dtype=tf.float32
            )
            outputs = tf.concat((fw_outputs, bw_outputs), axis=2)
            outputs = tf.nn.tanh(outputs)
            return outputs

    def attn_layer(self, input, scope_name, out_name, reuse):
        with tf.variable_scope(scope_name, reuse=reuse):
            u_context = tf.Variable(
                tf.truncated_normal([self.rnn_hidden_size * 2]),
                name=scope_name + '_u_context'
            )
            h = tf.contrib.layers.fully_connected(
                input, self.rnn_hidden_size * 2, activation_fn=tf.nn.tanh
            )
            alpha = tf.nn.softmax(
                tf.reduce_sum(tf.multiply(h, u_context), axis=2, keepdims=True),
                axis=1
            )
        attn_output = tf.reduce_sum(tf.multiply(input, alpha), axis=1, name=out_name)
        # attn_output = tf.nn.tanh(attn_output)
        return attn_output

    @staticmethod
    def compute_seq_length(sequences):
        used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
        seq_len = tf.reduce_sum(used, reduction_indices=1)
        return tf.cast(seq_len, tf.int32)
