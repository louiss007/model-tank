"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-10-18 下午11:27
# @FileName: tf_arg_parse.py
# @Email   : quant_master2000@163.com
======================
"""

import tensorflow as tf


def tf_arg_parse():
    flags = tf.flags
    flags.DEFINE_string("task_type", "classification", "task type: cls or reg")
    flags.DEFINE_string("model_name", "dssm", "nlp model")
    flags.DEFINE_boolean("is_cluster", False, "Whether run on cluster during training model or not")
    # flags.DEFINE_boolean("export_query_model", False, "Whether export model or not")
    # flags.DEFINE_boolean("export_doc_model", False, "Whether export model or not")
    flags.DEFINE_boolean("is_train", False, "Whether training model or not")

    flags.DEFINE_string("input", "data/nlp/sm/train.tfrecord", "input data")
    flags.DEFINE_string("job_name", "", "job name of ps")
    flags.DEFINE_string("ps_master", "", "comma-separated list of hostname:port pairs")
    flags.DEFINE_string("ps_worker", "", "comma-separated list of hostname:port pairs")
    flags.DEFINE_integer("task_index", 0, "Index of task within the job")

    flags.DEFINE_string("output", "output", "output path of model.")

    flags.DEFINE_integer("rnn_hidden_size", 64, "rnn_hidden_size")
    flags.DEFINE_integer("query_max_length", 50, "max token length of query text")
    flags.DEFINE_integer("doc_max_length", 200, "max token length of doc text")
    flags.DEFINE_integer("token_embedding_size", 128, "token embedding size")
    flags.DEFINE_string('layers', "128,64", "layers of nn")
    flags.DEFINE_integer("last_hidden_size", 64, "last hidden size")

    flags.DEFINE_float("lr", 0.001, "learning rate")
    flags.DEFINE_float("dropout", 1.0, "keep rate")
    flags.DEFINE_integer("token_vocab_size", 10000, "vocab size of token")
    flags.DEFINE_integer("neg_num", 50, "negative samples")
    flags.DEFINE_integer("epochs", 2, "training epochs")
    flags.DEFINE_integer("train_steps", 1000, "training steps")
    flags.DEFINE_integer("batch_size", 128, "training batch size")
    flags.DEFINE_integer("display_steps", 1000, "saving checkpoint of steps")
    FLAGS = flags.FLAGS
    return FLAGS


if __name__ == '__main__':
    args = tf_arg_parse()
    print(args.task_type)
    print(args.model_name)
