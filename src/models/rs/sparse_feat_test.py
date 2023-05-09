"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-12-4 下午6:09
# @FileName: sparse_feat_test.py
# @Email   : quant_master2000@163.com
======================
"""
import tensorflow as tf


def input_fn(data_file, batch_size, is_train=True):
    """
    svm format
    :param data_file:
    :param batch_size:
    :param is_train:
    :return:
    """

    def parse_libsvm(line):
        columns = tf.string_split([line], ' ')
        labels = tf.string_to_number(columns.values[0], out_type=tf.int32)
        labels = tf.reshape(labels, [-1])
        feat_kvs = tf.string_split(columns.values[2:], ':')  # ipinyou dataset, index is 2
        id_vals = tf.reshape(feat_kvs.values, feat_kvs.dense_shape)
        feat_ids, feat_vals = tf.split(id_vals, num_or_size_splits=2, axis=1)
        feat_ids = tf.string_to_number(feat_ids, out_type=tf.int64)
        feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
        sparse_feature = tf.SparseTensor(
            indices=feat_ids,
            values=tf.reshape(feat_vals, [-1]),
            dense_shape=[26*50]
        )
        # feat_ids-1 is wrong, feat_ids is right
        dense_feature = tf.sparse.to_dense(sparse_feature, validate_indices=False)
        # # return {"feat_ids": feat_ids, "feat_vals": feat_vals}, labels
        # return {"dense_input": dense_feature}, labels
        return sparse_feature, labels
        # return dense_feature, labels

    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    dataset = tf.data.TextLineDataset(data_file).map(
        parse_libsvm, num_parallel_calls=10
    ).prefetch(500000)  # multi-thread pre-process then prefetch

    # Randomizes input using a window of 256 elements (read into memory)
    if is_train:
        dataset = dataset.shuffle(buffer_size=256)

    # epochs from blending together.
    dataset = dataset.repeat(1)
    dataset = dataset.batch(batch_size)  # Batch size to use

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    # return tf.reshape(batch_ids,shape=[-1,field_size]), tf.reshape(batch_vals,shape=[-1,field_size]), batch_labels
    return batch_features, batch_labels


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


def embedding_lookup_demo():
    import numpy as np
    c = np.random.random([5, 1])
    print(c)
    b = tf.nn.embedding_lookup(c, [1, 3])
    tf.global_variables_initializer()
    with tf.Session() as sess:
        b = sess.run(b)
        print(b)


if __name__ == '__main__':
    # libsvm_train_file = '/home/louiss007/MyWorkShop/dataset/ipinyou/make-ipinyou-data/2997/train.yzx.txt'
    # features, labels = input_fn(libsvm_train_file, 32, True)
    # tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     feats =sess.run(features)
    #     # print(feats)
    #     for k in feats.indices:
    #         print(k)

    # line = "0 1:0 2:0.008292 3:6.32 4:0 5:0.887031 6:0 7:0 8:0.1 9:0.13 10:0 11:0 12:0 13:0.04 14:1 563:1 1563:1 12620:1 26190:1 26341:1 27377:1 35359:1 35614:1 35617:1 47339:1 51591:1 63110:1 65962:1 67431:1 71860:1 84092:1 84772:1 86889:1 88281:1 88503:1 100288:1 100300:1 100410:1 109933:1 110053:1"
    # kv, vv = parse_libsvm(line)
    # with tf.Session() as sess:
    #     print(sess.run(kv))

    # indices = tf.placeholder(tf.int64)
    # shape = tf.placeholder(tf.int64)
    # values = tf.placeholder(tf.float64)
    # sparse_tensor = tf.SparseTensor(indices, values, shape)
    # tf.global_variables_initializer()
    # sess = tf.Session()
    # st = sess.run(sparse_tensor, feed_dict={shape: [3, 3], indices: [[0, 1], [2, 2]], values: [1.5, 2.9]})
    # print(st.indices)

    embedding_lookup_demo()
