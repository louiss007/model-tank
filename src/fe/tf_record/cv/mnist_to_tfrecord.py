"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-10-7 下午10:14
# @FileName: mnist_to_tfrecord.py
# @Email   : quant_master2000@163.com
======================
"""
import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def create_example(image, label):
    example = tf.train.Example(features=tf.train.Features(feature={
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
        'num1': tf.train.Feature(float_list=tf.train.FloatList(value=[12.555])),
        'num2': tf.train.Feature(int64_list=tf.train.Int64List(value=[88]))
    }))
    return example


def create_tfrecord(out_path, mnist_data_path='../../data/mnist'):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    train_tfrecord_path = os.path.join(out_path, 'train.tfrecord')
    test_tfrecord_path = os.path.join(out_path, 'test.tfrecord')
    mnist = input_data.read_data_sets(train_dir=mnist_data_path, one_hot=True, validation_size=0)
    with tf.io.TFRecordWriter(train_tfrecord_path) as train_writer:
        print('执行训练数据生成')
        for idx in range(mnist.train.num_examples):
            image = mnist.train.images[idx]
            label = mnist.train.labels[idx]
            image = np.reshape(image, -1).astype(np.float32)
            label = np.reshape(label, -1).astype(np.float32)
            example = create_example(image.tobytes(), label.tobytes())
            train_writer.write(example.SerializeToString())
    with tf.io.TFRecordWriter(test_tfrecord_path) as test_writer:
        print('执行测试数据生成')
        for idx in range(mnist.test.num_examples):
            image = mnist.test.images[idx]
            label = mnist.test.labels[idx]
            image = np.reshape(image, -1).astype(np.float32)
            label = np.reshape(label, -1).astype(np.float32)
            example = create_example(image.tobytes(), label.tobytes())
            test_writer.write(example.SerializeToString())


def load_tfrecord(tfrecord_file_path, batch_size, height, width, channels, n_class):
    producer = tf.train.string_input_producer([tfrecord_file_path])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(queue=producer)
    features = tf.parse_single_example(serialized_example, features={
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.string),
        'num1': tf.FixedLenFeature([], tf.float32),
        'num2': tf.FixedLenFeature([], tf.int64)
    })
    image = tf.decode_raw(features['image'], tf.float32)
    label = tf.decode_raw(features['label'], tf.float32)
    num1 = features['num1']
    num2 = features['num2']
    image = tf.reshape(image, shape=[height, width, channels])
    label = tf.reshape(label, shape=[n_class])
    image, label, num1, num2 = tf.train.shuffle_batch(
        [image, label, num1, num2],
        batch_size=batch_size,
        capacity=batch_size*5,
        num_threads=1,
        min_after_dequeue=batch_size*2
    )
    return image, label, num1, num2


def display_tfrecord(tfrecord_file):
    item = next(tf.io.tf_record_iterator(tfrecord_file))
    print(tf.train.Example.FromString(item))


def count_tfrecord(tfrecord_file):
    count = 0
    for _ in tf.io.tf_record_iterator(tfrecord_file):
        count += 1
    print("数据{} 的样本条数为\t{}".format(tfrecord_file, count))
    return count


if __name__ == '__main__':
    out_path = '/home/louiss007/MyWorkShop/model/Practice/model-tank/data/cv/mnist'

    # create_tfrecord(out_path)
    # display_tfrecord('{op}/train.tfrecord'.format(op=out_path))
    # count_tfrecord('{op}/test.tfrecord'.format(op=out_path))
    mnist = input_data.read_data_sets(train_dir=out_path, one_hot=True, validation_size=0)
    print(mnist.train.images[0].shape[0])
