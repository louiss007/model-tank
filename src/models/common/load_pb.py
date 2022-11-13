"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-10-23 下午7:57
# @FileName: load_pb.py
# @Email   : quant_master2000@163.com
======================
"""

# coding: utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
from tensorflow.python.platform import gfile

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_model(sess, model_file):
    with gfile.FastGFile(model_file, 'rb') as fd:
        # 导入图
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fd.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        # return graph_def


mnist = input_data.read_data_sets('mnist_data', one_hot=True)

# 先检测看pb文件是否存在
savePath = '/home/louiss007/MyWorkShop/model/Practice/model-tank/output'
savePbPath = os.path.join(savePath, 'cnn')
savePbFile = os.path.join(savePbPath, 'cnnmodel.pb')
if os.path.exists(savePbFile) is False:
    print('Not found pb file!')
    exit()

with tf.Session() as sess:
    # 打开pb模型文件
    # with gfile.FastGFile(savePbFile, 'rb') as fd:
    #     # 导入图
    #     graph_def = tf.GraphDef()
    #     graph_def.ParseFromString(fd.read())
    #     sess.graph.as_default()
    #     tf.import_graph_def(graph_def, name='')

    # graph_def = load_model(sess, savePbFile)
    load_model(sess, savePbFile)

    # 根据名字获取对应的tensor
    input_x = sess.graph.get_tensor_by_name('input_x:0')
    input_y = sess.graph.get_tensor_by_name('input_y:0')

    # keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
    y_p = sess.graph.get_tensor_by_name('y_sm:0')

    # 测试准确率
    correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(input_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    mean_value = 0.0
    print("tot samles: ", mnist.test.labels.shape[0])
    for i in range(mnist.test.labels.shape[0]):
        batch = mnist.test.next_batch(50)
        train_accuracy = sess.run(accuracy, feed_dict={input_x: batch[0], input_y: batch[1]})
        mean_value += train_accuracy

    print("test accuracy %g" % (mean_value / mnist.test.labels.shape[0]))
        # #训练结束后，我们使用mnist.test在测试最后的准确率
        # print("test accuracy %g" % sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))
