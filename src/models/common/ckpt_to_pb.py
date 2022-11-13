"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-10-23 上午11:29
# @FileName: ckpt_to_pb.py
# @Email   : quant_master2000@163.com
======================
"""
import tensorflow as tf
from tensorflow.python.framework import graph_util
import os
"""
Reference: 
[1]. https://blog.csdn.net/guyuealian/article/details/82218092
[2]. https://blog.csdn.net/rookie_wei/article/details/90546290
"""


def freeze_graph(ckpt_model_path, pb_model_path):
    """
    convert ckpt to pb
    :param ckpt_model_path:
    :param pb_model_path:
    :return:
    """
    # checkpoint = tf.train.get_checkpoint_state(model_folder)  # 检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path   # huo得ckpt文件路径

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "input_x,input_y,y_sm"
    meta_file = os.path.join(ckpt_model_path, 'cnnmodel.ckpt.meta')
    saver = tf.train.import_meta_graph(meta_file, clear_devices=True)
    # the following two lines can be ignored
    graph = tf.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        saver.restore(sess, ckpt_model_path+'/cnnmodel.ckpt')  # 恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(",")
        )  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(pb_model_path, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点

        # for op in graph.get_operations():
        #     print(op.name, op.values())


def show_tf_node_name_pb(pb_model_path):
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ['serve'], pb_model_path)
        print('load model finished!')
        input_graph_def = tf.get_default_graph().as_graph_def()
        node_names = [n.name for n in input_graph_def.node]
        for node in node_names:
            print(node)


def show_tf_node_name_ckpt(ckpt_model_path):
    with tf.Session() as sess:
        meta_file = os.path.join(ckpt_model_path, 'cnnmodel.ckpt.meta')
        saver = tf.train.import_meta_graph(meta_file)

        # Method One is OK
        # saver.restore(sess, tf.train.latest_checkpoint(ckpt_model_path))
        #
        # for var_name, _ in tf.contrib.framework.list_variables(ckpt_model_path):
        #     print(var_name)
        #     var = tf.contrib.framework.load_variable(ckpt_model_path, var_name)
        #     print(var.shape)

        # Method Two is OK, list all tensors in the graph
        for tensor in tf.get_default_graph().get_operations():
            print(tensor.name)


if __name__ == '__main__':
    ckpt_model_path = '/home/louiss007/MyWorkShop/model/Practice/model-tank/output/cnn'
    pb_model_path = '/home/louiss007/MyWorkShop/model/Practice/model-tank/output/cnn/cnnmodel.pb'
    # show_tf_node_name_ckpt(ckpt_model_path)
    freeze_graph(ckpt_model_path, pb_model_path)
    # show_tf_node_name_pb('/home/louiss007/MyWorkShop/model/Practice/model-tank/output/cnn')
