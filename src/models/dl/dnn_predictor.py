"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-10-8 下午10:45
# @FileName: dnn_predictor.py
# @Email   : quant_master2000@163.com
======================
"""
import tensorflow as tf
from utils.arg_parse import arg_parse
import os

print('*******tf_version:%s********' % tf.__version__)


class DnnPredictor(object):

    def __init__(self, task_type):
        self.task_type = task_type

    def load_model(self, sess, model_file, model_format='pb'):
        """
        load pb model with tf.GraphDef()
        :param sess:
        :param model_file:
        :param model_format:
        :return:
        """
        with tf.gfile.FastGFile(model_file, 'rb') as fd:
            # 导入图
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fd.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
            # return graph_def  # make no sense

    def predict(self, x):
        """
        online predict api, offer algorithm service, when other service send a http request,
        then return a result
        :param x:
        :return:
        """
        pass

    def batch_predict(self, model_file, predict_file, is_eval=True):
        """
        offline evaluate model, get test metric or batch predict, write result to result file
        :param model_file: ckpt to pb format, that is pb format, use tf.GraphDef() load
        :param predict_file:
        :param is_eval:
        :return:
        """

        with tf.Session() as sess:
            # load tf model
            self.load_model(sess, model_file)
            # 根据名字获取对应的tensorflow
            input_x = sess.graph.get_tensor_by_name('input_x:0')
            input_y = sess.graph.get_tensor_by_name('input_y:0')

            # keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
            y_p = sess.graph.get_tensor_by_name('y_sm:0')

            # 测试准确率
            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(input_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            tot_sample = self.count_tfrecord(predict_file)
            mean_value = 0.0
            _x, _y = self.make_test_batch(predict_file, 'mnist', 1)
            for i in range(tot_sample):
                batch_x, batch_y = sess.run([_x, _y])
                train_accuracy = sess.run(accuracy, feed_dict={input_x: batch_x, input_y: batch_y})
                mean_value += train_accuracy
            print("test accuracy %g" % (mean_value / tot_sample))
            # mean_value = 0.0
            # for i in range(mnist.test.labels.shape[0]):
            #     batch = mnist.test.next_batch(50)
            #     train_accuracy = sess.run(accuracy, feed_dict={input_x: batch[0], input_y: batch[1]})
            #     mean_value += train_accuracy
            #
            # print("test accuracy %g" % (mean_value / mnist.test.labels.shape[0]))

    def batch_predict_pb(self, model_path, predict_file, is_eval=True):
        """
        offline evaluate model, get test metric or batch predict, write result to result file
        :param model_file: saved_model pb format, or ckpt to pb format
        :param predict_file:
        :param is_eval:
        :return:
        """
        pb_model_path = os.path.join(model_path, 'saved_model')
        with tf.Session() as sess:
            # load pb model with tf.saved_model
            tf.compat.v1.saved_model.loader.load(sess, ["serve"], pb_model_path)

            # 根据名字获取对应的tensor
            input_x = sess.graph.get_tensor_by_name('input_x:0')
            input_y = sess.graph.get_tensor_by_name('input_y:0')

            # keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
            y_p = sess.graph.get_tensor_by_name('y_sm:0')

            # 测试准确率
            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(input_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            tot_sample = self.count_tfrecord(predict_file)
            mean_value = 0.0
            _x, _y = self.make_test_batch(predict_file, 'mnist', 1)
            for i in range(tot_sample):
                batch_x, batch_y = sess.run([_x, _y])
                train_accuracy = sess.run(accuracy, feed_dict={input_x: batch_x, input_y: batch_y})
                mean_value += train_accuracy
            print("test accuracy %g" % (mean_value / tot_sample))

    def parse_tfrecord(self, tfrecord, record_type='mnist'):
        """
        通过解析TFrecord格式样本，返回x和y
        :param tfrecord: tfrecord格式样本
        :param record_type: cv/nlp/rs/mnist
        :return:
        """
        if record_type == 'mnist':
            # features = {
            #     'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.string])),
            #     'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.string])),
            #     'num1': tf.train.Feature(float_list=tf.train.FloatList(value=[tf.float32])),
            #     'num2': tf.train.Feature(int64_list=tf.train.Int64List(value=[tf.int64]))
            # }
            features = {
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string),
                'num1': tf.FixedLenFeature([], tf.float32),
                'num2': tf.FixedLenFeature([], tf.int64)
            }
            example = tf.parse_single_example(tfrecord, features=features)
            image = tf.decode_raw(example['image'], tf.float32)
            label = tf.decode_raw(example['label'], tf.float32)
            # num1 = example['num1']
            # num2 = example['num2']
            image = tf.reshape(image, shape=[28, 28, 1])
            image = tf.reshape(image, shape=[784])
            label = tf.reshape(label, shape=[10])
            return image, label
        elif record_type == 'cv':
            pass
        elif record_type == 'nlp':
            pass
        elif record_type == 'rs':
            features = {}
            # for col in self.feat_cols:
            #     features.setdefault(col, tf.FixedLenFeature([], tf.float32))
            # features.setdefault(self.target, tf.FixedLenFeature([], tf.float32))
            # example = tf.parse_single_example(tfrecord, features=features)
            # x = [example[col] for col in self.feat_cols]
            # y = [example[self.target]]
            # return x, y
        else:
            return None, None

    def make_test_batch(self, tfrecord_files, record_type='mnist', size=256):
        """
        TensorFlow训练数据持久化格式，通过加载tfrecord格式文件，方便模型测试
        :param tfrecord_files:
        :param size:
        :return:
        """
        dataset = tf.data.TFRecordDataset(tfrecord_files)
        if size is None or size == 0:
            dataset = dataset.map(lambda x: self.parse_tfrecord(x, record_type)).batch(256)
        else:
            dataset = dataset.map(lambda x: self.parse_tfrecord(x, record_type)).batch(size)
        iterator = dataset.make_one_shot_iterator()
        batch_x, batch_y = iterator.get_next()
        return batch_x, batch_y

    @staticmethod
    def count_tfrecord(tfrecord_file):
        count = 0
        for _ in tf.io.tf_record_iterator(tfrecord_file):
            count += 1
        print("数据{} 的样本条数为\t{}".format(tfrecord_file, count))
        return count


def main(_):
    """模型Infer程序运行入口"""
    task_description = 'Dnn Model Infer Task!'
    parser = arg_parse(task_description)
    args = parser.parse_args()
    print(args)
    task_type = args.task_type
    model_name = args.model_name
    model_path = args.model_path
    model_file = os.path.join(model_path, '{}model.pb'.format(model_name))
    test_file = args.input

    predictor = DnnPredictor(task_type)
    # predictor.batch_predict(model_file, test_file)
    predictor.batch_predict_pb(model_path, test_file)


if __name__ == '__main__':
    tf.compat.v1.app.run()
