"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-10-8 下午10:45
# @FileName: dnn_trainer.py
# @Email   : quant_master2000@163.com
======================
"""
import tensorflow as tf
import os
import sys
from utils.arg_parse import arg_parse
from src.models.dl.tf_models.fnn import FnnModel
from src.models.dl.tf_models.cnn import CnnModel
from src.models.dl.tf_models.rnn import RnnModel
from src.models.dl.tf_models.lstm import LstmModel
from src.models.dl.tf_models.bilstm import BilstmModel
from src.models.dl.tf_models.gru import GruModel
from src.models.dl.tf_models.gan import GanModel
from src.models.dl.tf_models.dcgan import DcganModel
from src.models.dl.tf_models.wgan import WganModel
from src.models.cv.alex_net import AlexNet


print('*******tf_version:%s********' % tf.__version__)
# config = tf.compat.v1.ConfigProto()
# # config = tf.ConfigProto()
# config.gpu_options.allow_growth = True      # TensorFlow按需分配显存
# config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 指定显存分配比例
# # config.gpu_options.allow_growth = True
# # config.gpu_options.allow_growth = True

# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class DnnTrainer(object):

    def __init__(self, task_type):
        self.task_type = task_type

    def train(self, model):
        """
        single machine training format
        :param model:
        :return:
        """
        if not os.path.exists(model.input):
            print('input train file is not exist, please check it! try it again.')
            sys.exit(1)
        train_file = model.input
        test_file = os.path.join(os.path.split(model.input)[0], 'test.tfrecord')
        with tf.Session() as sess:
            variables_initner = tf.global_variables_initializer()
            tables_initner = tf.tables_initializer()
            sess.run(variables_initner)
            sess.run(tables_initner)
            test_x, test_y = model.make_test_batch(test_file)
            t_x, t_y = sess.run([test_x, test_y])
            for epoch in range(model.epochs):
                _x, _y = model.make_train_batch(train_file)  # must in epoch loop, not in step loop
                for step in range(model.train_steps):
                    batch_x, batch_y = sess.run([_x, _y])
                    if self.task_type == 'regression':
                        loss, global_step = model.fit(sess, batch_x, batch_y)
                        if global_step % model.display_step == 0:
                            print('==========train loss:{0}, epoch:{1}, global step:{2}======'
                                  .format(loss, epoch, global_step))
                            model.save_model(sess, model.model_path)
                    else:
                        loss, acc, global_step = model.fit(sess, batch_x, batch_y)
                        if global_step % model.display_step == 0:
                            print('==========train loss:{0}, train acc:{1}, epoch:{2}, global step:{3}======'
                                  .format(loss, acc, epoch, global_step))
                            model.save_model(sess)

                print('===========validation start===========')
                if self.task_type == 'regression':
                    loss = model.eval(sess, t_x, t_y)
                    print('==========eval loss:{0}, epoch:{1}======'.format(loss, epoch))
                else:
                    loss, acc = model.eval(sess, t_x, t_y)
                    print('==========eval loss:{0}, eval acc:{1}, epoch:{2}======'.format(loss, acc, epoch))

    def train_unsupervised(self, model):
        """

        :param model:
        :return:
        """
        if not os.path.exists(model.input):
            print('input train file is not exist, please check it! try it again.')
            sys.exit(1)

        image_path = os.path.split(os.path.abspath(__file__))[0]
        train_file = model.input
        with tf.Session() as sess:
            variables_initner = tf.global_variables_initializer()
            tables_initner = tf.tables_initializer()
            sess.run(variables_initner)
            sess.run(tables_initner)
            for epoch in range(model.epochs):
                _x, _y = model.make_train_batch(train_file)  # must in epoch loop, not in step loop
                for step in range(model.train_steps):
                    batch_x, _ = sess.run([_x, _y])     # if del _y in sess.run(), will cause error!!!
                    feed_dict = model.make_train_batch_for_g(batch_x)
                    g_loss, d_loss, global_step = model.fit_unsupervised(sess, feed_dict)
                    if global_step % model.display_step == 0:
                        print('==========g_loss:{0}, d_loss:{1}, epoch:{2}, global step:{3}======'
                              .format(g_loss, d_loss, epoch, global_step))
                        model.save_model(sess)
            model.generate_image(sess, image_path)
            model.show(image_path)


def main(_):
    """模型训练程序运行入口"""
    task_description = 'Dnn Model Train Task!'
    parser = arg_parse(task_description)
    args = parser.parse_args()
    print(args)
    task_type = args.task_type
    model_name = args.model_name

    if model_name == 'fnn':
        model = FnnModel(args, task_type)
    elif model_name == 'cnn':
        model = CnnModel(args, task_type)
    elif model_name == 'rnn':
        model = RnnModel(args, task_type)
    elif model_name == 'lstm':
        model = LstmModel(args, task_type)
    elif model_name == 'bilstm':
        model = BilstmModel(args, task_type)
    elif model_name == 'gru':
        model = GruModel(args, task_type)
    elif model_name == 'gan':
        model = GanModel(args, task_type)
    elif model_name == 'dcgan':
        model = DcganModel(args, task_type)
    elif model_name == 'wgan':
        model = WganModel(args, task_type)
    elif model_name == 'alexnet':
        model = AlexNet(args, task_type)
    else:
        print('model name %s is not supported!' % model_name)
        return

    trainer = DnnTrainer(task_type)
    print('Model %s is called' % model.__class__.__name__)
    if model_name.find('gan') != -1:
        trainer.train_unsupervised(model)
        sys.exit(0)
    trainer.train(model)


if __name__ == '__main__':
    tf.compat.v1.app.run()
    # image_path = os.path.split(os.path.abspath(__file__))
    # print(image_path[0])
