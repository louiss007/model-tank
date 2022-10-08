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
from utils.arg_parse import arg_parse
from fnn import FnnModel
from cnn import CnnModel
from rnn import RnnModel
from lstm import LstmModel
from bilstm import BilstmModel
from gru import GruModel


class DnnTrainer(object):

    def __init__(self):
        pass

    def train(self, train_files, test_files, model_para, model, task_type, is_train=True):
        with tf.Session() as sess:
            variables_initner = tf.global_variables_initializer()
            tables_initner = tf.tables_initializer()
            sess.run(variables_initner)
            sess.run(tables_initner)
            if is_train:
                for epoch in range(model.epoch):
                    _x, _y = model.make_one_batch(train_files)  # must in epoch loop, not in step loop
                    for step in range(model.n_step):
                        batch_x, batch_y = sess.run([_x, _y])
                        if task_type is None or task_type == 'classification':
                            loss, acc, global_step = model.fit(sess, batch_x, batch_y)
                            if global_step % model_para.get('display_step') == 0:
                                print('==========train loss:{0}, train acc:{1}, epoch:{2}, global step:{3}======'
                                      .format(loss, acc, epoch, global_step))
                                model.save_model(sess, model.model_path)
                        if task_type == 'regression':
                            loss, global_step = model.fit(sess, batch_x, batch_y)
                            if global_step % model_para.get('display_step') == 0:
                                print('==========train loss:{0}, epoch:{1}, global step:{2}======'
                                      .format(loss, epoch, global_step))
                                model.save_model(sess, model.model_path)

                    print('===========validation start===========')
                    test_x, test_y = model.make_batch(test_files)
                    t_x, t_y = sess.run([test_x, test_y])
                    if task_type is None or task_type == 'classification':
                        loss, acc, _ = model.fit(sess, t_x, t_y)
                        print('==========test loss:{0}, test acc:{1}, epoch:{2}======'
                              .format(loss, acc, epoch))
                        # model.save_model(sess, model.model_path)
                    if task_type == 'regression':
                        loss, _ = model.fit(sess, t_x, t_y)
                        print('==========test loss:{0}, epoch:{1}======'
                              .format(loss, epoch))

            else:
                test_n_step = model.test_sample_size // model.batch_size + 1
                for _ in range(test_n_step):
                    batch_x, batch_y = model.make_one_batch(test_files)
                    _x, _y = sess.run([batch_x, batch_y])
                    result = model.predict(sess, _x, _y)


def main(task_type=None):
    """
    模型训练程序运行入口
    :param task_type: 分类、回归或者排序
    :return: null
    """
    task_description = 'dnn {} train task'.format(task_type)
    parser = arg_parse(task_description)
    args = parser.parse_args()
    model_name = args.model_name
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    if model_name == 'fnn':
        model = FnnModel()
    elif model_name == 'cnn':
        model = CnnModel()
    else:
        print('model name %s is not supported!' % model_name)
        return

    trainer = DnnTrainer()
    trainer.train()


if __name__ == '__main__':
    tf.app.run()
