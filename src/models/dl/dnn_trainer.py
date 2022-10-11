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
from utils.arg_parse import arg_parse
from fnn import FnnModel
from cnn import CnnModel
from rnn import RnnModel
from lstm import LstmModel
from bilstm import BilstmModel
from gru import GruModel


class DnnTrainer(object):

    def __init__(self, task_type):
        self.task_type = task_type

    def train(self, model):
        train_files = os.path.join(model.input, 'train.tfrecord')
        test_files = os.path.join(model.input, 'dev.tfrecord')
        with tf.Session() as sess:
            variables_initner = tf.global_variables_initializer()
            tables_initner = tf.tables_initializer()
            sess.run(variables_initner)
            sess.run(tables_initner)
            for epoch in range(model.epoch):
                _x, _y = model.make_train_batch(train_files)  # must in epoch loop, not in step loop
                for step in range(model.n_step):
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
                            model.save_model(sess, model.model_path)

                print('===========validation start===========')
                test_x, test_y = model.make_test_batch(test_files)
                t_x, t_y = sess.run([test_x, test_y])
                if self.task_type == 'regression':
                    loss = model.eval(sess, t_x, t_y)
                    print('==========eval loss:{0}, epoch:{1}======'.format(loss, epoch))
                else:
                    loss, acc = model.eval(sess, t_x, t_y)
                    print('==========eval loss:{0}, eval acc:{1}, epoch:{2}======'.format(loss, acc, epoch))


def main(_):
    """模型训练程序运行入口"""
    task_description = 'Dnn Model Train Task!'
    parser = arg_parse(task_description)
    args = parser.parse_args()
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
    else:
        print('model name %s is not supported!' % model_name)
        return

    trainer = DnnTrainer(task_type)
    trainer.train(model)


if __name__ == '__main__':
    tf.app.run()
