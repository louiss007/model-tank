"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-9-16 下午8:58
# @FileName: cnn.py
# @Email   : quant_master2000@163.com
======================
"""
from model import Model


class CnnModel(Model):
    """Convolution Neural Network"""
    def __init__(self, args, task_type=None):
        Model.__init__(self, args, task_type)
        self.height = 28
        self.width = 28
        self.channels = 1
        self.model_path = '{mp}/nn/cnn/cnn'.format(mp=args.get('model_path'))
        self.init_net()
        if self.task_type == 'regression':
            self.loss, self.train_op = self.build_model()
        else:
            self.loss, self.train_op, self.accuracy = self.build_model()
