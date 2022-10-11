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


class DnnPredictor(object):

    def __init__(self, task_type):
        self.task_type = task_type

    def load_model(self):
        pass

    def predict(self):
        pass

    def batch_predict(self):
        pass
