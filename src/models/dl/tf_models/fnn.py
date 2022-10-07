"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-9-16 下午8:57
# @FileName: fnn.py
# @Email   : quant_master2000@163.com
======================
"""
from model import Model


class FnnModel(Model):
    """Feedforward Neural Network"""
    def __init__(self, args, task_type=None):
        Model.__init__(self, args, task_type)
