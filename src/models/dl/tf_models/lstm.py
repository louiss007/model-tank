"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-9-16 下午8:59
# @FileName: lstm.py
# @Email   : quant_master2000@163.com
======================
"""
from model import Model


class LstmModel(Model):
    """Long Short Term Memory"""
    def __init__(self, args, task_type=None):
        Model.__init__(self, args, task_type)
