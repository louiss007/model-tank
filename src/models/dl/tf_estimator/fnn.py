"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-9-19 下午11:25
# @FileName: fnn.py
# @Email   : quant_master2000@163.com
======================
"""
from src.models.dl.tf_estimator.model import Model
import tensorflow as tf


class FnnModel(Model):

    def __init__(self, args, task_type=None):
        Model.__init__(self, args, task_type)
