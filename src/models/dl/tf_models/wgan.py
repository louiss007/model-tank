"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-9-16 下午9:02
# @FileName: wgan.py
# @Email   : quant_master2000@163.com
======================
"""
from model import Model


class WganModel(Model):
    """D C Generial"""
    def __init__(self, args, task_type=None):
        Model.__init__(self, args, task_type)
