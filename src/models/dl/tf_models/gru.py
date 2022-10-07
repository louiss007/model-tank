"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-9-16 下午9:00
# @FileName: gru.py
# @Email   : quant_master2000@163.com
======================
"""
from model import Model


class GruModel(Model):
    """Gated Recurrent Unit"""
    def __init__(self, args, task_type=None):
        Model.__init__(self, args, task_type)
