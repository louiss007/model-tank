"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-10-22 上午12:15
# @FileName: clsm.py
# @Email   : quant_master2000@163.com
======================
"""
from nlp.sm.sm_model import DtModel


class ClsmModel(DtModel):
    """ """
    def __init__(self, args, task_type=None):
        DtModel.__init__(args, task_type)

