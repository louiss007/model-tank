"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-9-16 下午9:02
# @FileName: wgan.py
# @Email   : quant_master2000@163.com
======================
"""
from src.models.dl.tf_models.model import Model


class WganModel(Model):
    """Wasserstein Generative Adversarial Network"""
    def __init__(self, args, task_type=None):
        Model.__init__(self, args, task_type)
