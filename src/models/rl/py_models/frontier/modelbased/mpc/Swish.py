#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# @Time     : 2022/10/11 14:01
# @Author   : ZhengFu
# @Company  : Alibaba Financial
# @File     : Swish.py
# @Software : PyCharm
import torch


class Swish(torch.nn.Module):
    """ Swish激活函数 """
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
