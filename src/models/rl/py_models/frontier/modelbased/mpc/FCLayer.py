#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# @Time     : 2022/10/11 14:03
# @Author   : ZhengFu
# @Company  : Alibaba Financial
# @File     : FCLayer.py
# @Software : PyCharm
import torch


class FCLayer(torch.nn.Module):
    """ 集成之后的全连接层 """
    def __init__(self, input_dim, output_dim, ensemble_size, activation, device):
        super(FCLayer, self).__init__()
        self.device = device
        self._input_dim, self._output_dim = input_dim, output_dim
        self.weight = torch.nn.Parameter(
            torch.Tensor(ensemble_size, input_dim, output_dim).to(self.device))
        self._activation = activation
        self.bias = torch.nn.Parameter(
            torch.Tensor(ensemble_size, output_dim).to(self.device))

    def forward(self, x):
        return self._activation(
            torch.add(torch.bmm(x, self.weight), self.bias[:, None, :]))
