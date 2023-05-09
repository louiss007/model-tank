"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-12-23 下午2:11
# @FileName: ValueNet.py
# @Email   : quant_master2000@163.com
======================
"""
import torch
import torch.nn.functional as F


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
