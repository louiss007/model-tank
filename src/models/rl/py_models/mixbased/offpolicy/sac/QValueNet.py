"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-12-23 下午3:56
# @FileName: QValueNet.py
# @Email   : quant_master2000@163.com
======================
"""
import torch
import torch.nn.functional as F


class QValueNet(torch.nn.Module):
    """ 只有一层隐藏层的Q网络 """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
