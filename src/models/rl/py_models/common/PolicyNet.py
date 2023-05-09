"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-12-23 下午2:10
# @FileName: PolicyNet.py
# @Email   : quant_master2000@163.com
======================
"""
import torch
import torch.nn.functional as F


class PolicyNet(torch.nn.Module):
    """ 确定性策略网络，DDPG算法用 """
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound
