#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# @Time     : 2022/10/11 13:35
# @Author   : ZhengFu
# @Company  : Alibaba Financial
# @File     : BehaviorClone.py
# @Software : PyCharm
import torch
from policybased.pg.PolicyNet import PolicyNet


class BehaviorClone:
    def __init__(self, state_dim, hidden_dim, action_dim, lr, device):
        self.device = device
        self.policy = PolicyNet(state_dim, hidden_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def learn(self, states, actions):
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions).view(-1, 1).to(self.device)
        log_probs = torch.log(self.policy(states).gather(1, actions))
        bc_loss = torch.mean(-log_probs)  # 最大似然估计

        self.optimizer.zero_grad()
        bc_loss.backward()
        self.optimizer.step()

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
