#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# @Time     : 2022/10/11 13:56
# @Author   : ZhengFu
# @Company  : Alibaba Financial
# @File     : GAIL.py
# @Software : PyCharm
import torch
import torch.nn.functional as F
from common.Discriminator import Discriminator


class GAIL:
    def __init__(self, agent, state_dim, action_dim, hidden_dim, lr_d, device):
        self.device = device
        self.discriminator = Discriminator(state_dim, hidden_dim,
                                           action_dim).to(self.device)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_d)
        self.agent = agent

    def learn(self, expert_s, expert_a, agent_s, agent_a, next_s, dones):
        expert_states = torch.tensor(expert_s, dtype=torch.float).to(self.device)
        expert_actions = torch.tensor(expert_a).to(self.device)
        agent_states = torch.tensor(agent_s, dtype=torch.float).to(self.device)
        agent_actions = torch.tensor(agent_a).to(self.device)
        expert_actions = F.one_hot(expert_actions, num_classes=2).float()
        agent_actions = F.one_hot(agent_actions, num_classes=2).float()

        expert_prob = self.discriminator(expert_states, expert_actions)
        agent_prob = self.discriminator(agent_states, agent_actions)
        discriminator_loss = torch.nn.BCELoss()(
            agent_prob, torch.ones_like(agent_prob)) + torch.nn.BCELoss()(
                expert_prob, torch.zeros_like(expert_prob))
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        rewards = -torch.log(agent_prob).detach().cpu().numpy()
        transition_dict = {
            'states': agent_s,
            'actions': agent_a,
            'rewards': rewards,
            'next_states': next_s,
            'dones': dones
        }
        self.agent.update(transition_dict)
