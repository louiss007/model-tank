#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# @Time     : 2022/10/11 14:02
# @Author   : ZhengFu
# @Company  : Alibaba Financial
# @File     : EnsembleModel.py
# @Software : PyCharm
import torch
import torch.nn.functional as F
import numpy as np
from FCLayer import FCLayer
from Swish import Swish


class EnsembleModel(torch.nn.Module):
    """ 环境模型集成 """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 ensemble_size=5,
                 learning_rate=1e-3):
        super(EnsembleModel, self).__init__()
        self.device = device
        # 输出包括均值和方差,因此是状态与奖励维度之和的两倍
        self._output_dim = (state_dim + 1) * 2
        self._max_logvar = torch.nn.Parameter((torch.ones(
            (1, self._output_dim // 2)).float() / 2).to(self.device),
                                        requires_grad=False)
        self._min_logvar = torch.nn.Parameter((-torch.ones(
            (1, self._output_dim // 2)).float() * 10).to(self.device),
                                        requires_grad=False)

        self.layer1 = FCLayer(state_dim + action_dim, 200, ensemble_size,
                              Swish(), self.device)
        self.layer2 = FCLayer(200, 200, ensemble_size, Swish(), self.device)
        self.layer3 = FCLayer(200, 200, ensemble_size, Swish(), self.device)
        self.layer4 = FCLayer(200, 200, ensemble_size, Swish(), self.device)
        self.layer5 = FCLayer(200, self._output_dim, ensemble_size, torch.nn.Identity(), self.device)
        self.apply(self.init_weights)  # 初始化环境模型中的参数
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def init_weights(self, m):
        """ 初始化模型权重 """
        def truncated_normal_init(t, mean=0.0, std=0.01):
            torch.nn.init.normal_(t, mean=mean, std=std)
            while True:
                cond = (t < mean - 2 * std) | (t > mean + 2 * std)
                if not torch.sum(cond):
                    break
                t = torch.where(
                    cond,
                    torch.nn.init.normal_(torch.ones(t.shape, device=self.device),
                                          mean=mean,
                                          std=std), t)
            return t

        if type(m) == torch.nn.Linear or isinstance(m, FCLayer):
            truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(m._input_dim)))
            m.bias.data.fill_(0.0)

    def forward(self, x, return_log_var=False):
        ret = self.layer5(self.layer4(self.layer3(self.layer2(
            self.layer1(x)))))
        mean = ret[:, :, :self._output_dim // 2]
        # 在PETS算法中,将方差控制在最小值和最大值之间
        logvar = self._max_logvar - F.softplus(
            self._max_logvar - ret[:, :, self._output_dim // 2:])
        logvar = self._min_logvar + F.softplus(logvar - self._min_logvar)
        return mean, logvar if return_log_var else torch.exp(logvar)

    def loss(self, mean, logvar, labels, use_var_loss=True):
        inverse_var = torch.exp(-logvar)
        if use_var_loss:
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) *
                                             inverse_var,
                                             dim=-1),
                                  dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss

    def train(self, loss):
        self.optimizer.zero_grad()
        loss += 0.01 * torch.sum(self._max_logvar) - 0.01 * torch.sum(
            self._min_logvar)
        loss.backward()
        self.optimizer.step()