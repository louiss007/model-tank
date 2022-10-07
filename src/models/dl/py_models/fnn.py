"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-9-16 下午9:12
# @FileName: fnn.py
# @Email   : quant_master2000@163.com
======================
"""
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from model import Model


class FnnModel(Model):
    def __init__(self, args, task_type=None):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
