"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-12-22 下午11:10
# @FileName: ReplayBuffer.py
# @Email   : quant_master2000@163.com
======================
"""
import collections
import random
import numpy as np


class ReplayBuffer:
    """ 经验回放池 """
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        # transitions = np.random.choice(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def all_sample(self):
        all_transitions = list(self.buffer)
        state, action, reward, next_state, done = zip(*all_transitions)
        return np.array(state), action, reward, np.array(next_state), done
