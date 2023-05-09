#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# @Time     : 2022/10/11 14:32
# @Author   : ZhengFu
# @Company  : Alibaba Financial
# @File     : Trajectory.py
# @Software : PyCharm


class Trajectory:
    ''' 用来记录一条完整轨迹 '''
    def __init__(self, init_state):
        self.states = [init_state]
        self.actions = []
        self.rewards = []
        self.dones = []
        self.length = 0

    def store_step(self, action, state, reward, done):
        self.actions.append(action)
        self.states.append(state)
        self.rewards.append(reward)
        self.dones.append(done)
        self.length += 1
