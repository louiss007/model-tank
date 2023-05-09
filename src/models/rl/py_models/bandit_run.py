"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-12-22 下午10:47
# @FileName: bandit_run.py
# @Email   : quant_master2000@163.com
======================
"""

import matplotlib.pyplot as plt
import numpy as np
from src.models.rl.py_models.bandit.BernoulliBandit import BernoulliBandit
from src.models.rl.py_models.bandit.EpsilonGreedy import EpsilonGreedy
from src.models.rl.py_models.bandit.DecayingEpsilonGreedy import DecayingEpsilonGreedy
from src.models.rl.py_models.bandit.UCB import UCB
from src.models.rl.py_models.bandit.ThompsonSampling import ThompsonSampling


def plot_results(solvers, solver_names):
    """生成累积懊悔随时间变化的图像。
    输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称
    """
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()


def eg_demo_1(bandit_10_arm):
    np.random.seed(1)
    epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
    epsilon_greedy_solver.run(5000)
    print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
    plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])


def eg_demo_4(bandit_10_arm):
    np.random.seed(0)
    epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
    epsilon_greedy_solver_list = [
        EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilons
    ]
    epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
    for solver in epsilon_greedy_solver_list:
        solver.run(5000)
    plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)


def deg_demo(bandit_10_arm):
    np.random.seed(1)
    decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
    decaying_epsilon_greedy_solver.run(5000)
    print('epsilon值衰减的贪婪算法的累积懊悔为：', decaying_epsilon_greedy_solver.regret)
    plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])


def ucb_demo(bandit_10_arm):
    np.random.seed(1)
    coef = 1  # 控制不确定性比重的系数
    UCB_solver = UCB(bandit_10_arm, coef)
    UCB_solver.run(5000)
    print('上置信界算法的累积懊悔为：', UCB_solver.regret)
    plot_results([UCB_solver], ["UCB"])


def ts_demo(bandit_10_arm):
    np.random.seed(1)
    thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
    thompson_sampling_solver.run(5000)
    print('汤普森采样算法的累积懊悔为：', thompson_sampling_solver.regret)
    plot_results([thompson_sampling_solver], ["ThompsonSampling"])


K = 10
bandit_10_arm = BernoulliBandit(K)
# eg_demo_1(bandit_10_arm)
# deg_demo(bandit_10_arm)
# ucb_demo(bandit_10_arm)
ts_demo(bandit_10_arm)
