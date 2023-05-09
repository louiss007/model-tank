"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-12-23 上午11:10
# @FileName: mb_run.py
# @Email   : quant_master2000@163.com
======================
"""
import torch
import gym
import numpy as np
import random
from utils import rl_utils
import matplotlib.pyplot as plt
from src.models.rl.py_models.mixbased.onpolicy.ActorCritic import ActorCritic
from src.models.rl.py_models.mixbased.onpolicy.TRPO import TRPO
from src.models.rl.py_models.mixbased.onpolicy.TRPOContinuous import TRPOContinuous
from src.models.rl.py_models.mixbased.onpolicy.PPO import PPO
from src.models.rl.py_models.mixbased.onpolicy.PPOContinuous import PPOContinuous
from src.models.rl.py_models.mixbased.offpolicy.DDPG import DDPG
from src.models.rl.py_models.mixbased.offpolicy.sac.SAC import SAC
from src.models.rl.py_models.mixbased.offpolicy.sac.SACContinuous import SACContinuous


def ac_demo():
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    env.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                        gamma, device)

    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Actor-Critic on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Actor-Critic on {}'.format(env_name))
    plt.show()


def trpo_demo():
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    critic_lr = 1e-2
    kl_constraint = 0.0005
    alpha = 0.5
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    env.seed(0)
    torch.manual_seed(0)
    agent = TRPO(hidden_dim, env.observation_space, env.action_space, lmbda,
                 kl_constraint, alpha, critic_lr, gamma, device)
    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('TRPO on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('TRPO on {}'.format(env_name))
    plt.show()


def trpo_continuous_demo():
    num_episodes = 2000
    hidden_dim = 128
    gamma = 0.9
    lmbda = 0.9
    critic_lr = 1e-2
    kl_constraint = 0.00005
    alpha = 0.5
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    env.seed(0)
    torch.manual_seed(0)
    agent = TRPOContinuous(hidden_dim, env.observation_space, env.action_space,
                           lmbda, kl_constraint, alpha, critic_lr, gamma, device)
    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('TRPO on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('TRPO on {}'.format(env_name))
    plt.show()


def ppo_demo():
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    env.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma, device)

    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))
    plt.show()


def ppo_continuous_demo():
    actor_lr = 1e-4
    critic_lr = 5e-3
    num_episodes = 2000
    hidden_dim = 128
    gamma = 0.9
    lmbda = 0.9
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    env.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]  # 连续动作空间
    agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                          lmbda, epochs, eps, gamma, device)

    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 21)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))
    plt.show()


def ddpg_demo():
    actor_lr = 3e-4
    critic_lr = 3e-3
    num_episodes = 200
    hidden_dim = 64
    gamma = 0.98
    tau = 0.005  # 软更新参数
    buffer_size = 10000
    minimal_size = 1000
    batch_size = 64
    sigma = 0.01  # 高斯噪声标准差
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]  # 动作最大值
    agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)

    return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DDPG on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DDPG on {}'.format(env_name))
    plt.show()


def sac_demo():
    actor_lr = 1e-3
    critic_lr = 1e-2
    alpha_lr = 1e-2
    num_episodes = 200
    hidden_dim = 128
    gamma = 0.98
    tau = 0.005  # 软更新参数
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    target_entropy = -1
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = SAC(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, alpha_lr,
                target_entropy, tau, gamma, device)

    return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes,
                                                  replay_buffer, minimal_size,
                                                  batch_size)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SAC on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SAC on {}'.format(env_name))
    plt.show()


def sac_continuous_demo():
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]  # 动作最大值
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)

    actor_lr = 3e-4
    critic_lr = 3e-3
    alpha_lr = 3e-4
    num_episodes = 100
    hidden_dim = 128
    gamma = 0.99
    tau = 0.005  # 软更新参数
    buffer_size = 100000
    minimal_size = 1000
    batch_size = 64
    target_entropy = -env.action_space.shape[0]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    agent = SACContinuous(state_dim, hidden_dim, action_dim, action_bound,
                          actor_lr, critic_lr, alpha_lr, target_entropy, tau,
                          gamma, device)

    return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes,
                                                  replay_buffer, minimal_size,
                                                  batch_size)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SAC on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SAC on {}'.format(env_name))
    plt.show()


# trpo_demo()
# trpo_continuous_demo()
ppo_demo()
# ppo_continuous_demo()
# ddpg_demo()
# sac_demo()
# sac_continuous_demo()
