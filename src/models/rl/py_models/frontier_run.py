"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-12-23 下午4:58
# @FileName: frontier_run.py
# @Email   : quant_master2000@163.com
======================
"""
import torch
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import rl_utils
from src.models.rl.py_models.mixbased.onpolicy.PPO import PPO
from src.models.rl.py_models.frontier.imitation.BehaviorClone import BehaviorClone
from src.models.rl.py_models.frontier.imitation.GAIL import GAIL
from src.models.rl.py_models.valuebased.modelfree.offpolicy.dqn.ReplayBuffer import ReplayBuffer
from src.models.rl.py_models.frontier.modelbased.mpc.PETS import PETS
from src.models.rl.py_models.mixbased.offpolicy.sac.SACContinuous import SACContinuous
from src.models.rl.py_models.mixbased.offpolicy.DDPG import DDPG
from src.models.rl.py_models.frontier.modelbased.mpc.EnsembleDynamicsModel import EnsembleDynamicsModel
from src.models.rl.py_models.frontier.modelbased.mpc.FakeEnv import FakeEnv
from src.models.rl.py_models.frontier.modelbased.mbpo.ReplayBuffer import ReplayBuffer as ReplayBuffer_mbpo
from src.models.rl.py_models.frontier.modelbased.mbpo.MBPO import MBPO
from src.models.rl.py_models.frontier.offrl.CQL import CQL
from src.models.rl.py_models.frontier.gorl.WorldEnv import WorldEnv
from src.models.rl.py_models.frontier.gorl.Trajectory import Trajectory
from src.models.rl.py_models.frontier.gorl.ReplayBuffer_Trajectory import ReplayBuffer_Trajectory
import src.models.rl.py_models.frontier.marl.PPO as IPPO
from src.models.rl.py_models.frontier.marl.MADDPG import MADDPG


def sample_expert_data(n_episode, ppo_agent):
    states = []
    actions = []
    for episode in range(n_episode):
        state = env.reset()
        done = False
        while not done:
            action = ppo_agent.take_action(state)
            states.append(state)
            actions.append(action)
            next_state, reward, done, _ = env.step(action)
            state = next_state
    return np.array(states), np.array(actions)


def get_expert(env, agent):
    env.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    n_episode = 1
    expert_s, expert_a = sample_expert_data(n_episode, agent)

    n_samples = 30  # 采样30个数据
    # random_index = random.sample(range(expert_s.shape[0]), n_samples)
    random_index = np.random.choice(range(expert_s.shape[0]), n_samples)
    expert_s = expert_s[random_index]
    expert_a = expert_a[random_index]
    return expert_s, expert_a


def test_agent(agent, env, n_episode):
    return_list = []
    for episode in range(n_episode):
        episode_return = 0
        state = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_return += reward
        return_list.append(episode_return)
    return np.mean(return_list)


def imitation_bc_demo(env, agent, device):
    env.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    expert_s, expert_a = get_expert(env, agent)

    lr = 1e-3
    bc_agent = BehaviorClone(state_dim, hidden_dim, action_dim, lr, device)
    n_iterations = 1000
    batch_size = 64
    test_returns = []

    with tqdm(total=n_iterations, desc="进度条") as pbar:
        for i in range(n_iterations):
            sample_indices = np.random.randint(low=0,
                                               high=expert_s.shape[0],
                                               size=batch_size)
            bc_agent.learn(expert_s[sample_indices], expert_a[sample_indices])
            current_return = test_agent(bc_agent, env, 5)
            test_returns.append(current_return)
            if (i + 1) % 10 == 0:
                pbar.set_postfix({'return': '%.3f' % np.mean(test_returns[-10:])})
            pbar.update(1)

    iteration_list = list(range(len(test_returns)))
    plt.plot(iteration_list, test_returns)
    plt.xlabel('Iterations')
    plt.ylabel('Returns')
    plt.title('BC on {}'.format(env_name))
    plt.show()


def imitation_gail_demo(env, agent, device):
    env.seed(0)
    torch.manual_seed(0)
    lr_d = 1e-3
    # agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)
    gail = GAIL(agent, state_dim, action_dim, hidden_dim, lr_d, device)
    expert_s, expert_a = get_expert(env, agent)

    n_episode = 500
    return_list = []

    with tqdm(total=n_episode, desc="进度条") as pbar:
        for i in range(n_episode):
            episode_return = 0
            state = env.reset()
            done = False
            state_list = []
            action_list = []
            next_state_list = []
            done_list = []
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                state_list.append(state)
                action_list.append(action)
                next_state_list.append(next_state)
                done_list.append(done)
                state = next_state
                episode_return += reward
            return_list.append(episode_return)
            gail.learn(expert_s, expert_a, state_list, action_list,
                       next_state_list, done_list)
            if (i + 1) % 10 == 0:
                pbar.set_postfix({'return': '%.3f' % np.mean(return_list[-10:])})
            pbar.update(1)

    iteration_list = list(range(len(return_list)))
    plt.plot(iteration_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('GAIL on {}'.format(env_name))
    plt.show()


def mb_mpc_demo(device):
    buffer_size = 100000
    n_sequence = 50
    elite_ratio = 0.2
    plan_horizon = 25
    num_episodes = 10
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)

    replay_buffer = ReplayBuffer(buffer_size)
    pets = PETS(env, replay_buffer, n_sequence, elite_ratio, plan_horizon, num_episodes, device)
    return_list = pets.train()

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PETS on {}'.format(env_name))
    plt.show()


def mb_mbpo_demo(device):
    real_ratio = 0.5
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    num_episodes = 20
    actor_lr = 5e-4
    critic_lr = 5e-3
    alpha_lr = 1e-3
    hidden_dim = 128
    gamma = 0.98
    tau = 0.005  # 软更新参数
    buffer_size = 10000
    target_entropy = -1
    model_alpha = 0.01  # 模型损失函数中的加权权重
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]  # 动作最大值

    rollout_batch_size = 1000
    rollout_length = 1  # 推演长度k,推荐更多尝试
    model_pool_size = rollout_batch_size * rollout_length

    agent = SACContinuous(state_dim, hidden_dim, action_dim, action_bound, actor_lr,
                critic_lr, alpha_lr, target_entropy, tau, gamma, device)
    # model = EnsembleDynamicsModel(state_dim, action_dim, device, model_alpha)
    model = EnsembleDynamicsModel(state_dim, action_dim, device)
    fake_env = FakeEnv(model)
    env_pool = ReplayBuffer_mbpo(buffer_size)
    model_pool = ReplayBuffer_mbpo(model_pool_size)
    mbpo = MBPO(env, agent, fake_env, env_pool, model_pool, rollout_length,
                rollout_batch_size, real_ratio, num_episodes)

    return_list = mbpo.train()

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('MBPO on {}'.format(env_name))
    plt.show()


def offrl_demo():
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)

    beta = 5.0
    num_random = 5
    num_epochs = 100
    num_trains_per_epoch = 500

    agent = CQL(state_dim, hidden_dim, action_dim, action_bound, actor_lr,
                critic_lr, alpha_lr, target_entropy, tau, gamma, device, beta,
                num_random)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_epochs / 10), desc='Iteration %d' % i) as pbar:
            for i_epoch in range(int(num_epochs / 10)):
                # 此处与环境交互只是为了评估策略,最后作图用,不会用于训练
                epoch_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    state = next_state
                    epoch_return += reward
                return_list.append(epoch_return)

                for _ in range(num_trains_per_epoch):
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(transition_dict)

                if (i_epoch + 1) % 10 == 0:
                    pbar.set_postfix({
                        'epoch':
                            '%d' % (num_epochs / 10 * i + i_epoch + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    epochs_list = list(range(len(return_list)))
    plt.plot(epochs_list, return_list)
    plt.xlabel('Epochs')
    plt.ylabel('Returns')
    plt.title('CQL on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('CQL on {}'.format(env_name))
    plt.show()


def gorl_demo():
    actor_lr = 1e-3
    critic_lr = 1e-3
    hidden_dim = 128
    state_dim = 4
    action_dim = 2
    action_bound = 1
    sigma = 0.1
    tau = 0.005
    gamma = 0.98
    num_episodes = 2000
    n_train = 20
    batch_size = 256
    minimal_episodes = 200
    buffer_size = 10000
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    env = WorldEnv()
    replay_buffer = ReplayBuffer_Trajectory(buffer_size)
    agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, actor_lr,
                 critic_lr, sigma, tau, gamma, device)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                traj = Trajectory(state)
                done = False
                while not done:
                    action = agent.take_action(state)
                    state, reward, done = env.step(action)
                    episode_return += reward
                    traj.store_step(action, state, reward, done)
                replay_buffer.add_trajectory(traj)
                return_list.append(episode_return)
                if replay_buffer.size() >= minimal_episodes:
                    for _ in range(n_train):
                        transition_dict = replay_buffer.sample(batch_size, True)
                        agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DDPG with HER on {}'.format('GridWorld'))
    plt.show()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    env = WorldEnv()
    replay_buffer = ReplayBuffer_Trajectory(buffer_size)
    agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, actor_lr,
                 critic_lr, sigma, tau, gamma, device)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                traj = Trajectory(state)
                done = False
                while not done:
                    action = agent.take_action(state)
                    state, reward, done = env.step(action)
                    episode_return += reward
                    traj.store_step(action, state, reward, done)
                replay_buffer.add_trajectory(traj)
                return_list.append(episode_return)
                if replay_buffer.size() >= minimal_episodes:
                    for _ in range(n_train):
                        # 和使用HER训练的唯一区别
                        transition_dict = replay_buffer.sample(batch_size, False)
                        agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DDPG without HER on {}'.format('GridWorld'))
    plt.show()


def marl_ippo_demo():
    actor_lr = 3e-4
    critic_lr = 1e-3
    num_episodes = 100000
    hidden_dim = 64
    gamma = 0.99
    lmbda = 0.97
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    team_size = 2
    grid_size = (15, 15)
    # 创建Combat环境，格子世界的大小为15x15，己方智能体和敌方智能体数量都为2
    env = Combat(grid_shape=grid_size, n_agents=team_size, n_opponents=team_size)

    state_dim = env.observation_space[0].shape[0]
    action_dim = env.action_space[0].n
    # 两个智能体共享同一个策略
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, eps, gamma, device)

    win_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                transition_dict_1 = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
                transition_dict_2 = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
                s = env.reset()
                terminal = False
                while not terminal:
                    a_1 = agent.take_action(s[0])
                    a_2 = agent.take_action(s[1])
                    next_s, r, done, info = env.step([a_1, a_2])
                    transition_dict_1['states'].append(s[0])
                    transition_dict_1['actions'].append(a_1)
                    transition_dict_1['next_states'].append(next_s[0])
                    transition_dict_1['rewards'].append(
                        r[0] + 100 if info['win'] else r[0] - 0.1)
                    transition_dict_1['dones'].append(False)
                    transition_dict_2['states'].append(s[1])
                    transition_dict_2['actions'].append(a_2)
                    transition_dict_2['next_states'].append(next_s[1])
                    transition_dict_2['rewards'].append(
                        r[1] + 100 if info['win'] else r[1] - 0.1)
                    transition_dict_2['dones'].append(False)
                    s = next_s
                    terminal = all(done)
                win_list.append(1 if info["win"] else 0)
                agent.update(transition_dict_1)
                agent.update(transition_dict_2)
                if (i_episode + 1) % 100 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(win_list[-100:])
                    })
                pbar.update(1)

    win_array = np.array(win_list)
    # 每100条轨迹取一次平均
    win_array = np.mean(win_array.reshape(-1, 100), axis=1)

    episodes_list = np.arange(win_array.shape[0]) * 100
    plt.plot(episodes_list, win_array)
    plt.xlabel('Episodes')
    plt.ylabel('Win rate')
    plt.title('IPPO on Combat')
    plt.show()


def marl_maddpg_demo():

    def evaluate(env_id, maddpg, n_episode=10, episode_length=25):
        # 对学习的策略进行评估,此时不会进行探索
        env = make_env(env_id)
        returns = np.zeros(len(env.agents))
        for _ in range(n_episode):
            obs = env.reset()
            for t_i in range(episode_length):
                actions = maddpg.take_action(obs, explore=False)
                obs, rew, done, info = env.step(actions)
                rew = np.array(rew)
                returns += rew / n_episode
        return returns.tolist()

    num_episodes = 5000
    episode_length = 25  # 每条序列的最大长度
    buffer_size = 100000
    hidden_dim = 64
    actor_lr = 1e-2
    critic_lr = 1e-2
    gamma = 0.95
    tau = 1e-2
    batch_size = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    update_interval = 100
    minimal_size = 4000

    env_id = "simple_adversary"
    env = make_env(env_id)
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)

    state_dims = []
    action_dims = []
    for action_space in env.action_space:
        action_dims.append(action_space.n)
    for state_space in env.observation_space:
        state_dims.append(state_space.shape[0])
    critic_input_dim = sum(state_dims) + sum(action_dims)

    maddpg = MADDPG(env, device, actor_lr, critic_lr, hidden_dim, state_dims,
                    action_dims, critic_input_dim, gamma, tau)

    return_list = []  # 记录每一轮的回报（return）
    total_step = 0
    for i_episode in range(num_episodes):
        state = env.reset()
        # ep_returns = np.zeros(len(env.agents))
        for e_i in range(episode_length):
            actions = maddpg.take_action(state, explore=True)
            next_state, reward, done, _ = env.step(actions)
            replay_buffer.add(state, actions, reward, next_state, done)
            state = next_state

            total_step += 1
            if replay_buffer.size(
            ) >= minimal_size and total_step % update_interval == 0:
                sample = replay_buffer.sample(batch_size)

                def stack_array(x):
                    rearranged = [[sub_x[i] for sub_x in x]
                                  for i in range(len(x[0]))]
                    return [
                        torch.FloatTensor(np.vstack(aa)).to(device)
                        for aa in rearranged
                    ]

                sample = [stack_array(x) for x in sample]
                for a_i in range(len(env.agents)):
                    maddpg.update(sample, a_i)
                maddpg.update_all_targets()
        if (i_episode + 1) % 100 == 0:
            ep_returns = evaluate(env_id, maddpg, n_episode=100)
            return_list.append(ep_returns)
            print(f"Episode: {i_episode + 1}, {ep_returns}")


actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 250
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
ppo_agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)

# return_list = rl_utils.train_on_policy_agent(env, ppo_agent, num_episodes)

# imitation_bc_demo(env, ppo_agent, device)
# imitation_gail_demo(env, ppo_agent, device)
# mb_mpc_demo(device)
mb_mbpo_demo(device)
