from tqdm import tqdm
import numpy as np
import torch
import collections
import random


def train_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'action': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()  # for each episode, reset the environment first.
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def train_IPPO(env, agent1, agent2, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
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
                s = torch.cat(s, dim=0)
                s.squeeze()
                terminal = False
                step = 0
                while not terminal and step < 200:
                    step += 1
                    a_1 = agent1.take_action(s[0])
                    a_2 = agent2.take_action(s[1])
                    a_1 = torch.LongTensor([a_1])
                    a_1.unsqueeze(0)
                    a_2 = torch.LongTensor([a_2])
                    a_2.unsqueeze(0)
                    next_s, r, done1, _ = env.step([a_1, a_2])
                    done = done1[0]
                    next_s = torch.cat(next_s, dim=0)
                    next_s.squeeze()
                    r = torch.Tensor(r)
                    r.squeeze()
                    transition_dict_1['states'].append(s[0])
                    transition_dict_1['actions'].append(a_1)
                    transition_dict_1['next_states'].append(next_s[0])
                    transition_dict_1['rewards'].append(r[0])
                    transition_dict_1['dones'].append(done)
                    transition_dict_2['states'].append(s[1])
                    transition_dict_2['actions'].append(a_2)
                    transition_dict_2['next_states'].append(next_s[1])
                    transition_dict_2['rewards'].append(r[1])
                    transition_dict_2['dones'].append(done)
                    s = next_s
                    terminal = done
                    episode_return += r[0]
                return_list.append(episode_return)
                agent1.update(transition_dict_1)
                agent2.update(transition_dict_2)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list
