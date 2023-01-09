import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym

from PPO import PPO
from vmas import make_env, Wrapper

reward_per_print = []


################################### Training ###################################
def train():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = "wheel"

    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 200  # max timesteps in one episode
    max_training_timesteps = int(1e6)  # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10  # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)  # save model frequency (in num timesteps)

    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4  # update policy every n timesteps
    K_epochs = 80  # update policy for K epochs in one PPO update

    eps_clip = 0.5  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.00005  # learning rate for actor network 0.00005
    lr_critic = 0.00005  # learning rate for critic network

    random_seed = 0  # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)

    env = make_env(
        scenario_name=env_name,
        num_envs=1,
        device='cpu',
        continuous_actions=has_continuous_action_space,
        wrapper=None,
        n_agents=2
    )

    # state space dimension
    state_dim = env.observation_space[0].shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = 2
    else:
        action_dim = 5

    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent1 = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                     action_std)
    ppo_agent2 = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                     action_std)
    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0
        flag = True
        for t in range(1, max_ep_len + 1):
            # select action with policy
            state1 = torch.squeeze(state[0])
            state2 = torch.squeeze(state[1])
            action1 = ppo_agent1.select_action(state1)
            action1 = torch.clamp(action1, -1.0, 1.0)
            action2 = ppo_agent2.select_action(state2)
            action2 = torch.clamp(action2, -1.0, 1.0)
            action1 = torch.unsqueeze(action1, dim=0)
            action2 = torch.unsqueeze(action2, dim=0)
            state, reward, done, _ = env.step([action1, action2])

            rews1 = float(reward[0][0])
            rews2 = float(reward[1][0])
            # saving reward and is_terminals
            ppo_agent1.buffer.rewards.append(rews1)
            ppo_agent1.buffer.is_terminals.append(bool(done[0]))
            ppo_agent2.buffer.rewards.append(rews2)
            ppo_agent2.buffer.is_terminals.append(bool(done[0]))
            time_step += 1
            if flag:
                current_ep_reward += (rews1 + rews2) / 2

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent1.update()
                ppo_agent2.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent1.decay_action_std(action_std_decay_rate, min_action_std)
                ppo_agent2.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)
                print("Episode : {} \t\t Timestep : {}/{} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                           max_training_timesteps,
                                                                                           print_avg_reward))
                reward_per_print.append(print_avg_reward)
                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if bool(done[0]):
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    train()
    finalSave = np.array(reward_per_print)
    np.save('IPPO_wheel_5.npy', finalSave)
