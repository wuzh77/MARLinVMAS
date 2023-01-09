import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

print("============================================================================================")

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.state_dim = state_dim
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space:
            # MAPPO创建两个智能体，两个智能体分别采用两个actor网络，输入不同的状态分别得到结果。
            self.actor1 = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
            )
            self.actor2 = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            )
        # 这个是MAPPO的critic网络，两个智能体采用一个critic网络，输入为两个智能体的共同状态空间。
        self.critic = nn.Sequential(
            nn.Linear(state_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def set_action_std(self, new_action_std):  # 设置智能体动作网络采样的高斯函数方差
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act1(self, state):  # 获取智能体1的动作
        if self.has_continuous_action_space:
            action_mean = self.actor1(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def act2(self, state):  # 获取智能体2的动作
        if self.has_continuous_action_space:
            action_mean = self.actor2(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate1(self, state, action):  # 智能体1动作的评估函数， 输入的state是用来评估这个状态下的价值。

        if self.has_continuous_action_space:  # 获取动作
            actorState = state[:, :self.state_dim]
            action_mean = self.actor1(actorState)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy

    def evaluate2(self, state, action):  # 智能体2动作的评估函数， 输入的state是用来评估这个状态下的价值。

        if self.has_continuous_action_space:
            actorState = state[:, self.state_dim:]
            action_mean = self.actor2(actorState)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class MAPPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer1 = RolloutBuffer()  # buffer for actor1
        self.buffer2 = RolloutBuffer()  # buffer for actor2

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        # 这一个优化器用来优化智能体1的策略网络和公共的critic网络
        self.optimizer1 = torch.optim.Adam([
            {'params': self.policy.actor1.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        # 这一个优化器用来优化智能体2的actor网络和公共的critic网络
        self.optimizer2 = torch.optim.Adam([
            {'params': self.policy.actor2.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        # 这里采用均方损失函数
        self.MseLoss = nn.MSELoss().to(device)

    def set_action_std(self, new_action_std):  # 这个函数用来修改高斯函数方差
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):  # 采取方差衰退，随着训练的进行，高斯函数取值到均值以外的取值概率减小
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state1, state2):  # 由输入的两个状态产生动作

        if self.has_continuous_action_space:
            with torch.no_grad():
                state1 = torch.FloatTensor(state1).to(device)
                state2 = torch.FloatTensor(state2).to(device)
                action1, action_logprob1 = self.policy_old.act1(state1)
                action2, action_logprob2 = self.policy_old.act2(state2)
            state = torch.cat([state1, state2])
            self.buffer1.states.append(state)
            self.buffer1.actions.append(action1)
            self.buffer1.logprobs.append(action_logprob1)

            self.buffer2.states.append(state)
            self.buffer2.actions.append(action2)
            self.buffer2.logprobs.append(action_logprob2)
            return action1.detach().flatten().to(device), action2.detach().flatten().to(device)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state1).to(device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()

    def update(self):
        # 更新actor1和actor2以及critic网络。
        rewards1 = []
        discounted_reward1 = 0
        for reward, is_terminal in zip(reversed(self.buffer1.rewards), reversed(self.buffer1.is_terminals)):
            if is_terminal:
                discounted_reward1 = 0
            discounted_reward1 = reward + (self.gamma * discounted_reward1)
            rewards1.insert(0, discounted_reward1)

        # 奖励归一化
        rewards1 = torch.tensor(rewards1, dtype=torch.float32).to(device)
        rewards1 = (rewards1 - rewards1.mean()) / (rewards1.std() + 1e-7)

        rewards2 = []
        discounted_reward2 = 0
        for reward, is_terminal in zip(reversed(self.buffer2.rewards), reversed(self.buffer1.is_terminals)):
            if is_terminal:
                discounted_reward2 = 0
            discounted_reward2 = reward + (self.gamma * discounted_reward2)
            rewards2.insert(0, discounted_reward2)

        rewards2 = torch.tensor(rewards2, dtype=torch.float32).to(device)
        rewards2 = (rewards2 - rewards2.mean()) / (rewards2.std() + 1e-7)
        # convert list to tensor
        old_states1 = torch.squeeze(torch.stack(self.buffer1.states, dim=0)).detach().to(device)
        old_actions1 = torch.squeeze(torch.stack(self.buffer1.actions, dim=0)).detach().to(device)
        old_logprobs1 = torch.squeeze(torch.stack(self.buffer1.logprobs, dim=0)).detach().to(device)

        old_states2 = torch.squeeze(torch.stack(self.buffer2.states, dim=0)).detach().to(device)
        old_actions2 = torch.squeeze(torch.stack(self.buffer2.actions, dim=0)).detach().to(device)
        old_logprobs2 = torch.squeeze(torch.stack(self.buffer2.logprobs, dim=0)).detach().to(device)
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate1(old_states1, old_actions1)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs1.detach())

            # Finding Surrogate Loss
            advantages = rewards1 - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards1) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer1.zero_grad()
            loss.mean().backward()
            self.optimizer1.step()

            # update agent2
            logprobs, state_values, dist_entropy = self.policy.evaluate2(old_states2, old_actions2)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs2.detach())

            # Finding Surrogate Loss
            advantages = rewards2 - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards2) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer2.zero_grad()
            loss.mean().backward()
            self.optimizer2.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer1.clear()
        self.buffer2.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
