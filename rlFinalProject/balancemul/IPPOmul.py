import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys
import balanceEnvir
import trainAgent

'''using fully connected linear network to create Policy network.'''

num_envs = 300
def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.state_dim = state_dim
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.func2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.func3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.reshape(x, (-1, 300, self.state_dim))
        x = F.relu(self.func2(F.relu(self.fc1(x))))
        x = self.func3(x)
        return F.softmax(x, dim=2)


'''This is value network created by fully connected linear network'''


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.state_dim = state_dim
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.reshape(x, (-1, 300, self.state_dim))
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    '''Continuous action IPPO'''

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device,
                 num_envs=300):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.lmbda = lmbda
        self.num_evns = num_envs

    def take_action(self, state):
        state.unsqueeze(0)
        probs = self.actor(state)
        probs = probs.squeeze()
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        action = action.unsqueeze(1)
        return action

    def update(self, transition_dict):
        states = torch.stack(transition_dict['states'])
        actions = torch.stack(transition_dict['actions'])
        rewards = torch.stack(transition_dict['rewards'])
        next_states = torch.stack(transition_dict['next_states'])
        dones = torch.stack(transition_dict['dones'])
        dones = dones.float()
        rewards = rewards.squeeze()
        td_target = rewards + self.gamma * self.critic(next_states).squeeze() * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states).squeeze()
        advantage = compute_advantage(self.gamma, self.lmbda,
                                      td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)

            ratio = ratio.squeeze()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states).squeeze(), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


if __name__ == "__main__":
    env = balanceEnvir.make_Balance_env('wheel', 2, False, num_envs)  # create environment
    lmbda = 0.95
    actor_lr = 1e-5
    critic_lr = 1e-5
    num_episodes = 500
    state_dim = 13
    hidden_dim = 128
    action_dim = 5
    epochs = 10
    gamma = 0.98
    eps = 0.2
    device = "cpu"
    agent1 = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device)
    agent2 = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device)
    result = trainAgent.train_IPPO(env, agent1, agent2, num_episodes)
    win_array = np.array(result)
    win_array = np.mean(win_array.reshape(-1, 10), axis=1)

    episodes_list = np.arange(win_array.shape[0]) * 10
    plt.plot(episodes_list, win_array)
    plt.xlabel('Episodes')
    plt.ylabel('rewards')
    plt.title('IPPO')
    plt.show()
