import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

# Individual Model (Execution) - Decentralized Actor
class UAVActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(UAVActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mean_layer = nn.Linear(64, action_dim)
        # log_std as a learnable parameter
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = torch.tanh(self.mean_layer(x)) # bounded actions
        std = self.log_std.exp().expand_as(mean)
        return mean, std
        
    def get_action(self, state):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.detach(), log_prob.detach()

# Global Model (Training) - Centralized Critic
class GlobalCritic(nn.Module):
    def __init__(self, global_state_dim):
        super(GlobalCritic, self).__init__()
        self.fc1 = nn.Linear(global_state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.value_layer = nn.Linear(64, 1)
        
    def forward(self, global_state):
        x = torch.relu(self.fc1(global_state))
        x = torch.relu(self.fc2(x))
        return self.value_layer(x)

class TFPPO:
    def __init__(self, num_uavs, state_dim=5, action_dim=2, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.num_uavs = num_uavs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        
        # CTDE: Independent actors, single global critic
        self.actors = [UAVActor(state_dim, action_dim) for _ in range(num_uavs)]
        self.critic = GlobalCritic(state_dim * num_uavs)
        
        # Optimizers
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=lr) for actor in self.actors]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.buffer = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'global_states': []}
        
    def select_action(self, global_state_list):
        actions = []
        log_probs = []
        # global_state_list is a list of local states
        for i, local_state in enumerate(global_state_list):
            state_tensor = torch.FloatTensor(local_state).unsqueeze(0)
            action, log_prob = self.actors[i].get_action(state_tensor)
            actions.append(action.numpy()[0])
            log_probs.append(log_prob.item())
        return actions, log_probs
        
    def store_transition(self, global_state, actions, log_probs, reward):
        # global_state: list of local states
        flat_global_state = np.concatenate(global_state)
        self.buffer['states'].append(global_state)
        self.buffer['global_states'].append(flat_global_state)
        self.buffer['actions'].append(actions)
        self.buffer['log_probs'].append(log_probs)
        self.buffer['rewards'].append(reward)
        
    def update(self):
        if len(self.buffer['rewards']) == 0:
            return
            
        rewards = []
        discounted_reward = 0
        for reward in reversed(self.buffer['rewards']):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        global_states = torch.FloatTensor(np.array(self.buffer['global_states']))
        
        # Critic update
        values = self.critic(global_states).squeeze()
        advantages = rewards - values.detach()
        critic_loss = nn.MSELoss()(values, rewards)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor updates
        for i in range(self.num_uavs):
            local_states = torch.FloatTensor(np.array([s[i] for s in self.buffer['states']]))
            old_actions = torch.FloatTensor(np.array([a[i] for a in self.buffer['actions']]))
            old_log_probs = torch.FloatTensor(np.array([lp[i] for lp in self.buffer['log_probs']]))
            
            mean, std = self.actors[i](local_states)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(old_actions).sum(dim=-1)
            
            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()
            
        # Clear buffer
        self.buffer = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'global_states': []}
