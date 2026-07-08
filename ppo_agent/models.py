import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class UAVActor(nn.Module):
    def __init__(self, state_dim=12, action_dim=2):
        super(UAVActor, self).__init__()
        # State: [X, Y, Z, Load, MD_N, MD_NE, MD_E, MD_SE, MD_S, MD_SW, MD_W, MD_NW]
        # Action: [angle, distance]
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mean_layer = nn.Linear(128, action_dim)
        
        # log_std as a learnable parameter, initialized to 0 (std=1)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.mean_layer(x)) # Bounded between -1 and 1
        
        std = self.log_std.exp().expand_as(mean)
        return mean, std
        
    def get_action(self, state):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()
        # Bound actions physically in the server script:
        # angle = (action[0] + 1) * pi
        # distance = ((action[1] + 1) / 2) * max_dist
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.detach(), log_prob.detach()
        
    def evaluate(self, state, action):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

class UAVCritic(nn.Module):
    def __init__(self, state_dim=12):
        super(UAVCritic, self).__init__()
        # Individual local critic
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.v_layer = nn.Linear(128, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v_local = self.v_layer(x)
        return v_local

class GlobalTaskFactorizer(nn.Module):
    """
    QMIX-style monotonic mixing network for PPO (Value Factorization).
    Ensures Individual-Global-Max (IGM) principle.
    Combines individual V_k into a global V_total.
    """
    def __init__(self, num_agents=12, global_state_dim=144, embed_dim=64):
        super(GlobalTaskFactorizer, self).__init__()
        self.num_agents = num_agents
        self.embed_dim = embed_dim
        self.global_state_dim = global_state_dim
        
        # Hypernetwork generating weights for the first mixing layer
        # Weights must be strictly non-negative for monotonicity
        self.hyper_w1 = nn.Sequential(
            nn.Linear(global_state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_agents * embed_dim)
        )
        self.hyper_b1 = nn.Linear(global_state_dim, embed_dim)
        
        # Hypernetwork generating weights for the final mixing layer
        self.hyper_w2 = nn.Sequential(
            nn.Linear(global_state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim * 1)
        )
        # Final bias does not need to be positive
        self.hyper_b2 = nn.Sequential(
            nn.Linear(global_state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
        
    def forward(self, local_vs, global_state):
        # local_vs shape: (batch_size, num_agents)
        # global_state shape: (batch_size, global_state_dim)
        batch_size = local_vs.size(0)
        
        local_vs = local_vs.view(batch_size, 1, self.num_agents)
        
        # Layer 1
        w1 = torch.abs(self.hyper_w1(global_state)) # Absolute for non-negativity
        w1 = w1.view(batch_size, self.num_agents, self.embed_dim)
        b1 = self.hyper_b1(global_state)
        b1 = b1.view(batch_size, 1, self.embed_dim)
        
        hidden = F.elu(torch.bmm(local_vs, w1) + b1)
        
        # Layer 2
        w2 = torch.abs(self.hyper_w2(global_state)) # Absolute for non-negativity
        w2 = w2.view(batch_size, self.embed_dim, 1)
        b2 = self.hyper_b2(global_state).view(batch_size, 1, 1)
        
        v_total = torch.bmm(hidden, w2) + b2
        v_total = v_total.view(batch_size, 1)
        return v_total
