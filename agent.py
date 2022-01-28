from collections import deque
from os import stat

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic
from noise import OUNoise, NormalNoise, ParameterNoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(
        self, 
        state_size, 
        action_size, 
        critic_regular,
        critic_target,
        critic_optimizer,
        noise_type="normal", 
        
        alpha_actor=1e-4, 

        gamma=.99, 
        tau=1e-3,
        desired_distance=.3, 
        scalar=.5, 
        scalar_decay=.99,
        normal_scalar=.4
        ):

        self.hyperparameters = '\n'.join(f"{key:>17}: {value}" for key, value in locals().items() if key not in ['self', 'critic_regular', 'critic_target', 'critic_optimizer'])
        
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.tau = tau

        self.noise_type = noise_type

        # parameter noise
        self.distances = []
        self.desired_distance = desired_distance
        self.scalar = scalar
        self.scalar_decay = scalar_decay

        # normal noise
        self.normal_scalar = normal_scalar

        # step counter
        self.t = 0

        # Actor Network (w/ Target Network)
        self.actor_regular = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_regular.parameters(), lr=alpha_actor)

        # Critic Network (w/ Target Network)
        self.critic_regular = critic_regular
        self.critic_target = critic_target
        self.critic_optimizer = critic_optimizer

        self.actor_noised = Actor(state_size, action_size).to(device)

        # hard update to ensure that regular and target start with same values
        self.soft_update(self.critic_regular, self.critic_target, 1.)
        self.soft_update(self.actor_regular, self.actor_target, 1.) 

        if noise_type == "ou":
            self.noise = OUNoise(self.action_size)
        elif noise_type == "param":
            self.noise = ParameterNoise(self.actor_noised, desired_distance, scalar, scalar_decay)
        else:
            self.noise = NormalNoise(self.action_size)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor_regular.eval()
        self.actor_noised.eval()
        with torch.no_grad():
            action = self.actor_regular(state).cpu().data.numpy()

            if add_noise:
                action = self.noise(action, state, self.actor_regular)

        self.actor_regular.train()
        return np.clip(action, -1, 1)

    # needed for ou noise only
    def reset(self):
        if hasattr(self, 'ou_noise'):
            self.ou_noise.reset()

    def learn(self, experiences):

        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_regular(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_regular(states)
        actor_loss = -self.critic_regular(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_regular, self.critic_target, self.tau)
        self.soft_update(self.actor_regular, self.actor_target, self.tau)                     

    @staticmethod
    def soft_update(regular_model, target_model, tau):
        for target_param, regular_param in zip(target_model.parameters(), regular_model.parameters()):
            target_param.data.copy_(tau*regular_param.data + (1.0-tau)*target_param.data)

    def __str__(self):
        return f"Hyperparameters: \n\n{self.hyperparameters}\n\n{self.actor_regular}"



