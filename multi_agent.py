import random

import torch
import numpy as np

from agent import Agent
from replay_buffer import ReplayBuffer
from torch.cuda.random import seed
import torch.optim as optim

from model import Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MultiAgent():

    def __init__(
        self, 
        n_agents, 
        state_size, 
        action_size,
        noise_type="normal", 
        random_seed=0,
        learn_every=1, 
        n_learn=8, 
        alpha_critic=1e-4,
        batch_size=64, 
        buffer_size=int(1e6), 
        ):
        
        # save hyperparameters for print
        self.hyperparameters = '\n'.join(f"{key:>17}: {value}" for key, value in locals().items() if key != 'self')

        # reset/seed all random generators
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.t = 0
        self.noise_type = noise_type

        self.n_agents = n_agents
        self.acttion_size = action_size
        self.learn_every = learn_every
        self.n_learn = n_learn

        self.batch_size = batch_size

        self.critic_regular = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic_regular.parameters(), lr=alpha_critic)

        # agents
        self.agents = [Agent(state_size, action_size, self.critic_regular, self.critic_target, self.critic_optimizer, noise_type) for _ in range(n_agents)]

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, random_seed)


    def reset(self):
        for agent in self.agents:
            agent.reset()


    def step(self, states, actions, rewards, next_states, dones):
        self.t += 1
        self.memory.add_multi(states, actions, rewards, next_states, dones)

        if not (self.t % self.learn_every == 0 and len(self.memory) > self.batch_size):
            return

        self.learn()


    def learn(self):
        for agent in self.agents:
            experiences = self.memory.sample()
            agent.learn(experiences)
        

    def act(self, states, add_noise=True):
        actions = np.zeros((self.n_agents, self.acttion_size), dtype=float)
        for index, agent in enumerate(self.agents):
            actions[index] = agent.act(states[index], add_noise)
        return actions


    def save(self, suffix=""):
        torch.save(self.critic_regular.state_dict(), f'saved_models/critic_{self.noise_type}_{suffix}.pth')
        for index, agent in enumerate(self.agents):
            torch.save(agent.actor_regular.state_dict(), f'saved_models/actor_{self.noise_type}_{index}_{suffix}.pth')


    def load(self, suffix=""):
        self.critic_regular.load_state_dict(torch.load(f'saved_models/critic_{self.noise_type}_{suffix}.pth'))
        for index, agent in enumerate(self.agents):
            agent.actor_regular.load_state_dict(torch.load(f'saved_models/actor_{self.noise_type}_{index}_{suffix}.pth'))
            
            
    def __str__(self):
        return (f"\n{'#'*80}\n\nHyperparameters (multiagent):\n\n{self.hyperparameters}\n\n{self.critic_regular}\n\n"
                f"Single Agent: {self.agents[0]} \n\n{'#'*80}\n\n")