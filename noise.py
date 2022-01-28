from collections import deque
import copy
import numpy as np

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.6, sigma_min=.05, decay_rate=.99995):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma_start = sigma
        self.sigma_min = sigma_min
        self.decay_rate = decay_rate
        self.t = 0
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        self.t += 1
        self.sigma = max(self.sigma_start*self.decay_rate**self.t, self.sigma_min)
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state
    
    def __call__(self, action, *args):
        return action + self.sample()
    
    def __str__(self):
        return f't: {self.t:06d} sigma: {self.sigma:.4f}'
    
    
class NormalNoise:

    def __init__(self, size, sigma=.6, sigma_min=.05, decay_rate=.99999):
        self.sigma_start = sigma
        self.sigma_min = sigma_min
        self.decay_rate = decay_rate
        self.size = size
        self.t = 0

    def sample(self):
        self.t += 1
        self.sigma = max(self.sigma_start*self.decay_rate**self.t, self.sigma_min)
        return np.random.randn(self.size) * self.sigma
    
    def __call__(self, action, *args):
        return action + self.sample()
    
    def __str__(self):
        return f't: {self.t:06d} sigma: {self.sigma:.4f}'


class ParameterNoise:
    
    def __init__(self, actor_noised, desired_distance, scalar, scalar_decay, desired_distance_min=.1, desired_distance_decay_rate=.99999):
        self.actor_noised = actor_noised
        self.desired_distance_start = desired_distance
        self.scalar = scalar
        self.scalar_decay = scalar_decay
        self.desired_distance_min = desired_distance_min
        self.desired_distance_decay_rate = desired_distance_decay_rate
        self.t = 0
        self.distances = []
        
    def __call__(self, action, state, actor_regular):
        self.t += 1
        self.desired_distance = max(self.desired_distance_start*self.desired_distance_decay_rate**self.t, self.desired_distance_min)
        # hard copy the actor_regular to actor_noised
        self.actor_noised.load_state_dict(actor_regular.state_dict().copy())
        # add noise to the copy
        self.actor_noised.add_parameter_noise(self.scalar)
        # get the next action values from the noised actor
        action_noised = self.actor_noised(state).cpu().data.numpy()
        # meassure the distance between the action values from the regular and 
        # the noised actor to adjust the amount of noise that will be added next round
        distance = np.sqrt(np.mean(np.square(action-action_noised)))
        # for stats and print only
        self.distances.append(distance)
        # adjust the amount of noise given to the actor_noised
        if distance > self.desired_distance:
            self.scalar *= self.scalar_decay
        if distance < self.desired_distance:
            self.scalar /= self.scalar_decay
        # set the noised action as action
        return action_noised
    
    def __str__(self):
        return f't: {self.t:06d} dis: {np.mean(self.distances[-10:]):.3f} sclr: {self.scalar:.4f} des_dis: {self.desired_distance:.4f}' 