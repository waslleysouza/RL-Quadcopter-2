import random
import numpy as np
import pandas as pd
import torch

from collections import namedtuple, deque
from datetime import datetime
from torch.autograd import Variable


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = None
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


def get_timestamp(t=None, format='%Y-%m-%d_%H-%M-%S'):
    """Return timestamp as a string; default: current time, format: YYYY-DD-MM_hh-mm-ss."""
    if t is None:
        t = datetime.now()
    return t.strftime(format)


def plot_stats(csv_filename, columns=['total_reward'], **kwargs):
    """Plot specified columns from CSV file."""
    df_stats = pd.read_csv(csv_filename)
    df_stats[columns].plot(**kwargs)


def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=torch.FloatTensor):
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)


def to_numpy(var):
    return var.data.numpy()


def fan_in_init(size, fan_in=None):
    fan_in = fan_in or size[0]
    v = 1. / np.sqrt(fan_in)
    return torch.Tensor(size).uniform_(-v, v)


class LambdaLayer(torch.nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
    
    
class Actor(torch.nn.Module):
    def __init__(self, state_size, action_size, action_low, action_high, h_units_1=64, h_units_2=64, weights_init=3e-3):
        super(Actor, self).__init__()
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here
        self.fc1 = torch.nn.Linear(state_size, h_units_1)
        self.fc2 = torch.nn.Linear(h_units_1, h_units_2)
        self.fc3 = torch.nn.Linear(h_units_2, action_size)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.scale = LambdaLayer(lambda x: (x * to_tensor(np.array([self.action_range]))) + to_tensor(np.array([self.action_low])))
        self.init_weights(weights_init)

    def init_weights(self, init_w):
        self.fc1.weight.data = fan_in_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fan_in_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        out = self.scale(out)
        return out


class Critic(torch.nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, h_units_1=64, h_units_2=64, weights_init=3e-3):
        super(Critic, self).__init__()
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here
        self.fc1 = torch.nn.Linear(state_size, h_units_1)
        self.fc2 = torch.nn.Linear(h_units_1 + action_size, h_units_2)
        self.fc3 = torch.nn.Linear(h_units_2, 1)
        self.relu = torch.nn.ReLU()
        self.init_weights(weights_init)

    def init_weights(self, init_w):
        self.fc1.weight.data = fan_in_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fan_in_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, xs):
        x, a = xs
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(torch.cat([out, a], 1))
        out = self.relu(out)
        out = self.fc3(out)
        return out