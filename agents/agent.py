import os
from util import *


class DDPG:
    """Reinforcement Learning agent that learns using DDPG."""

    def __init__(self, task):
        self.task = task        

        # Noise process
        self.mu = 0.99
        self.theta = 0.15
        self.sigma = 0.3

        # Replay memory
        self.buffer_size = 10000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.1  # discount factor
        self.tau = 0.0005  # for soft update of target parameters

        # Episode variables
        self.episode = 0
        self.episode_duration = 0
        self.total_reward = None
        self.best_total_reward = -np.inf
        self.score = None
        self.best_score = -np.inf
        self.last_states = None
        self.last_action = None
        
        # constrain to z only
        self.state_start = 2
        self.state_end = 3
        self.state_size = (self.state_end - self.state_start)*self.task.action_repeat
        
        # apply same rotor force to all rotor, see post process
        self.action_size = 1
        self.action_low = self.task.action_low
        self.action_high = self.task.action_high
        self.noise = OUNoise(self.action_size, self.mu, self.theta, self.sigma)

        # Actor (Policy) Model
        self.actor_learning_rate = 0.0001
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_opt = torch.optim.Adam(self.actor_local.parameters(), lr=self.actor_learning_rate)

        # Critic (Value) Model
        self.critic_learning_rate = 0.001
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)
        self.critic_opt = torch.optim.Adam(self.critic_local.parameters(), lr=self.critic_learning_rate)

    def preprocess_state(self, states):
        """Reduce state vector to relevant dimensions."""
        repeated_states = np.reshape(states, [self.task.action_repeat,-1])
        return repeated_states[:, self.state_start:self.state_end]  # z positions only

    def postprocess_action(self, action):
        """Return complete action vector."""
        complete_action = action * np.ones((self.task.action_size, 1))  # shape: (4,)
        return complete_action

    def reset_episode(self):
        self.score = self.total_reward / float(self.episode_duration) if self.episode_duration else -np.inf
        if self.best_score < self.score:
            self.best_score = self.score
        if self.total_reward and self.total_reward > self.best_total_reward:
            self.best_total_reward = self.total_reward
            
        self.total_reward = None
        self.episode_duration = 0
        self.last_states = None
        self.last_action = None
        state = self.task.reset()
        self.episode += 1
        return state

    def step(self, states, reward, done):
        states = self.preprocess_state(states)
        if self.total_reward:
            self.total_reward += reward
        else:
            self.total_reward = reward

        self.episode_duration += 1
        # Save experience / reward
        if self.last_states is not None and self.last_action is not None:
            self.memory.add(self.last_states, self.last_action, reward, states, done)

        self.last_states = states
        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

    def act(self, states):
        """Returns actions for given state(s) as per current policy."""
        states = self.preprocess_state(states)
        states = np.reshape(states, [-1, self.state_size])
        actions = self.predict_actions(states)
        actions = actions + self.noise.sample()  # add some noise for exploration
        self.last_action = actions
        actions = self.postprocess_action(actions)
        return actions

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        for target_param, param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )   

    def predict_actions(self, states):
        return to_numpy(
            self.actor_local(to_tensor(np.array([states])))
        ).squeeze(0)

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None]).reshape(-1, self.task.action_repeat)
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None]).reshape(-1, self.task.action_repeat)

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target(to_tensor(next_states, volatile=True))
        Q_targets_next = self.critic_target([
            to_tensor(next_states, volatile=True),
            actions_next,
        ])
        Q_targets_next.volatile = False

        # Compute Q targets for current states and train critic model (local)
        Q_targets = to_tensor(rewards) + to_tensor(np.array([self.gamma])) * Q_targets_next * (1 - to_tensor(dones))

        self.critic_local.zero_grad()
        Q_train = self.critic_local([to_tensor(states), to_tensor(actions)])
        v_loss = torch.nn.MSELoss()(Q_train,Q_targets)
        v_loss.backward()
        self.critic_opt.step()

        # Train actor model (local)
        self.actor_local.zero_grad()
        p_loss = -self.critic_local([
            to_tensor(states),
            self.actor_local(to_tensor(states))
        ])
        p_loss = p_loss.mean()
        p_loss.backward()
        self.actor_opt.step()

        # Soft-update target models
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)


