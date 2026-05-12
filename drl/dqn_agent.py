import random
import torch
import torch.nn as nn
import torch.optim as optim

from drl.replay_buffer import ReplayBuffer


class DQNNetwork(nn.Module):

    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.model(x)


class DQNAgent:

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        batch_size=64
    ):

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size

        self.memory = ReplayBuffer()

        self.policy_net = DQNNetwork(
            state_size,
            action_size
        )

        self.target_net = DQNNetwork(
            state_size,
            action_size
        )

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate
        )

        self.loss_fn = nn.MSELoss()

    def act(self, state):

        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):

        self.memory.push(
            state,
            action,
            reward,
            next_state,
            done
        )

    def train(self):

        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        current_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        def decay_epsilon(self):
              if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

              if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

        return loss.item()

    def update_target_network(self):

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )
    def save(self, path):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]        