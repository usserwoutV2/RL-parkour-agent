import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np


class ExperienceReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(
            *random.sample(self.buffer, batch_size))
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def can_provide_sample(self, batch_size):
        return len(self.buffer) >= batch_size


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.model_folder_path = './model'

    def load(self, path: str = 'model.pth'):
        file_name = os.path.join(self.model_folder_path, path)
        self.load_state_dict(torch.load(file_name))

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        if not os.path.exists(self.model_folder_path):
            os.makedirs(self.model_folder_path)

        file_name = os.path.join(self.model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma, replay_memory, batch_size=32):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.replay_memory = replay_memory
        self.batch_size = batch_size

    def train_step(self, states, actions, rewards, next_states, dones):
        if not self.replay_memory.can_provide_sample(self.batch_size):
            return

        states = torch.tensor(np.array(states), dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(-1) # CHANGED THIS LINE
        rewards = torch.tensor(np.array(rewards), dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.bool)

        if len(states.shape) == 1:
            states = states.unsqueeze(0)
            next_states = next_states.unsqueeze(0)
            actions = actions.unsqueeze(0)
            rewards = rewards.unsqueeze(0)
            dones = dones.unsqueeze(0)

        current_q_values = self.model(states).gather(1, actions)

        next_q_values = self.model(next_states).detach()
        next_v = rewards + self.gamma * next_q_values.max(1)[0].unsqueeze(1) * (~dones).float().unsqueeze(1)

        loss = self.criterion(current_q_values, next_v)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


'''
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            #target[idx][torch.argmax(action[idx]).item()] = Q_new
            target[idx][action[idx].item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
'''
