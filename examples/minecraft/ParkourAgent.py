import math

from Bot import Bot, Vec3
from time import sleep
import asyncio
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from reinforcement_learning.ReplayBuffer import ReplayBuffer
from collections import deque

SELECTED_MAP = ["easy_stairs", "1_block_jumps", "2_block_jumps", "1_block_jumps_up",
                "2_block_jumps_up"]  # Select the map you want to complete (see keys of parkour_maps.json)

GAMMA = 0.99  # discount factor
UPDATE_EVERY = 4

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size

f = open('./parkour_maps.json')
maps = json.load(f)
map_index = 0
map_fail_counter = 0

START_POS = maps[SELECTED_MAP[map_index]]["start"]
g = maps[SELECTED_MAP[map_index]]["goal"]
GOAL = Vec3(g["x"], g["y"], g["z"])
ACTION_TIME = 0.3  # seconds
ACTION_COUNT = 5
MAX_TIME_STEPS = 25
N_EPISODES = 1000

LR = 0.0005  # learning rate
TAU = 0.001  # for soft update of target parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # use GPU if available, else use CPU


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Agent():
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, device)

        self.last_moves = deque(maxlen=2)
        self.last_position = None
        self.unmoved_positions = 0
        self.t_step = 0
        self.last_dx = None
        self.last_dz = None

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def get_state(self, bot):
        pos = bot.get_position_floored()
        r = bot.get_rotation()
        # is_on_ground = bot.is_on_ground()

        # Environment:
        #
        #
        #       20 16 17 18 19
        #   21  13 o  1  4  5  10
        #   22  14 |  2  3  6  11
        #       15 23 7  8  9  12
        #

        block_1 = bot.is_blockAt(pos.x, pos.y + 1, pos.z + r)  # 1
        block_2 = bot.is_blockAt(pos.x, pos.y, pos.z + r)  # 2

        block_3 = bot.is_blockAt(pos.x, pos.y, pos.z + r * 2)  # 3
        block_4 = bot.is_blockAt(pos.x, pos.y + 1, pos.z + 2 * r)  # 4
        block_5 = bot.is_blockAt(pos.x, pos.y + 1, pos.z + 3 * r)  # 5
        block_6 = bot.is_blockAt(pos.x, pos.y, pos.z + 3 * r)  # 6
        block_7 = bot.is_blockAt(pos.x, pos.y - 1, pos.z + r)  # 7
        block_8 = bot.is_blockAt(pos.x, pos.y - 1, pos.z - 2 * r)  # 8
        block_9 = bot.is_blockAt(pos.x, pos.y - 1, pos.z - 3 * r)  # 9
        state = [
            block_1,
            block_2,
            block_3,
            block_4,
            block_5,
            block_6,
            block_7,
            block_8,
            block_9,
            # Whether or not the bot is looking towards to goal or not
            1 if (pos.z >= GOAL["z"] and r == 1) or (pos.z <= GOAL["z"] and r == -1) else 0,

            # goal location
            # pos.z < GOAL["z"],  # goal right
            # pos.y < GOAL.get("y"),  # goal up
            pos.y > GOAL["y"],  # goal down
            # is_on_ground
        ]
        # print_bot_view(state)

        return np.array(state, dtype=int)

    async def calculate_reward(self, bot):
        pos = bot.get_position()
        reward = ((GOAL.z - pos.z) ** 2 + (GOAL.y - pos.y) ** 2)
        average = sum(self.last_moves) / len(self.last_moves) if len(self.last_moves) > 0 else 0

        self.last_moves.append(reward)
        reward -= average

        if self.last_position is not None and self.last_position.xzDistanceTo(pos) < 0.1:
            self.unmoved_positions += 1
            reward -= 20000
            if self.unmoved_positions == 6:
                bot.reset(START_POS)
                await asyncio.sleep(0.5)
                bot.look(True)
                # clear last moves
                self.last_moves.clear()
                self.unmoved_positions = 0
        else:
            self.unmoved_positions = 0
        self.last_position = pos

        done = bot.has_reached_goal(GOAL)

        if done:
            reward += 100000

        return done, reward


# Training procedure
async def train():
    bot = Bot(f"bot_RL", "", START_POS, actionCount=ACTION_COUNT)
    bot.join()
    await asyncio.sleep(3)
    bot.reset(START_POS)  # Go to start position
    await asyncio.sleep(1)
    bot.look(True)

    agent = Agent(state_size=11, action_size=5, seed=0)

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores

    for i_episode in range(N_EPISODES):
        state = agent.get_state(bot)  # get the current state
        score = 0
        for time_step in range(MAX_TIME_STEPS):
            action = agent.act(state)  # select an action
            await bot.await_do_action(action)
            done, reward = await agent.calculate_reward(bot)  # you need to implement this function
            print(f"reward: {reward}")
            next_state = agent.get_state(bot)  # get the resulting state
            agent.step(state, action, reward, next_state, done)  # agent learns
            state = next_state  # update the current state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        bot.reset(START_POS)
        await asyncio.sleep(0.5)
        bot.look(True)

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
    return scores


# Initialize bot and start training
if __name__ == '__main__':
    asyncio.run(train())
