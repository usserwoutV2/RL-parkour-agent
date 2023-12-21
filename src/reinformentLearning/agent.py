import torch
import random
import numpy as np

from src.reinformentLearning.model import Linear_QNet, QTrainer, ExperienceReplay
import asyncio

MAX_MEMORY = 50_000
BATCH_SIZE = 32

LR = 0.001  # The learning rate could also be a factor. If it's too high, the bot might be forgetting its past knowledge too quickly to improve. If it's too low, it might not be learning fast enough. Try to adjust the learning rate to see if it has an impact.
ACTION_TIME = 0.3  # seconds
MAX_ACTION_COUNT = BATCH_SIZE
# "2_block_jumps_up"]  # Select the map you want to complete (see keys of parkour_maps.json)
SAVE_LAST_N_MOVES = 1


def print_bot_view(state):
    bot_symbols = ['O', 'I']
    block_present = '█'
    empty_space = ' '

    # Update symbols based on the presence of blocks
    environment = [block_present if block else empty_space for block in state]

    # Prepare the visual representation
    print(f"""
            |{bot_symbols[0]}|{environment[0]}|{environment[3]}|{environment[4]}|
            |{bot_symbols[1]}|{environment[1]}|{environment[2]}|{environment[5]}|{environment[10]}|
            |{empty_space}|{environment[6]}|{environment[7]}|{environment[8]}|{environment[11]}|
            |{empty_space}|{environment[12]}|{environment[13]}|{environment[14]}|{environment[15]}|
        """)


class Agent:
    def __init__(self, action_count=5, map_count=1):
        self.n_games = 0
        # Create list with n elements and fill it with 0
        self.epsilon = [
                           0.8] * map_count  # [80] * len(SELECTED_MAP) if LOAD_MODEL else [0] * len(SELECTED_MAP)  # randomness, the lower the more randomness
        self.gamma = 0.80  # discount rate. The value of gamma determines how far into the future the bot should "care" about. If it's too low the bot will mostly consider immediate rewards and thus might not learn beneficial long-term strategies. Try increasing gamma to see if that allows the bot to learn more complex strategies.
        # self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(17, 128, action_count)  #
        self.replay_memory = ExperienceReplay(MAX_MEMORY)  # Init ExperienceReplay
        self.action_count = action_count

        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma, replay_memory=self.replay_memory,
                                batch_size=BATCH_SIZE)

    def get_state(self, bot, GOAL):
        pos = bot.bot.get_actual_position_floored()
        r = bot.get_rotation()

        # Environment:
        #
        #
        #       20 16 17 18 19
        #   21  13 o  1  4  5  10
        #   22  14 |  2  3  6  11
        #       15 23 7  8  9  12
        #             24 25 26 27
        #

        block_1 = bot.is_blockAt(pos.x, pos.y + 1, pos.z + 1 * r)  # 1
        block_2 = bot.is_blockAt(pos.x, pos.y, pos.z + 1 * r)  # 2

        block_3 = bot.is_blockAt(pos.x, pos.y, pos.z + 2 * r)  # 3
        block_4 = bot.is_blockAt(pos.x, pos.y + 1, pos.z + 2 * r)  # 4
        block_5 = bot.is_blockAt(pos.x, pos.y + 1, pos.z + 3 * r)  # 5
        block_6 = bot.is_blockAt(pos.x, pos.y, pos.z + 3 * r)  # 6
        block_7 = bot.is_blockAt(pos.x, pos.y - 1, pos.z + 1 * r)  # 7
        block_8 = bot.is_blockAt(pos.x, pos.y - 1, pos.z + 2 * r)  # 8
        block_9 = bot.is_blockAt(pos.x, pos.y - 1, pos.z + 3 * r)  # 9
        block_11 = bot.is_blockAt(pos.x, pos.y, pos.z + 4 * r)  # 11
        block_12 = bot.is_blockAt(pos.x, pos.y - 1, pos.z + 4 * r)  # 12
        block_24 = bot.is_blockAt(pos.x, pos.y - 2, pos.z + 1 * r)  # 24
        block_25 = bot.is_blockAt(pos.x, pos.y - 2, pos.z + 2 * r)  # 25
        block_26 = bot.is_blockAt(pos.x, pos.y - 2, pos.z + 3 * r)  # 26
        block_27 = bot.is_blockAt(pos.x, pos.y - 2, pos.z + 4 * r)  # 27

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
            block_11,
            block_12,
            block_24,
            block_25,
            block_26,
            block_27,
            # Whether or not the bot is looking towards to goal or not
            1 if (pos.z >= GOAL["z"] and r == 1) or (pos.z <= GOAL["z"] and r == -1) else 0,
            pos.y > GOAL["y"],  # goal down
        ]
        # print_bot_view(state)
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached
        self.replay_memory.push(state, action, reward, next_state, done)

    def train_long_memory(self):
        if self.replay_memory.can_provide_sample(BATCH_SIZE):  # If can provide sample
            states, actions, rewards, next_states, dones = self.replay_memory.sample(BATCH_SIZE)
            self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # self.remember(state, action, reward, next_state, done)
        # self.train_long_memory()
        self.trainer.train_step(state, action, reward, next_state, done)
        self.remember(state, action, reward, next_state, done)

    def get_action(self, state, map_index):
        # random moves: tradeoff exploration / exploitation
        if random.randint(0, 100) < self.epsilon[map_index] * 100:
            move = random.randint(0, self.action_count - 1)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        self.epsilon[map_index] = max(self.epsilon[map_index] * 0.994, 0.03)
        return move

    def set_randomness(self, randomness, map_index):
        self.epsilon[map_index] = randomness
