import math

import torch
import random
import numpy as np
from collections import deque

from javascript import console

from model import Linear_QNet, QTrainer, ExperienceReplay
from helper import plot_old
from Bot import Bot, Vec3
from time import sleep
import asyncio
import json

MAX_MEMORY = 50_000
BATCH_SIZE = 32

LR = 0.001  # The learning rate could also be a factor. If it's too high, the bot might be forgetting its past knowledge too quickly to improve. If it's too low, it might not be learning fast enough. Try to adjust the learning rate to see if it has an impact.
ACTION_TIME = 0.3  # seconds
MAX_ACTION_COUNT = BATCH_SIZE
SELECTED_MAP = ["random_parkour"] #["random_parkour"]  # [  "easy_stairs", "1_block_jumps", "2_block_jumps", "1_block_jumps_up",
# "2_block_jumps_up"]  # Select the map you want to complete (see keys of parkour_maps.json)
SAVE_LAST_N_MOVES = 1
ACTION_COUNT = 5  # Amount of actions (jump, forward, backward, short jump)
LOAD_MODEL = False  # whether to load the model from the model.pth file

f = open('./parkour_maps.json')
maps = json.load(f)
map_index = -1
map_fail_counter = 0

START_POS = maps[str(SELECTED_MAP[map_index])]["start"]
g = maps[str(SELECTED_MAP[map_index])]["goal"]
GOAL = Vec3(g["x"], g["y"], g["z"])


def print_bot_view(state):
    bot_symbols = ['O', 'I']
    block_present = '█'
    empty_space = ' '

    # Update symbols based on the presence of blocks
    environment = [block_present if block else empty_space for block in state]

    # Prepare the visual representation
    print(f"""
               {bot_symbols[0]} {environment[0]} {environment[3]} {environment[4]}
               {bot_symbols[1]} {environment[1]} {environment[2]} {environment[5]}  {environment[10]}
               {empty_space} {environment[6]} {environment[7]} {environment[8]} {environment[11]}
        """)


class Agent:

    def __init__(self):
        self.n_games = 0
        # Create list with n elements and fill it with 0
        self.epsilon = [0.8] * len(
            SELECTED_MAP)  # [80] * len(SELECTED_MAP) if LOAD_MODEL else [0] * len(SELECTED_MAP)  # randomness, the lower the more randomness
        self.gamma = 0.80  # discount rate. The value of gamma determines how far into the future the bot should "care" about. If it's too low the bot will mostly consider immediate rewards and thus might not learn beneficial long-term strategies. Try increasing gamma to see if that allows the bot to learn more complex strategies.
        # self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(17, 256, ACTION_COUNT)  #
        self.replay_memory = ExperienceReplay(MAX_MEMORY)  # Init ExperienceReplay
        if LOAD_MODEL:
            self.model.load()  # Load the model.pth file

        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma, replay_memory=self.replay_memory,
                                batch_size=BATCH_SIZE)

    def get_state(self, bot):
        pos = bot.bot.get_actual_position_floored()
        r = bot.get_rotation()
        # is_on_ground = bot.is_on_ground()

        # Environment:
        #
        #
        #       20 16 17 18 19
        #   21  13 o  1  4  5  10
        #   22  14 |  2  3  6  11
        #       15 23 7  8  9  12
        #             24 25 26 27
        #

        block_1 = bot.is_blockAt(pos.x, pos.y + 1, pos.z + r)  # 1
        block_2 = bot.is_blockAt(pos.x, pos.y, pos.z + r)  # 2

        block_3 = bot.is_blockAt(pos.x, pos.y, pos.z + r * 2)  # 3
        block_4 = bot.is_blockAt(pos.x, pos.y + 1, pos.z + 2 * r)  # 4
        block_5 = bot.is_blockAt(pos.x, pos.y + 1, pos.z + 3 * r)  # 5
        block_6 = bot.is_blockAt(pos.x, pos.y, pos.z + 3 * r)  # 6
        block_7 = bot.is_blockAt(pos.x, pos.y - 1, pos.z + r)  # 7
        block_8 = bot.is_blockAt(pos.x, pos.y - 1, pos.z + 2 * r)  # 8
        block_9 = bot.is_blockAt(pos.x, pos.y - 1, pos.z + 3 * r)  # 9
        block_11 = bot.is_blockAt(pos.x, pos.y, pos.z + 4 * r)  # 11
        block_12 = bot.is_blockAt(pos.x, pos.y - 1, pos.z + 4 * r)  # 12
        block_24 = bot.is_blockAt(pos.x, pos.y - 2, pos.z + 1 * r)  # 24
        block_25 = bot.is_blockAt(pos.x, pos.y - 2, pos.z + 2 * r)  # 25
        block_26 = bot.is_blockAt(pos.x, pos.y - 2, pos.z + 3 * r)  # 26
        block_27 = bot.is_blockAt(pos.x, pos.y - 2, pos.z + 4 * r)  # 27

        '''

        block_dlll = bot.is_blockAt(pos.x, pos.y-1, pos.z-3*r) # 9
        block_dllll = bot.is_blockAt(pos.x, pos.y+1, pos.z-4*r) # 10

        block_rrr = bot.is_blockAt(pos.x, pos.y+1, pos.z-r) # 13
        block_rrrr = bot.is_blockAt(pos.x, pos.y, pos.z-r) # 14

        block_l = bot.is_blockAt(pos.x, pos.y-1, pos.z-r) # 15
        block_ll = bot.is_blockAt(pos.x, pos.y+2, pos.z) # 16
        block_lll = bot.is_blockAt(pos.x, pos.y+2, pos.z+r) # 17
        block_llll = bot.is_blockAt(pos.x, pos.y+2, pos.z+r*2) # 18

        block_ur = bot.is_blockAt(pos.x, pos.y+2, pos.z+r*3) # 19
        block_urr = bot.is_blockAt(pos.x, pos.y+2, pos.z-r) # 20
        block_urrr = bot.is_blockAt(pos.x, pos.y+1, pos.z-2*r) # 21
        block_urrrr = bot.is_blockAt(pos.x, pos.y, pos.z-2*r) # 22
        '''

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

            # goal location
            # pos.z < GOAL["z"],  # goal right
            # pos.y < GOAL.get("y"),  # goal up
            pos.y > GOAL["y"],  # goal down
            # is_on_ground
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

        # if len(self.memory) > BATCH_SIZE:
        #    mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        # else:
        #    mini_sample = self.memory

        # states, actions, rewards, next_states, dones = zip(*mini_sample)
        # self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # self.remember(state, action, reward, next_state, done)
        # self.train_long_memory()
        self.trainer.train_step(state, action, reward, next_state, done)
        self.remember(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        if random.randint(0, 100) < self.epsilon[map_index] * 100:
            move = random.randint(0, ACTION_COUNT - 1)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        self.epsilon[map_index] = max(self.epsilon[map_index] * 0.994, 0.03)
        return move

    def set_randomness(self, randomness):
        self.epsilon[map_index] = randomness

    async def next_map(self, bot):
        global map_index
        global START_POS
        global GOAL
        global map_fail_counter
        map_fail_counter = 0
        map_index += 1
        if map_index >= len(SELECTED_MAP):
            map_index = 0
        START_POS = maps[str(SELECTED_MAP[map_index])]["start"]
        if SELECTED_MAP[map_index] == "random_parkour":
            GOAL = bot.bot.createParkourMap(10, START_POS).offset(0.5,1,0.5)
            print(f"New goal: {GOAL}")
            await asyncio.sleep(1)
        elif isinstance(SELECTED_MAP[map_index],int):
            GOAL = bot.bot.createParkourMap(10, START_POS,SELECTED_MAP[map_index]).offset(0.5,1,0.5)
            print(f"New goal: {GOAL}")
            await asyncio.sleep(1)
        else:
            g = maps[SELECTED_MAP[map_index]]["goal"]
            GOAL = Vec3(g["x"], g["y"], g["z"])


async def train():
    last_moves = deque(maxlen=SAVE_LAST_N_MOVES)
    last_position = None
    plot_scores = []
    plot_mean_scores = deque(maxlen=25)
    total_score = 0
    unmoved_positions = 0  # How many times the bot didn't move in x-z direction, if it's higher than 3 we reset the bot (bcs it's most likely stuck)
    record = -100
    action_count = 0

    # Load model
    agent = Agent()
    map_completed = 0  # How many times the bot completed the current map
    bot = Bot(f"bot_RL", "", START_POS, actionCount=ACTION_COUNT)
    reward_array = []
    bot.join()
    sleep(3)
    bot.reset(START_POS)  # goto start position
    await asyncio.sleep(1)
    bot.look(True)
    await agent.next_map(bot)  # Load next map
    while True:

        # get old state
        state_old = agent.get_state(bot)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state

        await bot.await_do_action(final_move, min_time=ACTION_TIME)
        await asyncio.sleep(0.2)
        state_new = agent.get_state(bot)
        action_count += 1
        global map_fail_counter

        if bot.has_reached_goal(GOAL):
            map_completed += 1
            console.log(f"GOAL! Need to complete {6 - map_completed} more times to go to the next map.")
            done = True
            if map_completed > 1:
                await agent.next_map(bot)  # Load next map
                map_completed = 0
            reward = ((MAX_ACTION_COUNT * 2) - action_count)
        elif bot.is_dead():
            map_fail_counter += 1
            done = True
            pos = bot.get_position()
            dist = ((GOAL.z - pos.z) ** 2 + (GOAL.y - pos.y) ** 2)
            reward = -2000 - dist
        elif action_count > MAX_ACTION_COUNT:
            done = True
            map_fail_counter += 1
            pos = bot.get_position()
            dist = ((GOAL.z - pos.z) ** 2 + (GOAL.y - pos.y) ** 2)
            reward = -3000 - dist
        else:
            done = False
            pos = bot.get_position()
            reward = -((GOAL.z - pos.z) ** 2 + (GOAL.y - pos.y) ** 2)
            s = 0
            average = 0
            for i in range(len(last_moves)):
                s += last_moves[i]
            if s != 0:
                average = s / len(last_moves)

            last_moves.append(reward)
            reward -= average
            # We punish the bot if it didn't move
            if last_position is not None and last_position.xzDistanceTo(
                    pos) < 0.31:
                unmoved_positions += 1
                reward -= 500
                if unmoved_positions == 6:
                    bot.reset(START_POS)
                    await asyncio.sleep(0.5)
                    bot.look(True)
                    # clear last moves
                    last_moves.clear()
                    unmoved_positions = 0
            else:
                unmoved_positions = 0
            last_position = pos

            if reward > 0:
                reward = reward * (
                        ACTION_COUNT - final_move) * 8  # a move with a larger index takes more time to complete, so we reward it less

        print(f"Reward={reward}, new_move={final_move} ")
        reward_array.append(reward)

        if map_fail_counter >= 200:
            agent.set_randomness(0.5)
            map_fail_counter = 0
            #await agent.next_map(bot)
        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        # sleep(ACTION_TIME)

        if done:
            # train long memory, plot result
            bot.reset(START_POS)
            await asyncio.sleep(0.5)
            bot.look(True)
            action_count = 0
            agent.n_games += 1
            agent.train_long_memory()

            if agent.n_games % 15 == 0:
                agent.model.save()

            # score = sum(reward_array) / len(reward_array)
            score = -reward
            if score > record:
                record = score
            reward_array = []
            print('Game', agent.n_games, 'Score', score, f'epsilon: {int(agent.epsilon[map_index] * 1000) * 1.0/10}%')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot_old(plot_scores, plot_mean_scores)
            last_moves.clear()
            last_position = None


if __name__ == '__main__':
    asyncio.run(train())
