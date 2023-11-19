import torch
import random
import numpy as np
from collections import deque

from javascript import console

from model import Linear_QNet, QTrainer
from helper import plot
from Bot import Bot, Vec3
from time import sleep
import json

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
ACTION_TIME = 0.1  # seconds
MAX_ACTION_COUNT = 30 * 2
SELECTED_MAP = ["easy_stairs","1_block_jumps", "2_block_jumps","1_block_jumps_up","2_block_jumps_up"]  # Select the map you want to complete (see keys of parkour_maps.json)
SAVE_LAST_N_MOVES = 2

LOAD_MODEL = True  # whether to load the model from the model.pth file

f = open('./parkour_maps.json')
maps = json.load(f)
map_index = 0
map_fail_counter = 0

START_POS = maps[SELECTED_MAP[map_index]]["start"]
g = maps[SELECTED_MAP[map_index]]["goal"]
GOAL = Vec3(g["x"], g["y"], g["z"])

def next_map():
    global map_index
    global START_POS
    global GOAL
    global map_fail_counter
    map_fail_counter = 0
    map_index += 1
    if map_index >= len(SELECTED_MAP):
        map_index = 0
    START_POS = maps[SELECTED_MAP[map_index]]["start"]
    g = maps[SELECTED_MAP[map_index]]["goal"]
    GOAL = Vec3(g["x"], g["y"], g["z"])


class Agent:

    def __init__(self):
        self.n_games = 0
        # Create list with n elements and fill it with 0
        self.epsilon = [80] * len(SELECTED_MAP) if LOAD_MODEL else [0]* len(SELECTED_MAP)  # randomness, the lower the more randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(12, 1 << 12, 4)  # model_1:  7 params
        if LOAD_MODEL:
            self.model.load()  # Load the model.pth file

        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, bot):
        pos = bot.get_position_floored()
        r = bot.get_rotation()
        is_on_ground = bot.is_on_ground()

        # Environment:
        #
        #
        #       20 16 17 18 19
        #   21  13 o  1  4  5  10
        #   22  14 |  2  3  6  11
        #       15 23 7  8  9  12
        #

        block_d = bot.is_blockAt(pos.x, pos.y + 1, pos.z + r)  # 1
        block_uu = bot.is_blockAt(pos.x, pos.y, pos.z + r)  # 2

        block_dr = bot.is_blockAt(pos.x, pos.y, pos.z + r * 2)  # 3
        block_drr = bot.is_blockAt(pos.x, pos.y + 1, pos.z + 2 * r)  # 4
        block_dl = bot.is_blockAt(pos.x, pos.y - 1, pos.z + r)  # 7
        block_drrr = bot.is_blockAt(pos.x, pos.y + 1, pos.z + 3 * r)  # 5
        block_drrrr = bot.is_blockAt(pos.x, pos.y, pos.z + 3 * r)  # 6
        block_dll = bot.is_blockAt(pos.x, pos.y - 1, pos.z - 2 * r)  # 8
        block_u = bot.is_blockAt(pos.x, pos.y - 1, pos.z)  # 23

        '''

        block_dlll = bot.is_blockAt(pos.x, pos.y-1, pos.z-3*r) # 9
        block_dllll = bot.is_blockAt(pos.x, pos.y+1, pos.z-4*r) # 10

        block_r = bot.is_blockAt(pos.x, pos.y, pos.z+4*r) # 11
        block_rr = bot.is_blockAt(pos.x, pos.y-1, pos.z+2*r) # 12
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

        #block_ul = bot.is_blockAt(pos.x, pos.y+1, pos.z-1)
        #block_ull = bot.is_blockAt(pos.x, pos.y+1, pos.z-2)
        #block_ulll = bot.is_blockAt(pos.x, pos.y+1, pos.z-3)
        #block_ullll = bot.is_blockAt(pos.x, pos.y+1, pos.z-4)

        #block_uur = bot.is_blockAt(pos.x, pos.y+2, pos.z+1)
        #block_uurr = bot.is_blockAt(pos.x, pos.y+2, pos.z+2)
        #block_uurrr = bot.is_blockAt(pos.x, pos.y+2, pos.z+3)

        #block_uul = bot.is_blockAt(pos.x, pos.y+2, pos.z-1)
        #block_uull = bot.is_blockAt(pos.x, pos.y+2, pos.z-2)
        #block_uulll = bot.is_blockAt(pos.x, pos.y+2, pos.z-3)
        '''

        state = [
            block_d,
            block_uu,
            block_dr,
            block_drr,
            block_dl,
            block_drrr,
            block_drrrr,
            block_dll,
            block_u,
            # rotation:
            r == 1,

            # goal location
            # pos.z < GOAL["z"],  # goal right
            # pos.y < GOAL.get("y"),  # goal up
            pos.y > GOAL["y"],  # goal down
            is_on_ground
        ]

        # print(state,r)
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon[map_index] = 80 - self.n_games
        # final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon[map_index]:
            move = random.randint(0, 2)
            # final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            # final_move[move] = 1

        # return final_move
        return move

    def set_randomness(self, randomness):
        self.epsilon[map_index] = randomness


def train():
    last_moves = deque(maxlen=SAVE_LAST_N_MOVES)
    last_position = None
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    unmoved_positions = 0  # How many times the bot didn't move in x-z direction, if it's higher than 3 we reset the bot (bcs it's most likely stuck)
    record = -100
    action_count = 0
    # Load model
    agent = Agent()

    bot = Bot(f"bot_RL", "", START_POS)
    bot.join()
    sleep(3)
    bot.reset(START_POS)
    sleep(1)
    while True:

        # get old state
        state_old = agent.get_state(bot)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        bot.do_action(final_move)

        sleep(ACTION_TIME)
        state_new = agent.get_state(bot)
        score = 0
        action_count += 1

        if bot.has_reached_goal(GOAL):
            console.log("GOAL!")
            done = True
            score = action_count
            next_map()  # Load next map
            reward = ((MAX_ACTION_COUNT * 2) - action_count) ** 5
        elif action_count > MAX_ACTION_COUNT:
            done = True
            global map_fail_counter
            map_fail_counter += 1
            if map_fail_counter >= 50: # To prevent the bot from getting stuck on a map, we give him 50 tries to complete it, if it fails we go to the next map
                agent.set_randomness(0)
                next_map()
            pos = bot.get_position()
            score = -((GOAL.z - pos.z) ** 2 + (GOAL.y - pos.y) ** 2)
            reward = score
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
            if last_position is not None and last_position.xzDistanceTo(pos) < 0.1:
                unmoved_positions += 1
                reward -= 100
                if unmoved_positions == 6:
                    bot.reset(START_POS)
                    unmoved_positions = 0
            else:
                unmoved_positions = 0
            last_position = pos

            print(f"Reward={reward}, new_move={final_move} ")


        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        # sleep(ACTION_TIME)

        if done:
            # train long memory, plot result
            bot.reset(START_POS)
            action_count = 0
            agent.n_games += 1
            agent.train_long_memory()

            if agent.n_games % 15 == 0:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(-score)
            total_score += -score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            last_moves.clear()
            last_position = None


if __name__ == '__main__':
    train()
