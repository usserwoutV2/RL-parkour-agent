
import torch
import random
import numpy as np
from collections import deque

from javascript import console

from model import Linear_QNet, QTrainer
from helper import plot
from Bot import Bot, Vec3
from time import sleep

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
ACTION_TIME = 0.25  # seconds
MAX_ACTION_COUNT = 30
START_POS = {"x": 0.5, "y": 5, "z": 0.5}
GOAL = Vec3(0.5, 5, -9.5)


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, bot):
        pos = bot.get_position_floored()

        # Environment:
        #
        #
        #       *  *  *  *  *
        #   *   *  o  1  4  5  *
        #   *   *  |  2  3  6  *
        #       *     7  8  9  *
        #

        block_d = bot.is_blockAt(pos.x, pos.y+1, pos.z+1) # 1
        block_uu = bot.is_blockAt(pos.x, pos.y+2, pos.z)

        block_dr = bot.is_blockAt(pos.x, pos.y-1, pos.z+1)
        block_drr = bot.is_blockAt(pos.x, pos.y-1, pos.z+2)
        block_drrr = bot.is_blockAt(pos.x, pos.y-1, pos.z+3)
        block_drrrr = bot.is_blockAt(pos.x, pos.y-1, pos.z+4)

        block_dl = bot.is_blockAt(pos.x, pos.y-1, pos.z-1)
        block_dll = bot.is_blockAt(pos.x, pos.y-1, pos.z-2)
        block_dlll = bot.is_blockAt(pos.x, pos.y-1, pos.z-3)
        block_dllll = bot.is_blockAt(pos.x, pos.y-1, pos.z-4)

        block_r = bot.is_blockAt(pos.x, pos.y, pos.z+1)
        block_rr = bot.is_blockAt(pos.x, pos.y, pos.z+2)
        block_rrr = bot.is_blockAt(pos.x, pos.y, pos.z+3)
        block_rrrr = bot.is_blockAt(pos.x, pos.y, pos.z+4)

        block_l = bot.is_blockAt(pos.x, pos.y, pos.z-1)
        block_ll = bot.is_blockAt(pos.x, pos.y, pos.z-2)
        block_lll = bot.is_blockAt(pos.x, pos.y, pos.z-3)
        block_llll = bot.is_blockAt(pos.x, pos.y, pos.z-4)

        block_ur = bot.is_blockAt(pos.x, pos.y+1, pos.z+1)
        block_urr = bot.is_blockAt(pos.x, pos.y+1, pos.z+2)
        block_urrr = bot.is_blockAt(pos.x, pos.y+1, pos.z+3)
        block_urrrr = bot.is_blockAt(pos.x, pos.y+1, pos.z+4)

        block_ul = bot.is_blockAt(pos.x, pos.y+1, pos.z-1)
        block_ull = bot.is_blockAt(pos.x, pos.y+1, pos.z-2)
        block_ulll = bot.is_blockAt(pos.x, pos.y+1, pos.z-3)
        block_ullll = bot.is_blockAt(pos.x, pos.y+1, pos.z-4)

        block_uur = bot.is_blockAt(pos.x, pos.y+2, pos.z+1)
        block_uurr = bot.is_blockAt(pos.x, pos.y+2, pos.z+2)
        block_uurrr = bot.is_blockAt(pos.x, pos.y+2, pos.z+3)

        block_uul = bot.is_blockAt(pos.x, pos.y+2, pos.z-1)
        block_uull = bot.is_blockAt(pos.x, pos.y+2, pos.z-2)
        block_uulll = bot.is_blockAt(pos.x, pos.y+2, pos.z-3)

        state = [
            block_d,
            # block_uu,

            block_dr,
            block_drr,
            block_drrr,
            block_drrrr,

            block_dl,
            block_dll,
            block_dlll,
            block_dllll,

            # block_r,
            # block_rr,
            # block_rrr,
            # block_rrrr,
            #
            # block_l,
            # block_ll,
            # block_lll,
            # block_llll,
            #
            # block_ur,
            # block_urr,
            # block_urrr,
            # block_urrrr,
            #
            # block_ul,
            # block_ull,
            # block_ulll,
            # block_ullll,
            #
            # block_uur,
            # block_uurr,
            # block_uurrr,
            #
            # block_uul,
            # block_uull,
            # block_uulll,
            
            # goal location
            pos.z < GOAL.z,  # goal right
            # pos.y < GOAL.get("y"),  # goal up
            pos.y > GOAL.y  # goal down
            ]
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
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        # final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            # final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            # final_move[move] = 1

        # return final_move
        return move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = -100
    action_count = 0
    agent = Agent()
    bot = Bot(f"bot_RL", "", START_POS)
    bot.join()
    sleep(3)
    bot.reset()
    sleep(1)
    while True:
        # get old state
        state_old = agent.get_state(bot)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        bot.do_action(final_move)
        print("Move: ",final_move)

        sleep(ACTION_TIME)
        state_new = agent.get_state(bot)

        action_count += 1
        # console.log(state_new)
        # console.log(action_count)
        if bot.has_reached_goal(GOAL):
            console.log("GOAL!")
            done = True
            score = action_count
            reward = (MAX_ACTION_COUNT - action_count)**2 + 20;
        elif action_count > MAX_ACTION_COUNT:
            console.log("ran out of moves")
            done = True
            pos = bot.get_position()
            score = -(GOAL.z-pos.z)**2 + (GOAL.y-pos.y)**2
            reward = score
        else:
            done = False
            pos = bot.get_position()
            reward = -(GOAL.z-pos.z)**2 + (GOAL.y-pos.y)**2
            #reward = 0


        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        # sleep(ACTION_TIME)

        if done:
            # train long memory, plot result
            bot.reset()
            action_count = 0
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(-score)
            total_score += -score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()