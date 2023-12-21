from collections import deque

from javascript import console

from src.helper import plot_old
from time import sleep
import asyncio
from src.reinformentLearning.agent import Agent

BATCH_SIZE = 32
ACTION_TIME = 0.3  # seconds
MAX_ACTION_COUNT = BATCH_SIZE
SAVE_LAST_N_MOVES = 1
map_fail_counter = 0


async def RLParkour(bot, model=None):
    last_moves = deque(maxlen=SAVE_LAST_N_MOVES)
    last_position = None
    plot_scores = deque(maxlen=50)
    plot_mean_scores = []
    all_scores = []
    total_score = 0
    unmoved_positions = 0  # How many times the bot didn't move in x-z direction, if it's higher than 3 we reset the bot (bcs it's most likely stuck)
    record = -100
    action_count = 0

    # Load model
    agent = Agent(action_count=bot.actionCount, map_count=bot.getMapCount())
    if model is not None:
        agent.model.load(model)

    map_completed = 0  # How many times the bot completed the current map
    reward_array = []
    bot.join()
    sleep(3)
    bot.reset()  # goto start position
    await asyncio.sleep(1)
    bot.look(True)
    await bot.next_map()  # Load next map
    while True:

        # get old state
        state_old = agent.get_state(bot, bot.getGoal())

        # get move
        final_move = agent.get_action(state_old, bot.getMapIndex())

        # perform move and get new state
        await bot.await_do_action(final_move, min_time=ACTION_TIME)
        await asyncio.sleep(0.2)
        state_new = agent.get_state(bot, bot.getGoal())
        action_count += 1
        GOAL = bot.getGoal()
        global map_fail_counter

        if bot.has_reached_goal(GOAL):
            map_completed += 1
            console.log(f"GOAL! Need to complete {6 - map_completed} more times to go to the next map.")
            done = True
            if map_completed > 1:
                await bot.next_map()  # Load next map
                map_completed = 0
            reward = ((MAX_ACTION_COUNT * 2) - action_count) + 2000
        elif bot.is_dead():
            map_fail_counter += 1
            done = True
            pos = bot.get_position()
            dist = ((GOAL.z - pos.z) ** 2 + (GOAL.y - pos.y) ** 2)
            reward = -3000 - dist
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
                    bot.reset(bot.getStartPos())
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
                        bot.actionCount * 2 - bot.action_weight(final_move))  # some moves take more time than others

        print(f"Reward={reward}, new_move={final_move} ")
        reward_array.append(reward)

        if map_fail_counter >= 200:
            agent.set_randomness(0.5, bot.getMapIndex())
            map_fail_counter = 0

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            bot.reset(bot.getStartPos())
            await asyncio.sleep(0.5)
            bot.look(True)
            action_count = 0
            agent.n_games += 1
            agent.train_long_memory()

            if agent.n_games % 15 == 0:
                agent.model.save()

            score = -reward
            if score > record:
                record = score
            reward_array = []
            print('Game', agent.n_games, 'Score', score)

            # remove last element of deque plot_scores and decrement total_score by that amount
            if len(plot_scores) >= 50:
                total_score -= plot_scores.popleft()
            all_scores.append(score)
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / len(plot_scores)
            plot_mean_scores.append(mean_score)
            plot_old(all_scores, plot_mean_scores)
            last_moves.clear()
            last_position = None
