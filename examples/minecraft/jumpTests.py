from Bot import Bot, Vec3
from time import sleep
import asyncio

JUMP_COUNT = 11
ACTION_COUNT = 4
START_POS = Vec3(-8.5, 7, -20.5)


def init():
    bot = Bot(f"bot_RL", "", actionCount=ACTION_COUNT)
    bot.join()
    bot.setStartPos(START_POS)
    sleep(3)
    bot.reset(START_POS)
    sleep(1)
    return bot


def goal_reached(bot):
    p = bot.get_position()
    pos2 = bot.get_position_floored()
    if bot.blockAt(pos2.x, pos2.y - 1, pos2.z).name == "diamond_block" and abs(p.y-pos2.y) < 0.01:
        return True
    pos = bot.get_position().offset(0, 0, -0.2999).floored()
    if bot.blockAt(pos.x, pos.y - 1, pos.z).name == "diamond_block" and abs(p.y-pos.y) < 0.01:
        return True
    return False


async def main():
    bot = init()

    while True:
        for jmp in range(JUMP_COUNT):
            start_pos = START_POS.offset(jmp * 2, 0, 0)
            bot.reset(start_pos)
            await asyncio.sleep(1)
            bot.look(True)
            for i in range(2, 5):
                await bot.await_do_action(i)
                await asyncio.sleep(0.3)
                if goal_reached(bot):
                    print(f"reached ({i})")
                    break
                else:
                    print(f"not reached ({i})")
                    bot.reset(start_pos)
                    await asyncio.sleep(1)
                    bot.look(True)


if __name__ == "__main__":
    asyncio.run(main())