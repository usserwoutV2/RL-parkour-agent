import asyncio
import sys
import os
from Bot import Bot

current_dir = os.path.dirname(__file__)
sys.path.append("../..")
from src.RLParkour import RLParkour



ACTION_COUNT = 5  # Amount of actions


async def main():
    bot = Bot(f"bot_RL", "", actionCount=ACTION_COUNT)
    await RLParkour(bot,model="../../examples/minecraft/model/model.pth")



if __name__ == "__main__":
    asyncio.run(main())



