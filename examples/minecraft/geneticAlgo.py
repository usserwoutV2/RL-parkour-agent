import asyncio
from Bot import Bot
import sys
import os
import json

current_dir = os.path.dirname(__file__)
sys.path.append("../..")
from src.parkour import parkour

BOT_AMOUNT = 15
ACTION_TIME = 0.25  # seconds
MAX_ACTION_COUNT = 30
SELECTED_MAP = "short_parkour"  # Select the map you want to complete (see keys of parkour_maps.json)

f = open('./parkour_maps.json')
maps = json.load(f)

START_POS = maps[SELECTED_MAP]["start"]
GOAL = maps[SELECTED_MAP]["goal"]


async def main():
    clients = []
    for i in range(BOT_AMOUNT):
        client = Bot(f"bot_{i}", "", START_POS)
        clients.append(client.join())
    obj = {"clients": clients, "maxActionCount": MAX_ACTION_COUNT, "goal": GOAL}
    await asyncio.sleep(1.5)
    parkour_generator = parkour(obj)
    await asyncio.sleep(3)

    loop = asyncio.get_event_loop()

    async def doAction():
        next(parkour_generator)
        await asyncio.sleep(ACTION_TIME)

    while True:
        await loop.create_task(doAction())


asyncio.run(main())
