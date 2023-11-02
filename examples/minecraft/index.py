import asyncio
from Bot import Bot
import sys
import os

current_dir = os.path.dirname(__file__)
sys.path.insert(1, os.path.join(current_dir, "../../src"))

from parkour import parkour


BOT_AMOUNT = 20
ACTION_TIME = 0.25 # seconds
MAX_ACTION_COUNT = 30
START_POS = {"x":0.5, "y":5,"z":0.5}
GOAL = {"x":0.5, "y":10,"z":18.5}

async def main():
    clients = []
    for i in range(BOT_AMOUNT):
        client = Bot(f"bot_{i}", "", START_POS)
        clients.append(client.join())
    obj = {"clients":clients,"maxActionCount":MAX_ACTION_COUNT,"goal":GOAL}
    await asyncio.sleep(0.5)
    parkour_generator = parkour(obj)
    await asyncio.sleep(3)

    
    loop = asyncio.get_event_loop()
    async def doAction():
        next(parkour_generator)
        await asyncio.sleep(ACTION_TIME)
     
    while True:
        await loop.create_task(doAction())
        
   
   

    

asyncio.run(main())


