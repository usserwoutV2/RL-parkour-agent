from javascript import require, console, On, once
import math
import numpy as np
import asyncio
import json

createBot = require("./js/createBot.js")
Vec3 = require("vec3").Vec3

null_vector = Vec3(0, 0, 0)
f = open('../../examples/minecraft/parkour_maps.json')
maps = json.load(f)
map_index = -1
map_fail_counter = 0
SELECTED_MAP = ["random_parkour"] #["random_parkour"]  # [  "easy_stairs", "1_block_jumps", "2_block_jumps", "1_block_jumps_up",
START_POS = maps[str(SELECTED_MAP[map_index])]["start"]
g = maps[str(SELECTED_MAP[map_index])]["goal"]
GOAL = Vec3(g["x"], g["y"], g["z"])

class Bot:

    def __init__(self, username: str, host: str, actionCount=5):
        self.host = host
        self.username = username
        self.actionCount = actionCount  # forward, backward, jump, (long, middle, short jump)
        self.active = True  # if false, the bot will not do anything (eg when bot died)

    def join(self):
        self.bot = createBot({
            "host": self.host,
            "port": 25565,
            "version": "1.8.9",
            "username": self.username,
        })

        @On(self.bot, "death")
        def on_death():
            self.active = False
            console.log("died")

        return self

    def do_action(self, action: int):
        if not self.active:
            print("bot died, cannot do action")
            return
        if action == 0:
            self.forward()
        elif action == 1:
            self.backward()
        elif action == 2:
            self.jump()
        elif action == 3:
            self.idle()
        else:
            console.log("invalid action")

    '''
    Do action but wait until action is finished
    '''

    async def await_do_action(self, action: int, min_time: float = 0.3):
        self.bot.clearControlStates()
        if not self.active:
            print("bot died, cannot do action")
            return
        if action == 0:  # Forward
            self.bot.move_to_middle()
            # await asyncio.sleep(0.3)
            self.wait_for_ticks(6)
            self.bot.setControlState("forward", True)
            self.bot.setControlState("sprint", True)
            self.wait_for_ticks(5)
            # await asyncio.sleep(0.25)
            self.bot.clearControlStates()
            while self.bot.entity.velocity.z != 0 or not self.bot.entity.onGround:
                self.wait_for_ticks(2)
                # await asyncio.sleep(0.1)  # Wait until the bot is on the ground
        elif action == 1:
            self.bot.move_to_middle()
            self.wait_for_ticks(6)
            self.bot.clearControlStates()
        elif action == 2 or action == 3 or action == 4:
            self.bot.move_to_middle()
            self.wait_for_ticks(6)
            # await asyncio.sleep(0.3)
            self.bot.setControlState("forward", True)
            if action != 2:
                self.bot.setControlState("sprint", True)

            self.wait_for_ticks(1 if action != 4 else 4)

            # await self.bot.waitForTicks(2 if action == 2 else 4)
            # await asyncio.sleep(0.10 if action == 3 else 0.20)
            self.bot.setControlState("jump", True)
            self.bot.setControlState("jump", False)
            self.wait_for_ticks(6)
            #             await asyncio.sleep(min_time * ((action-2) * 0.5 + 1))
            self.bot.clearControlStates()
            while not self.bot.entity.onGround or self.bot.entity.velocity.z != 0:
                await asyncio.sleep(0.1)  # Wait until the bot is on the ground
            await asyncio.sleep(0.3)
            while not self.bot.entity.onGround or self.bot.entity.velocity.z != 0:
                await asyncio.sleep(0.1)  # Wait until the bot is on the ground
        else:
            console.log("invalid action")

    def look(self, forward):
        if forward:
            self.bot.look(0, 0, True)
        else:
            self.bot.look(math.pi, 0, True)

    # Rotate 180 degrees
    def rotate(self):
        if self.get_rotation() == -1:
            self.bot.look(math.pi, 0, True)
        else:
            self.bot.look(0, 0, True)

    def backward(self):
        self.bot.clearControlStates()
        # if self.get_rotation() == -1:
        self.bot.look(math.pi, 0, True)
        # else:
        #    self.bot.look(0, 0, True)
        self.bot.setControlState("forward", True)
        self.bot.setControlState("sprint", True)

    def forward(self):
        self.bot.clearControlStates()
        self.bot.look(0, 0, True)
        self.bot.setControlState("forward", True)
        self.bot.setControlState("sprint", True)

    def jump(self, sprint=True):
        self.bot.clearControlStates()
        self.bot.setControlState("jump", True)
        self.bot.setControlState("forward", True)
        if sprint: self.bot.setControlState("sprint", True)

    def idle(self):
        self.bot.clearControlStates()

    # Bot is dead when it stand on a redstone block
    def is_dead(self):
        return self.bot.isBlockBelow("redstone_block")

    def reset(self, pos=None):
        if pos is None:
            global START_POS
            pos = START_POS
        self.bot.clearControlStates()
        self.active = True
        self.bot.chat(f'/tp {self.username} {pos["x"]} {pos["y"]} {pos["z"]}')
        self.look(True)

    def get_position(self):
        return self.bot.player.entity.position

    def get_position_floored(self):
        return self.bot.player.entity.position.floored()

    def is_blockAt(self, x, y, z):
        return self.bot.blockAt(Vec3(x, y, z)).name != "air"

    def blockAt(self, x, y, z):
        return self.bot.blockAt(Vec3(x, y, z))

    def has_reached_goal(self, goal=None):
        if goal is None:
            goal = self.getGoal()
        if(self.bot.player is None): return False
        return self.bot.player.entity.position.distanceTo(goal) <= 1

    def is_on_ground(self):
        return self.bot.entity.onGround

    # 1 -> bot goes to positive z
    # -1 -> bot goes to negative z
    def get_rotation(self):
        return -1 if np.floor(self.bot.player.entity.yaw) == 0 else 1

    def wait_for_ticks(self, ticks):
        waitId = np.random.randint(0, 9999999)
        self.bot.wait(ticks, waitId)
        once(self.bot, "wait_complete")
        self.bot.ackWait(waitId)

    # The lower the weight the fast the action is done
    # We use this function to optimize the speed of the bot
    def action_weight(self, action):
        if action == 0:
            return 0
        elif action == 1:
            return 0
        elif action == 2:
            return 1
        elif action == 3:
            return 3
        elif action == 4:
            return 5

        return 0

    async def next_map(self):
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
            GOAL = self.bot.createParkourMap(10, START_POS).offset(0.5,1,0.5)
            print(f"New goal: {GOAL}")
            await asyncio.sleep(1)
        elif isinstance(SELECTED_MAP[map_index],int):
            GOAL = self.bot.createParkourMap(10, START_POS,SELECTED_MAP[map_index]).offset(0.5,1,0.5)
            print(f"New goal: {GOAL}")
            await asyncio.sleep(1)
        else:
            g = maps[SELECTED_MAP[map_index]]["goal"]
            GOAL = Vec3(g["x"], g["y"], g["z"])

    def getGoal(self):
        global GOAL
        return GOAL

    def getStartPos(self):
        global START_POS
        return START_POS

    def getMapIndex(self):
        global map_index
        return map_index

    def getMapCount(self):
        global SELECTED_MAP
        return len(SELECTED_MAP)

    def setStartPos(self,pos):
        global START_POS
        START_POS = pos
        return START_POS
