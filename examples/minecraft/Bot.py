from javascript import require, console, On
import math
import numpy as np
import asyncio

# mineflayer = require("mineflayer", "latest")
createBot = require("./ts/createBot.js")
Vec3 = require("vec3").Vec3

null_vector = Vec3(0, 0, 0)


class Bot:

    def __init__(self, username: str, host: str, pos, actionCount=3):
        self.host = host
        self.username = username
        self.pos = pos
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
            await asyncio.sleep(0.3)
            self.bot.setControlState("forward", True)
            self.bot.setControlState("sprint", True)
            await asyncio.sleep(0.25)
            self.bot.clearControlStates()
            while self.bot.entity.velocity.z != 0 or not self.bot.entity.onGround:
                await asyncio.sleep(0.1)  # Wait until the bot is on the ground
        elif action == 1:
            self.rotate()
            await asyncio.sleep(min_time)
            self.bot.clearControlStates()
        elif action == 2 or action == 3 or action == 4:
            self.bot.move_to_middle()
            await asyncio.sleep(0.3)
            self.bot.setControlState("forward", True)
            if action != 2:
                self.bot.setControlState("sprint", True)
            await asyncio.sleep(0.10 if action == 3 else 0.25)
            self.bot.setControlState("jump", True)
            self.bot.setControlState("jump", False)
            await asyncio.sleep(min_time)
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
            pos = self.pos
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

    def has_reached_goal(self, goal):
        # console.log("===>",self.bot.player.entity.position.distanceTo(goal))
        return self.bot.player.entity.position.distanceTo(goal) <= 1

    def is_on_ground(self):
        return self.bot.entity.onGround

    # 1 -> bot goes to positive z
    # -1 -> bot goes to negative z
    def get_rotation(self):
        return -1 if np.floor(self.bot.player.entity.yaw) == 0 else 1
