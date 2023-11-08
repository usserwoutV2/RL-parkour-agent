from javascript import require, console, On
import math
mineflayer = require("mineflayer", "latest")
Vec3 = require("vec3").Vec3


class Bot:
    def __init__(self,username:str, host:str, pos):
        self.host = host
        self.username = username
        self.pos = pos
        self.actionCount = 3 # forward, backward, jump
        self.active = True # if false, the bot will not do anything (eg when bot died)


    def join(self):
        self.bot = mineflayer.createBot({
            "host": self.host,
            "port": 25565,
            "version":"1.8.9",
            "username": self.username,
        })
        @On(self.bot, "death")
        def on_death():
            self.active = False
            console.log("died")
        return self
    
    def do_action(self,action:int):
        if not self.active:
            print("bot died, cannot do action")
            return 
        
        if action == 0:
            self.forward()
        elif action == 1:
            self.backward()
        elif action == 2:
            self.jump()
        else:
            console.log("invalid action")

# Rotate 180 degrees
    def rotate(self):
        self.bot.look(math.pi, 0, True)


    def backward(self):
        self.bot.clearControlStates()
        self.bot.look(math.pi, 0, True)
        # self.bot.setControlState("forward", True)
        # self.bot.setControlState("sprint", True)

    def forward(self):
        self.bot.clearControlStates()
        self.bot.look(0, 0, True)
        self.bot.setControlState("forward", True)
        self.bot.setControlState("sprint", True)

    def jump(self):
        self.bot.clearControlStates()
        self.bot.setControlState("jump", True)
        self.bot.setControlState("forward", True)
        self.bot.setControlState("sprint", True)

    def idle(self):
        self.bot.clearControlStates()

    def reset(self):
        self.bot.clearControlStates()
        self.active = True
        self.bot.chat(f'/tp {self.username} {self.pos["x"]} {self.pos["y"]} {self.pos["z"]}')

    def get_position(self):
        return self.bot.player.entity.position

    def get_position_floored(self):
        return self.bot.player.entity.position.floored()

    def is_blockAt(self, x, y, z):
        return self.bot.blockAt(Vec3(x, y, z)).name != "air"

    def has_reached_goal(self, goal):
        #console.log("===>",self.bot.player.entity.position,self.bot.player.entity.position.distanceTo(g))
        return self.bot.player.entity.position.distanceTo(goal) < 1

