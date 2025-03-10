"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const vec3_1 = __importStar(require("vec3"));
const createRandomMap_1 = __importDefault(require("./createRandomMap"));
const mineflayer_1 = __importDefault(require("mineflayer"));
const { toNotchianYaw } = require('../node_modules/mineflayer/lib/conversions.js');
function plugin(bot) {
    function get_rotation() {
        return Math.floor(bot.player.entity.yaw) === 0 ? -1 : 1;
    }
    bot.get_actual_position = () => {
        const pos = bot.player.entity.position;
        const rot = get_rotation();
        const args = {
            "x": Math.floor(pos.x) + 0.5,
            "y": Math.floor(pos.y),
            "z": Math.floor(pos.z) + 0.5,
            "yaw": toNotchianYaw(rot === -1 ? 0 : Math.PI),
            "pitch": 0,
        };
        const v = new vec3_1.Vec3(args.x, args.y - 1, args.z).floored();
        if (bot.blockAt(v)?.name === 'air') {
            v.z -= rot;
            if (bot.blockAt(v)?.name === 'air')
                args.z += rot;
            else
                args.z -= rot;
        }
        return args;
    };
    bot.get_actual_position_floored = () => {
        const pos = bot.get_actual_position();
        return new vec3_1.Vec3(pos.x, pos.y, pos.z).floored();
    };
    let acked_time = {};
    bot.wait = (ticks, id) => {
        bot.waitForTicks(ticks).then(() => {
            acked_time[id] = setInterval(() => {
                // @ts-ignore
                bot.emit("wait_complete");
            }, 50);
            // @ts-ignore
            bot.emit("wait_complete");
        });
    };
    bot.ackWait = (id) => {
        clearInterval(acked_time[id]);
    };
    bot.move_to_middle = () => {
        let args = bot.get_actual_position();
        bot._client.write('position_look', args);
        bot._client.emit('position', {
            x: args.x,
            y: args.y,
            z: args.z,
            yaw: args.yaw,
            pitch: args.pitch,
            flags: 0,
            teleportId: 0,
        });
    };
    bot.isBlockBelow = (blockName) => {
        let p = bot.player.entity.position;
        let pos2 = bot.player.entity.position.floored();
        const b = bot.blockAt(pos2.offset(0, -1, 0));
        if (!b)
            return false;
        if (b.name === blockName && Math.abs(p.y - pos2.y) < 0.01)
            return true;
        let pos = bot.player.entity.position.offset(0, 0, -0.2999).floored();
        return bot.blockAt(pos.offset(0, -1, 0))?.name === blockName && Math.abs(p.y - pos.y) < 0.01;
    };
    function clearMap(startPos, jumps) {
        // Fist we remove all blocks
        for (let z = 1; z < jumps * 5; z++) {
            for (let y = Math.max(-startPos.y, -15); y < 15; y++) {
                const block = bot.blockAt(startPos.offset(0, y, -z));
                if (!block || block.type === 0 || block.name === 'redstone_block')
                    continue;
                bot.chat(`/setblock ${block.position.x} ${block.position.y} ${block.position.z} 0`);
            }
        }
    }
    bot.createParkourMap = (jumps, _startPos = bot.get_actual_position_floored(), jumpType) => {
        let startPos;
        if (!(_startPos instanceof vec3_1.default)) {
            startPos = new vec3_1.Vec3(_startPos.x, _startPos.y, _startPos.z).floored();
        }
        else
            startPos = _startPos;
        clearMap(startPos, jumps);
        let coordinates = (0, createRandomMap_1.default)(startPos, jumps, jumpType);
        coordinates.forEach((coordinate) => {
            bot.chat(`/setblock ${coordinate.x} ${coordinate.y} ${coordinate.z} 1 6`);
        });
        return coordinates[coordinates.length - 1];
    };
}
function createBot(options) {
    let bot = mineflayer_1.default.createBot(options);
    bot.loadPlugin(plugin);
    bot.on("end", () => {
        console.log("Disconnected...");
    });
    return bot;
}
module.exports = createBot;
//# sourceMappingURL=createBot.js.map