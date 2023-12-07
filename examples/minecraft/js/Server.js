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
// @ts-ignore
const flying_squid_1 = __importDefault(require("flying-squid"));
const node_events_1 = require("node:events");
const dns = __importStar(require("dns"));
const createRandomMap_1 = __importDefault(require("./createRandomMap"));
dns.setDefaultResultOrder('ipv4first');
async function launchServer() {
    const server = flying_squid_1.default.createMCServer({
        'motd': 'AI parkour bot testing server',
        'port': 25565,
        'max-players': 10,
        'online-mode': false,
        'logging': true,
        'gameMode': 1,
        'difficulty': 1,
        'worldFolder': 'world',
        'generation': {
            'name': 'superflat',
            'options': {
                'worldHeight': 80
            }
        },
        'kickTimeout': 10000,
        'plugins': {},
        'modpe': false,
        'view-distance': 10,
        'player-list-text': {
            'header': 'AI ',
            'footer': 'Server used for AI bot training'
        },
        'everybody-op': true,
        'max-entities': 100,
        'version': '1.8.9',
    });
    await (0, node_events_1.once)(server, "listening");
    console.log("\x1b[1;33mCONNECT TO FOLLOWING IP: " + server._server.socketServer._connectionKey + "\x1b[0m");
    server.commands.add({
        base: 'map',
        info: 'Returns a random number from 0 to num',
        usage: '/random <num>',
        parse(str) {
            const match = str.match(/^\d+$/); // Check to see if they put numbers in a row
            if (!match)
                return 10; // Anything else, show them the usage
            else
                return parseInt(match[0]); // Otherwise, pass our number as an int to action()
        },
        action(jumps, ctx) {
            ctx.player.chat(`Generating parkour map...`);
            let coordinates = (0, createRandomMap_1.default)(ctx.player.position.floored(), jumps);
            coordinates.forEach((coordinate) => {
                console.log(coordinate);
                ctx.player.setBlock(coordinate, 1, 6);
                server.setBlock(ctx.player.world, coordinate, 1, 6);
            });
            ctx.player.chat(`Map ${server.color.green}successfully${server.color.reset} generated!`);
        }
    });
}
launchServer().then(() => console.log("Server launched")).catch((err) => console.log(err));
exports.default = launchServer;
//# sourceMappingURL=Server.js.map