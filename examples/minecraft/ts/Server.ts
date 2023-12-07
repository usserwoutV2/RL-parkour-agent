// @ts-ignore
import MCServer from 'flying-squid'
import {once} from "node:events";
import * as dns from "dns";
import {Vec3} from "vec3";
import blockCoordinates from "./createRandomMap";
dns.setDefaultResultOrder('ipv4first')
async function launchServer():Promise<void>{
    const server = MCServer.createMCServer({
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
        'plugins': {

        },
        'modpe': false,
        'view-distance': 10,
        'player-list-text': {
            'header': 'AI ',
            'footer': 'Server used for AI bot training'
        },
        'everybody-op': true,
        'max-entities': 100,
        'version': '1.8.9',
    })
    await once(server,"listening")
    console.log("\x1b[1;33mCONNECT TO FOLLOWING IP: "+ server._server.socketServer._connectionKey+"\x1b[0m")

    server.commands.add({
        base: 'map', // This is what the user starts with, so in this case: /map
        info: 'Returns a random number from 0 to num', // Description of the command
        usage: '/random <num>', // Usage displayed if parse() returns false (which means they used it incorrectly)
        parse(str:string) { // str contains everything after "/random "
          const match = str.match(/^\d+$/); // Check to see if they put numbers in a row
          if (!match) return 10; // Anything else, show them the usage
          else return parseInt(match[0]); // Otherwise, pass our number as an int to action()
        },
        action(jumps:number, ctx:any) { // ctx - context who is using it
            ctx.player.chat(`Generating parkour map...`);
            let coordinates:Vec3[] = blockCoordinates(ctx.player.position.floored(), jumps)
            coordinates.forEach((coordinate)=>{
                console.log(coordinate)
                ctx.player.setBlock(coordinate, 1,6)
                server.setBlock(ctx.player.world,coordinate, 1,6)
            })
            ctx.player.chat(`Map ${server.color.green}successfully${server.color.reset} generated!`);

        }
    })
} 

launchServer().then(()=>console.log("Server launched")).catch((err)=>console.log(err))
export default launchServer