// @ts-ignore
import MCServer from 'flying-squid'
import {once} from "node:events";
import * as dns from "dns";
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
} 

launchServer().then(()=>console.log("Server launched")).catch((err)=>console.log(err))
export default launchServer