import {Bot} from "mineflayer";
import vec3,{Vec3} from "vec3";
import createRandomMap from "./createRandomMap";
import mineflayer from "mineflayer";
const {toNotchianYaw} = require('../../../venv/lib/python3.10/site-packages/javascript/js/node_modules/mineflayer/lib/conversions.js');


type BotExtentions = {
    get_actual_position: () => {x:number, y:number, z:number, yaw:number, pitch:number},
    get_actual_position_floored: () => Vec3,
    move_to_middle: () => void,
    isBlockBelow: (blockName:string) => boolean,
    createParkourMap: (jumps:number,startPos:any) => Vec3,
    createTypeParkourMap: (jumps:number,startPos:any, jumpType:number) => Vec3,
}

function plugin(bot:Bot & BotExtentions){

    function get_rotation (){
        return  Math.floor(bot.player.entity.yaw) === 0 ? -1: 1
    }

    bot.get_actual_position = () => {

        const pos = bot.player.entity.position;
        const rot = get_rotation();
        const args = {
            "x": Math.floor(pos.x) + 0.5,
            "y": Math.floor(pos.y),
            "z": Math.floor(pos.z) + 0.5,
            "yaw":  toNotchianYaw(rot === -1? 0 : Math.PI),
            "pitch": 0,
        }

        if(bot.blockAt(new Vec3(args.x, args.y-1, args.z).floored())?.name === 'air'){
            args.z -= rot;
        }
        return args
    }

    bot.get_actual_position_floored =    () => {
        const pos = bot.get_actual_position()
        return new Vec3(pos.x, pos.y, pos.z).floored()
    }

    bot.move_to_middle = () => {
        let args = bot.get_actual_position()

        bot._client.write('position_look', args)

        bot._client.emit('position', {
            x: args.x,
            y: args.y,
            z: args.z,
            yaw: args.yaw,
            pitch: args.pitch,
            flags: 0,
            teleportId: 0,
          })
    }


    bot.isBlockBelow = (blockName) => {
        let p = bot.player.entity.position
        let pos2 =  bot.player.entity.position.floored()
        const b = bot.blockAt(pos2.offset(0,-1,0))
        if(!b) return false;
        if (b.name === blockName && Math.abs(p.y - pos2.y) < 0.01)
            return true
        let pos = bot.player.entity.position.offset(0, 0, -0.2999).floored()
        return bot.blockAt(pos.offset(0,-1,0))?.name === blockName && Math.abs(p.y - pos.y) < 0.01;
    }

    function clearMap(startPos:Vec3, jumps:number){
        // Fist we remove all blocks
        for(let z = 1; z < jumps*5; z++){
            for(let y= Math.max(-startPos.y,-15); y < 15; y++){
                const block = bot.blockAt(startPos.offset(0,y,-z))
                if(!block || block.type === 0 || block.name === 'redstone_block') continue;
                bot.chat(`/setblock ${block.position.x} ${block.position.y} ${block.position.z} 0`)
            }
        }
    }

    bot.createParkourMap = (jumps:number,_startPos:Vec3| {x:number, y:number, z:number} = bot.get_actual_position_floored(),jumpType?:number ) => {
        let startPos: Vec3;
        if (!(_startPos instanceof vec3)) {
            startPos = new Vec3(_startPos.x, _startPos.y, _startPos.z).floored()
        } else startPos = _startPos as Vec3;

        clearMap(startPos, jumps)


        let coordinates:Vec3[] = createRandomMap(startPos as Vec3, jumps,jumpType)
        coordinates.forEach((coordinate)=>{
            bot.chat(`/setblock ${coordinate.x} ${coordinate.y} ${coordinate.z} 1 6`)
        })
        return coordinates[coordinates.length-1]
    }
}


function createBot(options:mineflayer.BotOptions) {
    let bot = mineflayer.createBot(options);
    bot.loadPlugin(plugin as any);
    bot.on("end",()=> {
        console.log("Disconnected...");
    })
    return bot;
}

module.exports = createBot;