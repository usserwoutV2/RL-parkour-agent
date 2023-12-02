

const mineflayer = require('mineflayer');
const Vec3 = require('vec3').Vec3;
const {toNotchianYaw} = require('../../../venv/lib/python3.10/site-packages/javascript/js/node_modules/mineflayer/lib/conversions.js');
function plugin(bot){

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

        if(bot.blockAt(new Vec3(args.x, args.y-1, args.z).floored()).name === 'air'){
            args.z -= rot;
        }
        return args
    }

    bot.get_actual_position_floored =    () => {
        const pos = bot.get_actual_position()
        return new Vec3(pos.x, pos.y, pos.z).floored()
    }

    bot.move_to_middle = () => {
        args = bot.get_actual_position()

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
        return bot.blockAt(pos.offset(0,-1,0)).name === blockName && Math.abs(p.y - pos.y) < 0.01;

    }


}


function createBot(options) {
    let bot = mineflayer.createBot(options);
    bot.loadPlugin(plugin);
    bot.on("end",()=> {
        console.log("Disconnected...");
    })
    return bot;
}

module.exports = createBot;