

const mineflayer = require('mineflayer');
const Vec3 = require('vec3').Vec3;
const {toNotchianYaw} = require('../../../venv/lib/python3.10/site-packages/javascript/js/node_modules/mineflayer/lib/conversions.js');
function plugin(bot){

    function get_rotation (){
        return  Math.floor(bot.player.entity.yaw) === 0 ? -1: 1
    }

    bot.move_to_middle = () => {
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


}


function createBot(options) {
    const bot = mineflayer.createBot(options);
    bot.loadPlugin(plugin);
    return bot;
}

module.exports = createBot;