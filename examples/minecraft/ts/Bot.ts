import Client from "../../../src/game/Client";
import {once} from "events";

import mineflayer from "mineflayer"
import {Vec3} from "vec3"


class Bot implements Client  {
    bot!:mineflayer.Bot
    constructor(public username:string, public host: string, public pos:Vec3) {}

    async join(): Promise<Client> {
        this.bot = mineflayer.createBot({
            username:this.username,
            host:this.host
        })
        await once(this.bot,"spawn")
        return this
    }

    backward(): void{
        this.bot.clearControlStates()
        this.bot.setControlState("back", true)
    }

    forward(): void {
        this.bot.clearControlStates()
        this.bot.setControlState("forward", true)
    }


    jump(): void {
        this.bot.clearControlStates()
        this.bot.setControlState("jump", true)
    }

    async reset(){
        this.bot.clearControlStates()
        this.bot.chat(`/tp ${this.username} ${this.pos.x} ${this.pos.y} ${this.pos.z}`)
        await once(this.bot, "forceMove")
    }

    getPosition(): Vec3 {
        return this.bot.player.entity.position
    }

}

export default Bot