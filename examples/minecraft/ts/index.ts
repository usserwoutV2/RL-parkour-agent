import startServer from "./Server";
import Client from "../../../src/game/Client";
import Bot from "./Bot";
import {Vec3} from "vec3";

const BOT_AMOUNT = 10
const START_POS = new Vec3(0,5,0)
async function main(){
    await startServer();
    const clientsList:Promise<Client>[] = []
    for(let i = 0; i < BOT_AMOUNT; i++){
        const client = new Bot("bot_"+i,"",START_POS);
        clientsList.push(client.join());
    }
    const clients = await Promise.all(clientsList)

}
main()