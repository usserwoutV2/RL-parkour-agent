import Genetics from "./geneticAlgorithm/Genetics";
import Client from "./game/Client";
import {Vec3} from "vec3";


async function parkour(options:{clients: Client[], maxActionCount:number, goal:Vec3 }){
    const genetics = new Genetics(options.clients)
    await genetics.run();
}

export default parkour;