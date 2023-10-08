import {Vec3} from "vec3";

/**
 * A client that will perform the actions
 */
interface Client  {


    /**
     * Connects to the server
     */
    join(): Promise<Client>;

    jump(): void;

    forward(): void;

    backward():void;

    /**
     * Go back to the begin position
     */
    reset():Promise<void>;

    getPosition():Vec3
}

export default Client