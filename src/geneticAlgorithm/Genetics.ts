import Client from "../game/Client";


class Genetics {

    constructor(private clients:Client[]) {

    }

    public async run() {
         await this.reset();
    }

    private async reset(){
        let promisses:Promise<void>[] = [];
        for(let client of this.clients){
            promisses.push(client.reset());
        }
        await Promise.all(promisses);
    }

}

export default Genetics;