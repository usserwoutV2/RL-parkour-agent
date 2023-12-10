import {Vec3} from "vec3";

enum JUMP_TYPE {
    JUMP_1,
    JUMP_1_UP,
    JUMP_1_DOWN,
    JUMP_2,
    JUMP_2_DOWN,
    JUMP_2_UP,
    JUMP_3_DOWN,
    JUMP_3,
    JUMP_3_UP,
    JUMP_4_DOWN,
    JUMP_4,
    JUMP_5_DOWN,
}


function jumpType(jump:JUMP_TYPE){
    switch(jump){
        case JUMP_TYPE.JUMP_1:
            return [1,0]
        case JUMP_TYPE.JUMP_1_UP:
            return [1,1]
        case JUMP_TYPE.JUMP_1_DOWN:
            return [1,-1]
        case JUMP_TYPE.JUMP_2:
            return [2,0]
        case JUMP_TYPE.JUMP_2_DOWN:
            return [2,-1]
        case JUMP_TYPE.JUMP_2_UP:
            return [2,1]
        case JUMP_TYPE.JUMP_3_DOWN:
            return [3,-1]
        case JUMP_TYPE.JUMP_3:
            return [3,0]
        case JUMP_TYPE.JUMP_3_UP:
            return [3,1]
        case JUMP_TYPE.JUMP_4_DOWN:
            return [4,-1]
        case JUMP_TYPE.JUMP_4:
           return [4,0]
        case JUMP_TYPE.JUMP_5_DOWN:
            return [5,-1]
    }
}

function addJump(startPos:Vec3,type:null | JUMP_TYPE = null ) {
    let jump:JUMP_TYPE = type === null ? Math.floor(Math.random() * (JUMP_TYPE.JUMP_4+1)): type
    let jumpOffset = jumpType(jump)
    return startPos.offset(0,jumpOffset[1], -jumpOffset[0])
}

function blockCoordinates(startPos:Vec3, length:number,jumpType: JUMP_TYPE|null = null):Vec3[] {
    let coordinates:Vec3[] = []
    coordinates.push(startPos.offset(0,-1,0).floored())
    for(let i = 0; i < length; i++){
        coordinates.push(addJump(coordinates[i],jumpType))
    }
    return coordinates
}

export default blockCoordinates

