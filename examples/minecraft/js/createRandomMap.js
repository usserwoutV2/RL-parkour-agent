"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var JUMP_TYPE;
(function (JUMP_TYPE) {
    JUMP_TYPE[JUMP_TYPE["JUMP_1"] = 0] = "JUMP_1";
    JUMP_TYPE[JUMP_TYPE["JUMP_1_UP"] = 1] = "JUMP_1_UP";
    JUMP_TYPE[JUMP_TYPE["JUMP_1_DOWN"] = 2] = "JUMP_1_DOWN";
    JUMP_TYPE[JUMP_TYPE["JUMP_2"] = 3] = "JUMP_2";
    JUMP_TYPE[JUMP_TYPE["JUMP_2_DOWN"] = 4] = "JUMP_2_DOWN";
    JUMP_TYPE[JUMP_TYPE["JUMP_2_UP"] = 5] = "JUMP_2_UP";
    JUMP_TYPE[JUMP_TYPE["JUMP_3_DOWN"] = 6] = "JUMP_3_DOWN";
    JUMP_TYPE[JUMP_TYPE["JUMP_3"] = 7] = "JUMP_3";
    JUMP_TYPE[JUMP_TYPE["JUMP_3_UP"] = 8] = "JUMP_3_UP";
    JUMP_TYPE[JUMP_TYPE["JUMP_4_DOWN"] = 9] = "JUMP_4_DOWN";
    JUMP_TYPE[JUMP_TYPE["JUMP_4"] = 10] = "JUMP_4";
    JUMP_TYPE[JUMP_TYPE["JUMP_5_DOWN"] = 11] = "JUMP_5_DOWN";
})(JUMP_TYPE || (JUMP_TYPE = {}));
function jumpType(jump) {
    switch (jump) {
        case JUMP_TYPE.JUMP_1:
            return [1, 0];
        case JUMP_TYPE.JUMP_1_UP:
            return [1, 1];
        case JUMP_TYPE.JUMP_1_DOWN:
            return [1, -1];
        case JUMP_TYPE.JUMP_2:
            return [2, 0];
        case JUMP_TYPE.JUMP_2_DOWN:
            return [2, -1];
        case JUMP_TYPE.JUMP_2_UP:
            return [2, 1];
        case JUMP_TYPE.JUMP_3_DOWN:
            return [3, -1];
        case JUMP_TYPE.JUMP_3:
            return [3, 0];
        case JUMP_TYPE.JUMP_3_UP:
            return [3, 1];
        case JUMP_TYPE.JUMP_4_DOWN:
            return [4, -1];
        case JUMP_TYPE.JUMP_4:
            return [4, 0];
        case JUMP_TYPE.JUMP_5_DOWN:
            return [5, -1];
    }
}
function addJump(startPos, type = null) {
    let jump = type === null ? Math.floor(Math.random() * (JUMP_TYPE.JUMP_4 + 1)) : type;
    let jumpOffset = jumpType(jump);
    return startPos.offset(0, jumpOffset[1], -jumpOffset[0]);
}
function blockCoordinates(startPos, length, jumpType = null) {
    let coordinates = [];
    coordinates.push(startPos.offset(0, -1, 0).floored());
    for (let i = 0; i < length; i++) {
        coordinates.push(addJump(coordinates[i], jumpType));
    }
    return coordinates;
}
exports.default = blockCoordinates;
//# sourceMappingURL=createRandomMap.js.map