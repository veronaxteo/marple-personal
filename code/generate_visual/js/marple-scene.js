import TILES from "./tile-mapping-v2.js";
export let tileset = "marple_cogsci_v2.png";
export let exp = "v2_demo";
export let gridName = "demo";

let grid;
import(`../trials_${exp}/js/${gridName}.js`).then(module => {
  grid = module.default;
}).catch(error => {
  console.error("Failed to load the grid:", error);
});

export default class MarpleScene extends Phaser.Scene {
  constructor() {
    super();
    this.mission = "";
    this.gridName = gridName;
  }

  preload() {
    this.load.image(
      "tiles",
      `../tilesets/${tileset}`
    );
  }

  create() {
    this.grid = JSON.parse(grid);
    console.log(this.grid);

    // CREATE BLANK MAP
    const map = this.make.tilemap({
      tileWidth: 32,
      tileHeight: 32,
      width: this.grid.width,
      height: this.grid.height + 1,
    });

    const tileset = map.addTilesetImage("tiles", null);
    this.floor = map.createBlankLayer("Floor", tileset);
    this.background = map.createBlankLayer("Background", tileset);
    this.furniture = map.createBlankLayer("Furniture", tileset);
    this.objects = map.createBlankLayer("Objects", tileset);
    this.agent = map.createBlankLayer("Agent", tileset);
    this.foreground = map.createBlankLayer("Foreground", tileset);

    const wall_type = this.grid.wall_type;
    const floor_type = this.grid.floor_type;

    // 1. ROOMS
    this.grid.rooms.initial.forEach(room => {
        // NOTE: +1 TO ALL TOP POSITIONS TO ALLOW FOR EXTRA WALL TOPPER IN ROW 0
        const top = room.top[1] + 1;
        const left = room.top[0];
        const width = room.size[0];
        const height = room.size[1];
        const right = left + width - 1;
        const bottom = top + height - 1;

        // FLOORS
        let TF = TILES.FLOOR[`${floor_type}`];
        this.floor.putTileAt(TF.TOP_LEFT, left, top);
        this.floor.fill(TF.TOP, left + 1, top, width - 1, 1);
        this.floor.fill(TF.LEFT, left, top + 1, 1, height - 1);
        this.floor.fill(TF.DEFAULT, left + 1, top + 1, width - 1, height - 1);

        // ROOM CORNERS AND WALLS
        const TW = TILES.WALL[`${wall_type}`];
        this.background.putTileAt(TW.TOP_LEFT, left - 1, top - 1);
        this.background.putTileAt(TW.TOP_RIGHT, right + 1, top - 1);
        this.background.putTileAt(TW.BOTTOM_RIGHT, right + 1, bottom + 1);
        this.background.putTileAt(TW.BOTTOM_LEFT, left - 1, bottom + 1);
        this.background.fill(TW.TOP, left, top - 1, width, 1);
        this.background.fill(TW.BOTTOM, left, bottom + 1, width, 1);
        this.background.fill(TW.LEFT, left - 1, top, 1, height);
        this.background.fill(TW.RIGHT, right + 1, top, 1, height);

        // WALL TOPPERS
        // these are above agent so agent walks behind them
        const TWT = TILES.WALL_TOPPER[`${wall_type}`];
        this.foreground.putTileAt(TWT.TOP_LEFT, left - 1, top - 2);
        // only at rightmost edge
        if (right + 2 == map.width) {
            this.foreground.putTileAt(TWT.TOP_RIGHT, right + 1, top - 2);
        }
        this.foreground.fill(TWT.TOP, left, top - 2, width, 1);

        // FURNITURE
        room.furnitures.initial.forEach(item => {
          const x = item.pos[0];
          // + 1 for extra row of wall topper
          const y = item.pos[1] + 1;
          if (item.type === 'light'){
            if (item.state.toggleable == 1){
              const TLO = TILES.LIGHT['ON'];
              this.furniture.putTileAt(TLO.TOP_LEFT, x - 1, y);
              this.furniture.putTileAt(TLO.TOP_MIDDLE, x, y);
              this.furniture.putTileAt(TLO.TOP_RIGHT, x + 1, y);
              this.furniture.putTileAt(TLO.BOTTOM_LEFT, x - 1, y + 1);
              this.furniture.putTileAt(TLO.BOTTOM_MIDDLE, x, y + 1);
              this.furniture.putTileAt(TLO.BOTTOM_RIGHT, x + 1, y + 1);
            } else {
              const TLO = TILES.LIGHT['OFF'];
              this.furniture.putTileAt(TLO.TOP, x, y);
              this.furniture.putTileAt(TLO.BOTTOM, x, y + 1);
            }
          } else if (item.type === 'table') {
            const TT = TILES.TABLE;
            this.furniture.putTileAt(TT.TOP_LEFT, x, y);
            this.furniture.putTileAt(TT.TOP_MIDDLE, x + 1, y);
            this.furniture.putTileAt(TT.TOP_RIGHT, x + 2, y);
            this.furniture.putTileAt(TT.BOTTOM_LEFT, x, y + 1);
            this.furniture.putTileAt(TT.BOTTOM_MIDDLE, x + 1, y + 1);
            this.furniture.putTileAt(TT.BOTTOM_RIGHT, x + 2, y + 1);
          } else if (item.type === 'electric_refrigerator'){
            if (item.state.openable) {
            // openable true = currently closed
              const TR = TILES.REFRIGERATOR['CLOSED'];
              this.furniture.putTileAt(TR.TOP_LEFT, x, y);
              this.furniture.putTileAt(TR.TOP_RIGHT, x + 1, y);
              this.furniture.putTileAt(TR.MIDDLE_LEFT, x, y + 1);
              this.furniture.putTileAt(TR.MIDDLE_RIGHT, x + 1, y + 1);
              this.furniture.putTileAt(TR.BOTTOM_LEFT, x, y + 2);
              this.furniture.putTileAt(TR.BOTTOM_RIGHT, x + 1, y + 2);
            } else {
              const TR = TILES.REFRIGERATOR['OPEN'];
              this.furniture.putTileAt(TR.TOP_LEFT, x, y);
              this.furniture.putTileAt(TR.TOP_RIGHT, x + 1, y);
              this.furniture.putTileAt(TR.TOP_DOOR, x + 2, y);
              this.furniture.putTileAt(TR.MIDDLE_LEFT, x, y + 1);
              this.furniture.putTileAt(TR.MIDDLE_RIGHT, x + 1, y + 1);
              this.furniture.putTileAt(TR.MIDDLE_DOOR, x + 2, y + 1);
              this.furniture.putTileAt(TR.BOTTOM_LEFT, x, y + 2);
              this.furniture.putTileAt(TR.BOTTOM_RIGHT, x + 1, y + 2);
              this.furniture.putTileAt(TR.BOTTOM_DOOR, x + 2, y + 2);
            }
          } else if (item.type === 'bed'){
            const TB = TILES.BED;
            this.furniture.putTileAt(TB.TOP_LEFT, x, y);
            this.furniture.putTileAt(TB.TOP_MIDDLE, x + 1, y);
            this.furniture.putTileAt(TB.TOP_RIGHT, x + 2, y);
            this.furniture.putTileAt(TB.BOTTOM_LEFT, x, y + 1);
            this.furniture.putTileAt(TB.BOTTOM_MIDDLE, x + 1, y + 1);
            this.furniture.putTileAt(TB.BOTTOM_RIGHT, x + 2, y + 1);
          } else if (item.type === 'sofa'){
            const TS = TILES.SOFA;
            this.furniture.putTileAt(TS.TOP_LEFT, x, y);
            this.furniture.putTileAt(TS.TOP_MIDDLE, x + 1, y);
            this.furniture.putTileAt(TS.TOP_RIGHT, x + 2, y);
            this.furniture.putTileAt(TS.BOTTOM_LEFT, x, y + 1);
            this.furniture.putTileAt(TS.BOTTOM_MIDDLE, x + 1, y + 1);
            this.furniture.putTileAt(TS.BOTTOM_RIGHT, x + 2, y + 1);
          } else if (item.type === 'side_table'){
              this.furniture.putTileAt(TILES.SIDE_TABLE, x, y);
          } else if (item.type === 'crumbs'){
              this.furniture.putTileAt(TILES.CRUMBS, x, y);
          } else if (item.type === 'tv'){
            if (item.state.toggleable == 1){
              const TTO = TILES.TV['ON'];
              this.furniture.putTileAt(TTO.TOP_LEFT, x, y);
              this.furniture.putTileAt(TTO.TOP_MIDDLE, x + 1, y);
              this.furniture.putTileAt(TTO.TOP_RIGHT, x + 2, y);
              this.furniture.putTileAt(TTO.BOTTOM_LEFT, x, y + 1);
              this.furniture.putTileAt(TTO.BOTTOM_BOTTOM, x + 1, y + 1);
              this.furniture.putTileAt(TTO.BOTTOM_RIGHT, x + 2, y + 1);
            } else {
              const TTO = TILES.TV['OFF'];
              this.furniture.putTileAt(TTO.LEFT, x, y);
              this.furniture.putTileAt(TTO.MIDDLE, x + 1, y);
              this.furniture.putTileAt(TTO.RIGHT, x + 2, y);
            }
          }
        });

        // OBJECTS ON FURNITURE
        room.furnitures.initial.forEach(furniture => {
          furniture.objs.initial.forEach(item => {
            const x = item.pos[0];
            // + 1 for extra row of wall topper
            const y = item.pos[1] + 1;
            if (item.type === 'remote'){
              this.objects.putTileAt(TILES.REMOTE, x, y);
            }
          });
        });

    });

    // 2. DOORS
    this.grid.doors.initial.forEach(door => {
        const x = door.pos[0];
        // + 1 for extra row of wall topper
        const y = door.pos[1] + 1;
        const isVertical = door.dir === "vert";
        const isHorizontal = door.dir === "horz";
        let TF = TILES.FLOOR[`${floor_type}`];

        if (isVertical) {
          let TDV = TILES.DOOR.VERTICAL[`${wall_type}`];
          if (door.state === "open") {
            this.floor.fill(TF.DEFAULT, x, y - 1, 2, 2);
            this.background.putTileAt(TDV.OPEN_TOP, x, y - 1);
            this.background.putTileAt(TDV.OPEN_BOTTOM, x, y);
            // for agent walking through open door
            this.foreground.putTileAt(TDV.OPEN_TOP, x, y - 1);
            this.foreground.putTileAt(TDV.OPEN_BOTTOM, x, y);
          } else {
            this.floor.fill(TF.DEFAULT, x, y - 1, 1, 2);
            this.background.putTileAt(TDV.CLOSED_TOP, x, y - 1);
            this.background.putTileAt(TDV.CLOSED_BOTTOM, x, y - 1);         }
        } else if (isHorizontal) {
          let TDH = TILES.DOOR.HORIZONTAL;
          if (door.state === "open"){
            this.floor.fill(TF.DEFAULT, x, y, 1, 2);
            // extend one tile below to replace shadow
            this.background.putTileAt(TDH.OPEN_BOTTOM, x, y);
            this.foreground.putTileAt(TDH.OPEN_TOP, x, y - 1);
            // if any agents walking through, put doors in foreground
            this.grid.agents.initial.forEach(agent => {
                if ((agent.pos[0] == x) && (agent.pos[1] == y - 1)) {
                    this.foreground.putTileAt(TDH.OPEN_BOTTOM, x, y);
                }
            });
          } else {
            this.floor.fill(TF.DEFAULT, x, y, 1, 1);
            this.background.putTileAt(TDH.CLOSED_BOTTOM, x, y);
            this.foreground.putTileAt(TDH.CLOSED_TOP, x, y - 1);
          }
        }

    });

    // AGENTS
    this.grid.agents.initial.forEach(agent => {
      let x = agent.pos[0];
      // + 1 for extra row of wall topper
      let y = agent.pos[1] + 1;
      let direction = agent.dir;
      let TA = TILES.AGENT[`${agent.name}`];

      if (direction === 0) {
          this.agent.putTileAt(TA[`EAST_BODY`], x, y);
          this.agent.putTileAt(TA[`EAST_HEAD`], x, y - 1);
      } else if (direction === 1){
          this.agent.putTileAt(TA[`SOUTH_BODY`], x, y);
          this.agent.putTileAt(TA[`SOUTH_HEAD`], x, y - 1);
      } else if (direction === 2){
          this.agent.putTileAt(TA[`WEST_BODY`], x, y);
          this.agent.putTileAt(TA[`WEST_HEAD`], x, y - 1);
      } else if (direction === 3){
          this.agent.putTileAt(TA[`NORTH_BODY`], x, y);
          this.agent.putTileAt(TA[`NORTH_HEAD`], x, y - 1);
      }
    });
  }
}




