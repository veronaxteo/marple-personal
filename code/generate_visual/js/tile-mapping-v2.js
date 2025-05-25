const TILE_MAPPING = {
  BLANK: 238,
  FLOOR: {
    1: {
      TOP_LEFT: 144,
      TOP: 145,
      LEFT: 156,
      DEFAULT: 157
    },
    2: {
      TOP_LEFT: 147,
      TOP: 148,
      LEFT: 159,
      DEFAULT: 160
    },
    3: {
      TOP_LEFT: 150,
      TOP: 151,
      LEFT: 162,
      DEFAULT: 163
    }
  },
  WALL: {
    1: {
      TOP_LEFT: 72,
      TOP_RIGHT: 73,
      BOTTOM_LEFT: 84,
      BOTTOM_RIGHT: 85,
      TOP: 74,
      LEFT: 75,
      RIGHT: 76,
      BOTTOM: 77
    },
    2: {
      TOP_LEFT: 96,
      TOP_RIGHT: 97,
      BOTTOM_LEFT: 108,
      BOTTOM_RIGHT: 109,
      TOP: 98,
      LEFT: 99,
      RIGHT: 100,
      BOTTOM: 101
    },
    3: {
      TOP_LEFT: 120,
      TOP_RIGHT: 121,
      BOTTOM_LEFT: 132,
      BOTTOM_RIGHT: 133,
      TOP: 122,
      LEFT: 123,
      RIGHT: 124,
      BOTTOM: 125
    }
  },
  WALL_TOPPER: {
    1: {
      TOP_LEFT: 78,
      TOP_RIGHT: 79,
      TOP: 86,
    },
    2: {
      TOP_LEFT: 102,
      TOP_RIGHT: 103,
      TOP: 110,
    },
    3: {
      TOP_LEFT: 126,
      TOP_RIGHT: 127,
      TOP: 134
    }
  },
  DOOR: {
    HORIZONTAL: {
      CLOSED_TOP: 92,
      CLOSED_BOTTOM: 104,
      OPEN_TOP: 93,
      OPEN_BOTTOM: 105,
    },
    VERTICAL: {
     1: {
       CLOSED_TOP: 68,
       CLOSED_BOTTOM: 80,
       OPEN_TOP: 116,
       OPEN_BOTTOM: 128,
     },
     2: {
       CLOSED_TOP: 69,
       CLOSED_BOTTOM: 81,
       OPEN_TOP: 117,
       OPEN_BOTTOM: 129,
     },
     3: {
       CLOSED_TOP: 70,
       CLOSED_BOTTOM: 82,
       OPEN_TOP: 118,
       OPEN_BOTTOM: 130,
     }
    },
  },
  LIGHT:{
    ON:{
      TOP_LEFT: 4,
      TOP_MIDDLE: 5,
      TOP_RIGHT: 6,
      BOTTOM_LEFT: 16,
      BOTTOM_MIDDLE: 17,
      BOTTOM_RIGHT: 18,
    },
    OFF: {
      TOP: 3,
      BOTTOM: 15
    }
  },
  SOFA: {
    TOP_LEFT: 24,
    TOP_MIDDLE: 25,
    TOP_RIGHT: 26,
    BOTTOM_LEFT: 36,
    BOTTOM_MIDDLE: 37,
    BOTTOM_RIGHT: 38
  },
  REFRIGERATOR: {
    CLOSED: {
      TOP_LEFT: 7,
      TOP_RIGHT: 8,
      MIDDLE_LEFT: 19,
      MIDDLE_RIGHT: 20,
      BOTTOM_LEFT: 31,
      BOTTOM_RIGHT: 32
    },
    OPEN: {
      TOP_LEFT: 9,
      TOP_RIGHT: 10,
      TOP_DOOR: 11,
      MIDDLE_LEFT: 21,
      MIDDLE_RIGHT: 22,
      MIDDLE_DOOR: 23,
      BOTTOM_LEFT: 33,
      BOTTOM_RIGHT: 34,
      BOTTOM_DOOR: 35
    }
  },
  BED: {
    TOP_LEFT: 48,
    TOP_MIDDLE: 49,
    TOP_RIGHT: 50,
    BOTTOM_LEFT: 60,
    BOTTOM_MIDDLE: 61,
    BOTTOM_RIGHT: 62
  },
  TABLE: {
    TOP_LEFT: 51,
    TOP_MIDDLE: 52,
    TOP_RIGHT: 53,
    BOTTOM_LEFT: 63,
    BOTTOM_MIDDLE: 64,
    BOTTOM_RIGHT: 65
  },
  TV: {
    ON: {
      TOP_LEFT: 27,
      TOP_MIDDLE: 28,
      TOP_RIGHT: 29,
      BOTTOM_LEFT: 39,
      BOTTOM_MIDDLE: 40,
      BOTTOM_RIGHT: 41
    },
    OFF: {
      LEFT: 42,
      MIDDLE: 43,
      RIGHT: 44,
    }
  },
  SANDWICH: 0,
  CRUMBS: 1,
  SIDE_TABLE: 47,
  REMOTE: 46,
  PLANT: 12,

  AGENT: {
    A: {
      EAST_HEAD: 168,
      EAST_BODY: 180,
      SOUTH_HEAD: 169,
      SOUTH_BODY: 181,
      WEST_HEAD: 170,
      WEST_BODY: 182,
      NORTH_HEAD: 171,
      NORTH_BODY: 183,
    },
    B: {
      EAST_HEAD: 173,
      EAST_BODY: 185,
      SOUTH_HEAD: 174,
      SOUTH_BODY: 186,
      WEST_HEAD: 175,
      WEST_BODY: 187,
      NORTH_HEAD: 176,
      NORTH_BODY: 188,
    },
    C: {
      EAST_HEAD: 192,
      EAST_BODY: 204,
      SOUTH_HEAD: 193,
      SOUTH_BODY: 205,
      WEST_HEAD: 194,
      WEST_BODY: 206,
      NORTH_HEAD: 195,
      NORTH_BODY: 207,
    },
    D: {
      EAST_HEAD: 197,
      EAST_BODY: 209,
      SOUTH_HEAD: 198,
      SOUTH_BODY: 210,
      WEST_HEAD: 199,
      WEST_BODY: 211,
      NORTH_HEAD: 200,
      NORTH_BODY: 212,
    },
    E: {
      EAST_HEAD: 216,
      EAST_BODY: 228,
      SOUTH_HEAD: 217,
      SOUTH_BODY: 229,
      WEST_HEAD: 218,
      WEST_BODY: 230,
      NORTH_HEAD: 219,
      NORTH_BODY: 231,
    },
    F: {
      EAST_HEAD: 221,
      EAST_BODY: 233,
      SOUTH_HEAD: 222,
      SOUTH_BODY: 234,
      WEST_HEAD: 223,
      WEST_BODY: 235,
      NORTH_HEAD: 224,
      NORTH_BODY: 236,
    }
  }
};


export default TILE_MAPPING;
