import logging
from typing import Dict, Tuple, Optional


class CoordinateMapper:
    """
    This class handles coordinate transformations between world space and kitchen space.
    """
    def __init__(self, kitchen_info: Dict):
        self.kitchen_info = kitchen_info
        self.kitchen_width = kitchen_info['size'][0]
        self.kitchen_height = kitchen_info['size'][1] 
        self.kitchen_top_x = kitchen_info['top'][0]
        self.kitchen_top_y = kitchen_info['top'][1]
        self.logger = logging.getLogger(self.__class__.__name__)

    # TODO: maybe add furniture size here?
    
    def world_to_kitchen_coords(self, world_x: int, world_y: int) -> Optional[Tuple[int, int]]:
        """Convert world coordinates to kitchen array coordinates"""
        if (self.kitchen_top_x <= world_x < self.kitchen_top_x + self.kitchen_width and
            self.kitchen_top_y <= world_y < self.kitchen_top_y + self.kitchen_height):
            return int(world_x - self.kitchen_top_x), int(world_y - self.kitchen_top_y)
        return None
    
    def kitchen_to_world_coords(self, kitchen_x: int, kitchen_y: int) -> Tuple[int, int]:
        """Convert kitchen array coordinates to world coordinates"""
        return int(kitchen_x + self.kitchen_top_x), int(kitchen_y + self.kitchen_top_y) 
    