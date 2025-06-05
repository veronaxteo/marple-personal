import logging
from typing import Dict, List, Tuple, Optional
from igraph import Graph
from ...utils.math_utils import furniture_size


class WorldGeometry:
    """
    This class handles the geometry of the world, including the positions of various furniture and doors.
    """    
    def __init__(self, world_info: Dict):
        self.world_info = world_info
        self.logger = logging.getLogger(self.__class__.__name__)
        self._valid_kitchen_crumb_coords_cache = None
    
    def is_furniture_at(self, location_tuple: Tuple[int, int]) -> bool:
        """Check if world coordinate is occupied by furniture (excluding crumbs)"""
        loc_x, loc_y = location_tuple
        for room in self.world_info['rooms']['initial']:
            for furniture in room['furnitures']['initial']:
                if furniture['type'] == 'crumbs':
                    continue
                f_x, f_y = furniture['pos']
                f_w, f_h = furniture_size.get(furniture['type'], (1, 1))
                if (f_x <= loc_x < f_x + f_w and f_y <= loc_y < f_y + f_h):
                    return True
        return False
    
    def get_initial_door_states(self) -> Dict[Tuple[int, int], str]:
        """Get initial door states as coordinate->state mapping"""
        return {tuple(door['pos']): door['state'] for door in self.world_info['doors']['initial']}
    
    def get_fridge_access_point(self) -> Optional[Tuple[int, int]]:
        """Get fridge access point coordinate from world info"""
        kitchen_info = next((r for r in self.world_info['rooms']['initial'] if r['type'] == 'Kitchen'), None)
        fridge_info = next((f for f in kitchen_info['furnitures']['initial'] if f['type'] == 'electric_refrigerator'), None)
        fp = fridge_info['pos']
        fridge_access_point = (fp[0] - 1, fp[1] + 2)
        
        return fridge_access_point
    
    def get_valid_kitchen_crumb_coords(self, kitchen_info: Dict, node_to_vid: Dict, igraph: Graph) -> List[Tuple[int, int]]:
        """Get list of valid world coordinates for crumbs in kitchen"""
        if self._valid_kitchen_crumb_coords_cache is not None:
            return self._valid_kitchen_crumb_coords_cache
        
        valid_coords = []
        kx, ky = kitchen_info['top']
        kw, kh = kitchen_info['size']
        
        for world_y in range(ky, ky + kh):
            for world_x in range(kx, kx + kw):
                coord_tuple = (world_x, world_y)
                if coord_tuple in node_to_vid:
                    vid = node_to_vid[coord_tuple]
                    if not igraph.vs[vid]['is_door']:
                        valid_coords.append(coord_tuple)
        
        self._valid_kitchen_crumb_coords_cache = sorted(valid_coords)
        return self._valid_kitchen_crumb_coords_cache 
    