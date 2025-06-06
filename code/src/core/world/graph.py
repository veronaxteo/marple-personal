import logging
from typing import Dict, List, Tuple, Optional
from igraph import Graph
from ...utils.math_utils import furniture_size


class WorldGraph:
    """
    This class creates the graph representation of the world, including the nodes and edges.
    Also handles simple pathfinding operations.
    """
    def __init__(self):
        self.igraph = None
        self.node_to_vid = {}
        self.vid_to_node = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_graph(self, world_info: Dict, geometry) -> None:
        """Create the igraph representation of the world."""
        self.igraph = Graph(directed=False)
        self.node_to_vid = {}
        self.vid_to_node = {}
        
        # Collect all nodes to add
        all_nodes_to_add = []
        edges_to_add = []
        
        # Add room nodes
        for room in world_info['rooms']['initial']:
            room_coords = []
            for x_coord in range(room['top'][0], room['top'][0] + room['size'][0]):
                for y_coord in range(room['top'][1], room['top'][1] + room['size'][1]):
                    loc = (x_coord, y_coord)
                    if not geometry.is_furniture_at(loc):
                        room_coords.append(loc)
                        if loc not in self.node_to_vid:
                            all_nodes_to_add.append({
                                'name': str(loc),
                                'coords': loc,
                                'is_door': False,
                                'room': room['type']
                            })
            
            # Add room edges
            for coord in room_coords:
                neighbor_h = (coord[0] + 1, coord[1])
                if neighbor_h in room_coords:
                    edges_to_add.append((coord, neighbor_h))
                neighbor_v = (coord[0], coord[1] + 1)
                if neighbor_v in room_coords:
                    edges_to_add.append((coord, neighbor_v))
        
        # Add door nodes
        for door_info in world_info['doors']['initial']:
            loc = tuple(door_info['pos'])
            if loc not in self.node_to_vid:
                all_nodes_to_add.append({
                    'name': str(loc),
                    'coords': loc,
                    'is_door': True,
                    'state': door_info['state'],
                    'room': None
                })
        
        # Batch add vertices
        self.igraph.add_vertices(len(all_nodes_to_add))
        for i, node_attrs in enumerate(all_nodes_to_add):
            self.igraph.vs[i]['name'] = node_attrs['name']
            self.igraph.vs[i]['coords'] = node_attrs['coords']
            self.igraph.vs[i]['is_door'] = node_attrs['is_door']
            self.igraph.vs[i]['room'] = node_attrs.get('room')
            if node_attrs['is_door']:
                self.igraph.vs[i]['state'] = node_attrs.get('state')
            
            self.node_to_vid[node_attrs['coords']] = i
            self.vid_to_node[i] = node_attrs['coords']
        
        # Batch add room edges
        igraph_edges = []
        for u_coord, v_coord in edges_to_add:
            if u_coord in self.node_to_vid and v_coord in self.node_to_vid:
                igraph_edges.append((self.node_to_vid[u_coord], self.node_to_vid[v_coord]))
        self.igraph.add_edges(igraph_edges)
        
        # Add door connection edges
        self._add_door_connections(world_info)
    
    def _add_door_connections(self, world_info: Dict) -> None:
        """Add edges connecting doors to adjacent rooms."""
        door_connection_edges = []
        for door_info in world_info['doors']['initial']:
            door_pos = tuple(door_info['pos'])
            potential_neighbors = []
            if door_info['dir'] == 'horz':
                potential_neighbors = [(door_pos[0], door_pos[1]-1), (door_pos[0], door_pos[1]+1)]
            elif door_info['dir'] == 'vert':
                potential_neighbors = [(door_pos[0]-1, door_pos[1]), (door_pos[0]+1, door_pos[1])]
            
            if door_pos in self.node_to_vid:
                door_vid = self.node_to_vid[door_pos]
                for neighbor_pos in potential_neighbors:
                    if neighbor_pos in self.node_to_vid:
                        neighbor_vid = self.node_to_vid[neighbor_pos]
                        door_connection_edges.append(tuple(sorted((door_vid, neighbor_vid))))
        
        # Add unique door edges
        unique_door_edges = sorted(list(set(door_connection_edges)))
        self.igraph.add_edges(unique_door_edges)
    
    def find_closest_door_to_agent(self, agent_start_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Find closest door node to agent start position.
        """
        if agent_start_pos not in self.node_to_vid:
            self.logger.error(f"Agent start position {agent_start_pos} not found in graph")
            return None
        
        source_vid = self.node_to_vid[agent_start_pos]
        
        # Find all door vertices
        door_vids_and_pos = []
        for vid in range(len(self.igraph.vs)):
            vertex = self.igraph.vs[vid]
            try:
                if vertex['is_door'] is True and 'coords' in vertex.attributes():
                    door_vids_and_pos.append((vid, vertex['coords']))
            except KeyError:
                pass
        
        if not door_vids_and_pos:
            self.logger.warning("No valid door vertices found in graph")
            return None
        
        # Find closest door
        closest_door_pos = None
        min_dist = float('inf')
        
        for door_vid, door_pos_tuple in door_vids_and_pos:
            try:
                path_len_matrix = self.igraph.shortest_paths(source=source_vid, target=door_vid, weights=None, mode='all')
                if path_len_matrix and path_len_matrix[0]:
                    dist = path_len_matrix[0][0]
                    if dist != float('inf') and dist < min_dist:
                        min_dist = dist
                        closest_door_pos = door_pos_tuple
            except Exception as e:
                continue
        
        if closest_door_pos is None:
            self.logger.warning(f"No reachable door found from {agent_start_pos}")
        else:
            self.logger.debug(f"Closest door to {agent_start_pos} is {closest_door_pos} at distance {min_dist}")
        
        return closest_door_pos


def get_shortest_paths(igraph_instance, source_vid: int, target_vid: int, vid_to_node_map: dict):
    """
    Finds all shortest paths between source and target in an igraph graph.
    """
    vid_paths = igraph_instance.get_all_shortest_paths(source_vid, to=target_vid, weights=None, mode='all')
    coord_paths = []
    for vid_path in vid_paths:
        coord_path = [vid_to_node_map[vid] for vid in vid_path]
        coord_paths.append(coord_path)
    return coord_paths


def get_simple_paths(igraph_instance, source_vid: int, target_vid: int, cutoff: int, vid_to_node_map: dict):
    """
    Finds all simple paths up to cutoff length using igraph.
    """
    vid_paths = igraph_instance.get_all_simple_paths(source_vid, to=target_vid, cutoff=cutoff, mode='all')
    coord_paths = []
    for vid_path in vid_paths:
        coord_path = [vid_to_node_map[vid] for vid in vid_path]
        coord_paths.append(coord_path)
    return coord_paths


def get_shortest_path_length(igraph_instance, source_vid: int, target_vid: int) -> Optional[int]:
    """
    Get shortest path length between two vertices in an igraph graph.
    """
    path_matrix = igraph_instance.shortest_paths(source=source_vid, target=target_vid)
    return int(path_matrix[0][0]) if path_matrix and path_matrix[0] and path_matrix[0][0] != float('inf') else None 
