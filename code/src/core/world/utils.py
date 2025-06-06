import logging
from pathlib import Path
import pickle

from ...cfg import SimulationConfig
from .planning import compute_agent_path_sequences


def load_or_compute_simple_path_sequences(
    world, trial_name: str, 
    config: SimulationConfig, max_steps: int):
    """Load or compute simple path sequences for a trial"""
    logger = logging.getLogger(__name__)
    
    # Determine cache file path
    cache_dir = Path(config.log_dir_base) / 'simple_paths'
    cache_file = f'{trial_name}_simple_paths_{max_steps}.pkl'
    cache_path = cache_dir / cache_file

    # Try to load from cache
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                logger.info(f"Loaded simple path sequences for {trial_name} from cache")
                return cached_data['A'], cached_data['B']
        except Exception as e:
            logger.warning(f"Failed to load cache for {trial_name}: {e}")
                    
    # Compute subgoal simple path sequences using utility function
    logger.info(f"Computing simple path sequences for {trial_name} (max_steps={max_steps})")
    
    paths_A = compute_agent_path_sequences('A', world.world_graph, world.geometry, 
                                         world.start_coords, world.mission, max_steps)
    paths_B = compute_agent_path_sequences('B', world.world_graph, world.geometry, 
                                         world.start_coords, world.mission, max_steps)
    
    # Save results to cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        cache_data = {'A': paths_A, 'B': paths_B}
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        logger.info(f"Cached simple path sequences for {trial_name}")
    except Exception as e:
        logger.error(f"Failed to cache paths for {trial_name}: {e}")

    return paths_A, paths_B 
