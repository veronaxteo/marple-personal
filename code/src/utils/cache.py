"""
Caching utilities for simulation data.

Provides caching functionality for path sequences and other simulation data
to improve performance by avoiding recomputation.
"""

import logging
import os
import pickle
from typing import Tuple, Optional, Any


class PathSequenceCache:
    """
    This class handles caching and loading of simple path sequences for World objects.
    """
    def __init__(self, log_dir_base: str):
        self.log_dir_base = log_dir_base
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _load_from_cache(self, pickle_path: str, evidence_type_str: str, trial_name: str) -> Optional[Tuple]:
        """
        Try to load cached path sequences.
        """
        if not os.path.exists(pickle_path):
            return None
            
        try:
            with open(pickle_path, 'rb') as f:
                cached_data = pickle.load(f)
                
            if not (isinstance(cached_data, dict) and 'A' in cached_data and 'B' in cached_data):
                self.logger.warning(f"Cached {evidence_type_str} path data is not in expected dict format for {trial_name}. Recomputing.")
                return None
                
            paths_A_tuple = cached_data['A']
            paths_B_tuple = cached_data['B']
            
            # Validate 4-tuple structure
            if not (isinstance(paths_A_tuple, tuple) and len(paths_A_tuple) == 4 and
                   isinstance(paths_B_tuple, tuple) and len(paths_B_tuple) == 4):
                self.logger.warning(f"Cached {evidence_type_str} path tuples have incorrect structure for {trial_name}. Recomputing.")
                return None
                
            self.logger.info(f"Loaded cached {evidence_type_str} simple path sequences for {trial_name} from {pickle_path}.")
            
            # Log path counts for debugging
            if evidence_type_str == 'visual':
                self.logger.info(f"  A: P1:{len(paths_A_tuple[0])}, P2:{len(paths_A_tuple[1])}, P3:{len(paths_A_tuple[2])}")
                self.logger.info(f"  B: P1:{len(paths_B_tuple[0])}, P2:{len(paths_B_tuple[1])}, P3:{len(paths_B_tuple[2])}")
            elif evidence_type_str == 'audio':
                self.logger.info(f"  A: P1(S->F):{len(paths_A_tuple[0])}, P_FS(F->S):{len(paths_A_tuple[3])}")
                self.logger.info(f"  B: P1(S->F):{len(paths_B_tuple[0])}, P_FS(F->S):{len(paths_B_tuple[3])}")
                
            return paths_A_tuple, paths_B_tuple
            
        except Exception as e:
            self.logger.warning(f"Error loading cached {evidence_type_str} paths for {trial_name}: {e}. Recomputing.")
            return None
    
    def _save_to_cache(self, pickle_path: str, paths_A_tuple: Tuple, paths_B_tuple: Tuple, 
                      evidence_type_str: str, trial_name: str) -> None:
        """
        Save computed path sequences to cache.
        """
        try:
            os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
            cached_data = {'A': paths_A_tuple, 'B': paths_B_tuple}
            with open(pickle_path, 'wb') as f:
                pickle.dump(cached_data, f)
            self.logger.info(f"Cached {evidence_type_str} simple path sequences for {trial_name} to {pickle_path}")
        except Exception as e:
            self.logger.error(f"Failed to cache {evidence_type_str} paths for {trial_name}: {e}")


class SimulationDataCache:
    """
    General caching utility for simulation data including results, models, and temporary computations.
    """
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        os.makedirs(cache_dir, exist_ok=True)
    
    def save(self, key: str, data: Any, subdir: str = None) -> bool:
        """
        Save data to cache with the given key.
        """
        try:
            cache_path = os.path.join(self.cache_dir, subdir or "", f"{key}.pkl")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            
            self.logger.debug(f"Cached data with key '{key}' to {cache_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cache data with key '{key}': {e}")
            return False
    
    def load(self, key: str, subdir: str = None) -> Optional[Any]:
        """
        Load data from cache with the given key.
        """
        try:
            cache_path = os.path.join(self.cache_dir, subdir or "", f"{key}.pkl")
            
            if not os.path.exists(cache_path):
                return None
            
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            self.logger.debug(f"Loaded cached data with key '{key}' from {cache_path}")
            return data
        except Exception as e:
            self.logger.warning(f"Failed to load cached data with key '{key}': {e}")
            return None
    
    def exists(self, key: str, subdir: str = None) -> bool:
        """Check if cached data exists for the given key."""
        cache_path = os.path.join(self.cache_dir, subdir or "", f"{key}.pkl")
        return os.path.exists(cache_path)
    
    def clear(self, key: str = None, subdir: str = None) -> bool:
        """Clear cached data."""
        try:
            if key:
                # Clear specific key
                cache_path = os.path.join(self.cache_dir, subdir or "", f"{key}.pkl")
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                    self.logger.info(f"Cleared cache for key '{key}'")
            else:
                # Clear directory
                clear_dir = os.path.join(self.cache_dir, subdir or "")
                if os.path.exists(clear_dir):
                    import shutil
                    shutil.rmtree(clear_dir)
                    os.makedirs(clear_dir, exist_ok=True)
                    self.logger.info(f"Cleared cache directory: {clear_dir}")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            return False 