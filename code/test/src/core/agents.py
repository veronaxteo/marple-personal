import json
import logging
import os

from src.core.world import World
from src.utils.io_utils import ensure_serializable
from src.configs import EvidenceConfig, SimulationConfig
from src.utils.evidence_utils import create_cfg


class Agent:
    """
    Base agent class for suspects and detectives
    """
    def __init__(self, agent_id: str, data_type: str, cfg: SimulationConfig):
        self.id = agent_id
        self.data_type = data_type
        self.cfg = cfg
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{self.id}]")


class Suspect(Agent):
    """
    Suspect agent that generates paths based on utility functions
    """
    def simulate_suspect(self, world: World, paths_A: list, paths_B: list, 
                        agent_type: str, num_sample_paths: int) -> dict:
        """Generate suspect paths for both agents"""
        self.logger.info(f"Simulating {agent_type} suspect paths using {self.cfg.evidence.type} evidence")
        
        # For now, return dummy data structure to avoid breaking the simulation
        # TODO: Implement proper path sampling when PathSampler is updated
        
        agent_A_data = {
            'full_sequences': [],
            'middle_sequences': [],
            'chosen_plant_spots': [],
            'audio_sequences': [],
            'to_fridge_sequences': [],
            'full_sequence_lengths': [],
            'to_fridge_sequence_lengths': [],
            'middle_sequence_lengths': []
        }
        
        agent_B_data = {
            'full_sequences': [],
            'middle_sequences': [],
            'chosen_plant_spots': [],
            'audio_sequences': [],
            'to_fridge_sequences': [],
            'full_sequence_lengths': [],
            'to_fridge_sequence_lengths': [],
            'middle_sequence_lengths': []
        }
        
        return {'A': agent_A_data, 'B': agent_B_data}


class Detective(Agent):
    """
    Detective agent that makes predictions about suspects based on evidence
    """
    def __init__(self, agent_id: str, data_type: str, cfg: SimulationConfig):
        super().__init__(agent_id, data_type, cfg)
        self.evidence_processor = create_cfg(self.cfg.evidence.type)
    
    def simulate_detective(self, world: World, trial_name: str, sampled_data: dict, 
                         agent_type_being_simulated: str, param_log_dir: str,
                         source_data_type: str = None, mismatched_run: bool = None) -> tuple:
        """
        Generate detective predictions about suspects
        
        Returns:
            tuple: (predictions_dict, agent_A_model_output, agent_B_model_output)
        """
        self.logger.info(f"Simulating detective modeling {agent_type_being_simulated} suspects "
                        f"using {self.cfg.evidence.type} evidence")

        # Create detective task configuration (simplified for now)
        # TODO: Implement proper task configuration when evidence_utils is updated
        
        # For now, return dummy data to avoid breaking the simulation
        predictions = {}
        model_output_A = {}
        model_output_B = {}
        
        return predictions, model_output_A, model_output_B

    def _save_results_to_json(self, prediction_data: list, param_log_dir: str, trial_name: str, 
                            agent_type_simulated: str, source_data_type: str = None, 
                            mismatched_run: bool = None):
        """Save prediction results to JSON file"""
        # Determine filename based on simulation type
        condition_name = agent_type_simulated
        if mismatched_run and source_data_type:
            condition_name = f"{agent_type_simulated}_as_{source_data_type}"
        
        filename = f"{trial_name}_{condition_name}_{self.cfg.evidence.type}_predictions.json"
        filepath = os.path.join(param_log_dir, filename)
        
        try:
            serializable_data = ensure_serializable(prediction_data)
            with open(filepath, 'w') as f:
                json.dump(serializable_data, f, indent=4)
            self.logger.info(f"Saved predictions to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save predictions to {filepath}: {e}")
