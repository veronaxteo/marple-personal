import json
import logging
import os

from src.world import World
from src.params import SimulationParams
from src.utils.io_utils import ensure_serializable
from src.utils.evidence_utils import create_evidence_processor
from src.config import SimulationConfig, PathSamplingTask, DetectiveTaskConfig


class Agent:
    """
    Base agent class for suspects and detectives
    """
    def __init__(self, agent_id: str, data_type: str, params: SimulationParams):
        self.id = agent_id
        self.data_type = data_type
        self.params = params
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{self.id}]")


class Suspect(Agent):
    """
    Suspect agent that generates paths based on utility functions
    """
    
    def simulate_suspect(self, world: World, paths_A: list, paths_B: list, 
                        agent_type: str, num_sample_paths: int) -> dict:
        """Generate suspect paths for both agents"""
        self.logger.info(f"Simulating {agent_type} suspect paths using {self.params.evidence} evidence")
        
        # Create simulation config from params
        config = self._create_config_from_params(num_sample_paths)
        
        # Create path sampling tasks
        task_A = PathSamplingTask(
            world=world,
            agent_id='A',
            simple_path_sequences=paths_A,
            config=config,
            agent_type=agent_type
        )
        
        task_B = PathSamplingTask(
            world=world,
            agent_id='B', 
            simple_path_sequences=paths_B,
            config=config,
            agent_type=agent_type
        )
        
        agent_A_data = world.path_sampler.sample_paths(task_A)
        agent_B_data = world.path_sampler.sample_paths(task_B)
        
        return {'A': agent_A_data, 'B': agent_B_data}
    
    def _create_config_from_params(self, num_sample_paths: int) -> SimulationConfig:
        """Convert legacy SimulationParams to new SimulationConfig"""
        if self.params.evidence == 'visual':
            config = SimulationConfig.create_visual_config(
                trial_name="temp",
                cost_weight=self.params.w,
                naive_temp=self.params.n_temp,
                sophisticated_temp=self.params.s_temp,
                max_steps=self.params.max_steps
            )
        elif self.params.evidence == 'audio':
            config = SimulationConfig.create_audio_config(
                trial_name="temp",
                cost_weight=self.params.w,
                naive_temp=self.params.n_temp,
                sophisticated_temp=self.params.s_temp,
                max_steps=self.params.max_steps,
                audio_similarity_sigma=self.params.audio_segment_similarity_sigma
            )
        else:
            raise ValueError(f"Unsupported evidence type: {self.params.evidence}")
        
        # Override sampling numbers
        config.sampling.num_suspect_paths = num_sample_paths
        config.sampling.noisy_planting_sigma = self.params.noisy_planting_sigma
        
        # Transfer sophisticated agent models if available
        if hasattr(self.params, 'naive_A_visual_likelihoods_map'):
            config.evidence.naive_A_visual_likelihoods_map = self.params.naive_A_visual_likelihoods_map
        if hasattr(self.params, 'naive_B_visual_likelihoods_map'):
            config.evidence.naive_B_visual_likelihoods_map = self.params.naive_B_visual_likelihoods_map
        if hasattr(self.params, 'naive_A_to_fridge_steps_model'):
            config.evidence.naive_A_to_fridge_steps_model = self.params.naive_A_to_fridge_steps_model
        if hasattr(self.params, 'naive_A_from_fridge_steps_model'):
            config.evidence.naive_A_from_fridge_steps_model = self.params.naive_A_from_fridge_steps_model
        if hasattr(self.params, 'naive_B_to_fridge_steps_model'):
            config.evidence.naive_B_to_fridge_steps_model = self.params.naive_B_to_fridge_steps_model
        if hasattr(self.params, 'naive_B_from_fridge_steps_model'):
            config.evidence.naive_B_from_fridge_steps_model = self.params.naive_B_from_fridge_steps_model
        
        return config


class Detective(Agent):
    """
    Detective agent that makes predictions about suspects based on evidence
    """
    
    def __init__(self, agent_id: str, data_type: str, params: SimulationParams):
        super().__init__(agent_id, data_type, params)
        self.evidence_processor = create_evidence_processor(self.params.evidence)
    
    def simulate_detective(self, world: World, trial_name: str, sampled_data: dict, 
                         agent_type_being_simulated: str, param_log_dir: str,
                         source_data_type: str = None, mismatched_run: bool = None) -> tuple:
        """
        Generate detective predictions about suspects
        
        Returns:
            tuple: (predictions_dict, agent_A_model_output, agent_B_model_output)
        """
        self.logger.info(f"Simulating detective modeling {agent_type_being_simulated} suspects "
                        f"using {self.params.evidence} evidence")

        # Create config from params
        config = self._create_config_from_params()
        
        # Create detective task configuration
        task = DetectiveTaskConfig(
            world=world,
            trial_name=trial_name,
            sampled_data=sampled_data,
            agent_type_being_simulated=agent_type_being_simulated,
            config=config,
            source_data_type=source_data_type,
            mismatched_run=mismatched_run,
            param_log_dir=param_log_dir
        )
        
        # Compute predictions using evidence processor
        result = self.evidence_processor.compute_detective_predictions(task)

        # Save results to file if requested
        if param_log_dir and result.prediction_data_for_json:
            self._save_results_to_json(
                result.prediction_data_for_json, param_log_dir, trial_name, 
                agent_type_being_simulated, source_data_type, mismatched_run
            )

        return result.predictions, result.model_output_A, result.model_output_B
    
    def _create_config_from_params(self) -> SimulationConfig:
        """Convert legacy SimulationParams to new SimulationConfig"""
        if self.params.evidence == 'visual':
            config = SimulationConfig.create_visual_config(
                trial_name="temp",
                cost_weight=self.params.w,
                naive_temp=self.params.n_temp,
                sophisticated_temp=self.params.s_temp,
                max_steps=self.params.max_steps
            )
            # Visual-specific parameters
            config.evidence.sophisticated_detective_sigma = self.params.soph_detective_sigma
            config.evidence.visual_smoothing_sigma = getattr(self.params, 'soph_suspect_sigma', 0.0)
            
        elif self.params.evidence == 'audio':
            config = SimulationConfig.create_audio_config(
                trial_name="temp",
                cost_weight=self.params.w,
                naive_temp=self.params.n_temp,
                sophisticated_temp=self.params.s_temp,
                max_steps=self.params.max_steps,
                audio_similarity_sigma=self.params.audio_segment_similarity_sigma
            )
            # Audio-specific parameters
            config.evidence.audio_gt_step_size = getattr(self.params, 'audio_gt_step_size', 2)
            
        else:
            raise ValueError(f"Unsupported evidence type: {self.params.evidence}")
        
        # Transfer sophisticated agent models if available  
        if hasattr(self.params, 'naive_A_visual_likelihoods_map'):
            config.evidence.naive_A_visual_likelihoods_map = self.params.naive_A_visual_likelihoods_map
        if hasattr(self.params, 'naive_B_visual_likelihoods_map'):
            config.evidence.naive_B_visual_likelihoods_map = self.params.naive_B_visual_likelihoods_map
        if hasattr(self.params, 'naive_A_to_fridge_steps_model'):
            config.evidence.naive_A_to_fridge_steps_model = self.params.naive_A_to_fridge_steps_model
        if hasattr(self.params, 'naive_A_from_fridge_steps_model'):
            config.evidence.naive_A_from_fridge_steps_model = self.params.naive_A_from_fridge_steps_model
        if hasattr(self.params, 'naive_B_to_fridge_steps_model'):
            config.evidence.naive_B_to_fridge_steps_model = self.params.naive_B_to_fridge_steps_model
        if hasattr(self.params, 'naive_B_from_fridge_steps_model'):
            config.evidence.naive_B_from_fridge_steps_model = self.params.naive_B_from_fridge_steps_model
        
        return config

    def _save_results_to_json(self, prediction_data: list, param_log_dir: str, trial_name: str, 
                            agent_type_simulated: str, source_data_type: str = None, 
                            mismatched_run: bool = None):
        """Save prediction results to JSON file"""
        # Determine filename based on simulation type
        condition_name = agent_type_simulated
        if mismatched_run and source_data_type:
            condition_name = f"{agent_type_simulated}_as_{source_data_type}"
        
        filename = f"{trial_name}_{condition_name}_{self.params.evidence}_predictions.json"
        filepath = os.path.join(param_log_dir, filename)
        
        try:
            serializable_data = ensure_serializable(prediction_data)
            with open(filepath, 'w') as f:
                json.dump(serializable_data, f, indent=4)
            self.logger.info(f"Saved predictions to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save predictions to {filepath}: {e}")
