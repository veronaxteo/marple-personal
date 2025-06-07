from abc import ABC
import json
import logging
import os
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from .math_utils import compute_all_graph_neighbors, normalized_slider_prediction, smooth_likelihoods
from .io_utils import ensure_serializable
from src.core.evidence.visual import get_visual_evidence_likelihood
from src.core.evidence.audio import generate_ground_truth_audio_sequences, single_segment_audio_likelihood
from src.cfg import SimulationConfig, DetectiveTaskConfig


@dataclass
class EvidenceData:
    """Container for evidence-specific data"""
    visual_likelihoods_A: Optional[Dict] = None
    visual_likelihoods_B: Optional[Dict] = None
    audio_sequences_A: Optional[List] = None
    audio_sequences_B: Optional[List] = None
    coordinates: List[Tuple[int, int]] = None
    sequences: List[List] = None
    likelihoods: List[float] = None
    metadata: Dict[str, Any] = None


@dataclass
class PredictionResult:
    """Container for detective prediction results"""
    predictions: Dict[str, float]
    model_output_A: Any
    model_output_B: Any
    prediction_data_for_json: Optional[List] = None


class EvidenceProcessor(ABC):
    """Abstract base class for evidence processors"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def compute_detective_predictions(self, task: DetectiveTaskConfig) -> PredictionResult:
        """Compute detective predictions using task configuration"""
        raise NotImplementedError


class VisualEvidenceProcessor(EvidenceProcessor):
    """Handles visual evidence processing and predictions."""

    def __init__(self, config: SimulationConfig):
        super().__init__()
        self.config = config
    
    def compute_detective_predictions(self, task: DetectiveTaskConfig) -> PredictionResult:
        """Compute visual evidence predictions using new config system"""
        
        self.logger.info(f"Computing VISUAL detective predictions for {task.agent_type_being_simulated} agents")
        
        possible_crumb_coords = task.world.get_valid_kitchen_crumb_coords()
        if not possible_crumb_coords:
            self.logger.warning("No valid crumb coordinates found")
            return PredictionResult({}, {}, {}, [])
        
        # Access data from the dictionary of lists
        agent_A_data = task.sampled_data.get('A', {})
        agent_B_data = task.sampled_data.get('B', {})
        
        agent_A_sequences = agent_A_data.get('full_sequences', [])
        agent_A_middle_sequences = agent_A_data.get('middle_sequences', [])
        agent_A_plant_spots = agent_A_data.get('chosen_plant_spots', [])
        
        agent_B_sequences = agent_B_data.get('full_sequences', [])
        agent_B_middle_sequences = agent_B_data.get('middle_sequences', [])
        agent_B_plant_spots = agent_B_data.get('chosen_plant_spots', [])
        
        raw_likelihood_map_A = {}
        raw_likelihood_map_B = {}
        
        for crumb_coord in possible_crumb_coords:
            # Agent A likelihood
            likelihood_A = get_visual_evidence_likelihood(
                crumb_coord,
                agent_A_sequences,
                agent_A_middle_sequences,
                task.world,
                task.agent_type_being_simulated,
                agent_A_plant_spots,
                task.config.evidence
            )
            raw_likelihood_map_A[crumb_coord] = likelihood_A
            
            # Agent B likelihood  
            likelihood_B = get_visual_evidence_likelihood(
                crumb_coord,
                agent_B_sequences,
                agent_B_middle_sequences,
                task.world,
                task.agent_type_being_simulated,
                agent_B_plant_spots,
                task.config.evidence
            )
            raw_likelihood_map_B[crumb_coord] = likelihood_B
        
        # Apply smoothing for sophisticated agents
        final_likelihood_map_A = raw_likelihood_map_A
        final_likelihood_map_B = raw_likelihood_map_B
        
        if task.agent_type_being_simulated == 'sophisticated':
            detective_sigma = task.config.evidence.sophisticated_detective_sigma
            if detective_sigma > 0:
                self.logger.info(f"Smoothing visual likelihood maps (sigma={detective_sigma})")
                if raw_likelihood_map_A and raw_likelihood_map_B:
                    sigma_steps = max(1, int(detective_sigma))
                    neighbors = compute_all_graph_neighbors(task.world, list(raw_likelihood_map_A.keys()))
                    final_likelihood_map_A = smooth_likelihoods(raw_likelihood_map_A, sigma_steps, neighbors)
                    final_likelihood_map_B = smooth_likelihoods(raw_likelihood_map_B, sigma_steps, neighbors)
                    
                    try:
                        # Only attempt to plot if a logging directory is provided
                        if task.param_log_dir:
                            from src.analysis.plot import plot_smoothing_comparison
                            plot_smoothing_comparison(task.trial_name, task.param_log_dir, raw_likelihood_map_A, raw_likelihood_map_B, task.world, detective_sigma)
                    except ImportError:
                        self.logger.warning("Could not import plot_smoothing_comparison for debugging plots")
        
        prediction_data = []
        for crumb_coord in possible_crumb_coords:
            likelihood_A = final_likelihood_map_A.get(crumb_coord, 0)
            likelihood_B = final_likelihood_map_B.get(crumb_coord, 0)
            prediction = normalized_slider_prediction(likelihood_A, likelihood_B)
            
            prediction_data.append({
                'crumb_coord': crumb_coord,
                'likelihood_A': likelihood_A,
                'likelihood_B': likelihood_B,
                'prediction': prediction
            })
        
        predictions = {
            f"crumb_{coord[0]}_{coord[1]}": normalized_slider_prediction(
                final_likelihood_map_A[coord], final_likelihood_map_B[coord]
            )
            for coord in possible_crumb_coords
        }
        
        if task.param_log_dir:
            save_detective_predictions(prediction_data, task)
        
        return PredictionResult(
            predictions=predictions,
            model_output_A=final_likelihood_map_A,
            model_output_B=final_likelihood_map_B,
            prediction_data_for_json=prediction_data
        )


class AudioEvidenceProcessor(EvidenceProcessor):
    """Handles audio evidence processing and predictions."""
    def __init__(self, config: SimulationConfig):
        super().__init__()
        self.config = config
    
    def compute_detective_predictions(self, task: DetectiveTaskConfig) -> PredictionResult:
        """Compute audio evidence predictions using new config system"""
        self.logger.info(f"Computing AUDIO detective predictions for {task.agent_type_being_simulated} agents")
        
        gt_audio_sequences = generate_ground_truth_audio_sequences(task.world, task.config)
        if not gt_audio_sequences:
            self.logger.warning("No ground truth audio sequences generated")
            return PredictionResult({}, ([], []), ([], []), [])
        
        # Access data from the dictionary of lists
        agent_A_data = task.sampled_data.get('A', {})
        agent_B_data = task.sampled_data.get('B', {})
        agent_A_audio_sequences = agent_A_data.get('audio_sequences', [])
        agent_B_audio_sequences = agent_B_data.get('audio_sequences', [])

        
        agent_A_to_steps, agent_A_from_steps = self._extract_step_lengths(agent_A_audio_sequences)
        agent_B_to_steps, agent_B_from_steps = self._extract_step_lengths(agent_B_audio_sequences)
        
        predictions = {}
        prediction_data = []
        
        for i, gt_sequence in enumerate(gt_audio_sequences):
            likelihood_A = self._calculate_audio_likelihood(gt_sequence, agent_A_audio_sequences, task.config)
            likelihood_B = self._calculate_audio_likelihood(gt_sequence, agent_B_audio_sequences, task.config)
            
            prediction = normalized_slider_prediction(likelihood_A, likelihood_B)
            predictions[f"audio_gt_{i}"] = prediction
            
            prediction_data.append({
                'gt_sequence': gt_sequence,
                'likelihood_A': likelihood_A,
                'likelihood_B': likelihood_B,
                'prediction': prediction
            })
        
        if task.param_log_dir:
            save_detective_predictions(prediction_data, task)
        
        return PredictionResult(
            predictions=predictions,
            model_output_A=(agent_A_to_steps, agent_A_from_steps),
            model_output_B=(agent_B_to_steps, agent_B_from_steps),
            prediction_data_for_json=prediction_data
        )
    
    def _extract_step_lengths(self, audio_sequences: List) -> Tuple[List[int], List[int]]:
        """Extract to_fridge and from_fridge step lengths from audio sequences"""
        to_steps = []
        from_steps = []
        
        for seq in audio_sequences:
            if seq and len(seq) >= 5:
                to_steps.append(seq[0])
                from_steps.append(seq[4])
        
        return to_steps, from_steps
    
    def _calculate_audio_likelihood(self, gt_sequence: List, agent_sequences: List, config: SimulationConfig) -> float:
        """Calculate likelihood of ground truth sequence given agent's sequences"""
        gt_steps_to = gt_sequence[0]
        gt_steps_from = gt_sequence[4]
        
        total_likelihood = 0.0
        valid_sequences = 0
        sigma_factor = config.evidence.audio_similarity_sigma
        
        for agent_seq in agent_sequences:            
            agent_steps_to = agent_seq[0]
            agent_steps_from = agent_seq[4]
            
            lik_to = single_segment_audio_likelihood(gt_steps_to, agent_steps_to, sigma_factor)
            lik_from = single_segment_audio_likelihood(gt_steps_from, agent_steps_from, sigma_factor)
            
            sequence_likelihood = lik_to * lik_from
            total_likelihood += sequence_likelihood
            valid_sequences += 1
        
        return total_likelihood / valid_sequences if valid_sequences > 0 else 0.0


class MultimodalEvidenceProcessor(EvidenceProcessor):
    """Processes combined visual and audio evidence."""
    def __init__(self, config: SimulationConfig):
        super().__init__()
        self.config = config
        self.visual_processor = VisualEvidenceProcessor(config)
        self.audio_processor = AudioEvidenceProcessor(config)

    def compute_detective_predictions(self, task: DetectiveTaskConfig) -> PredictionResult:
        self.logger.info(f"Computing MULTIMODAL detective predictions for {task.agent_type_being_simulated} agents")

        visual_task_info = {
            'world': task.world,
            'sampled_data': task.sampled_data,
            'agent_type_being_simulated': task.agent_type_being_simulated,
            'trial_name': task.trial_name,
            'param_log_dir': None,
            'config': task.config
        }
        visual_task = DetectiveTaskConfig(**visual_task_info)
        
        visual_result = self.visual_processor.compute_detective_predictions(visual_task)
        visual_model_A = visual_result.model_output_A
        visual_model_B = visual_result.model_output_B
        
        gt_audio_sequences = generate_ground_truth_audio_sequences(task.world, task.config)
        
        agent_A_data = task.sampled_data.get('A', {})
        agent_B_data = task.sampled_data.get('B', {})
        agent_A_audio_sequences = agent_A_data.get('audio_sequences', [])
        agent_B_audio_sequences = agent_B_data.get('audio_sequences', [])

        predictions = {}
        prediction_data = []
        
        possible_crumb_coords = list(visual_model_A.keys())

        for crumb_coord in possible_crumb_coords:
            for gt_audio_seq in gt_audio_sequences:
                lik_A_visual = visual_model_A.get(crumb_coord, 0)
                lik_B_visual = visual_model_B.get(crumb_coord, 0)
                
                lik_A_audio = self.audio_processor._calculate_audio_likelihood(gt_audio_seq, agent_A_audio_sequences, task.config)
                lik_B_audio = self.audio_processor._calculate_audio_likelihood(gt_audio_seq, agent_B_audio_sequences, task.config)

                total_lik_A = lik_A_visual * lik_A_audio
                total_lik_B = lik_B_visual * lik_B_audio

                prediction = normalized_slider_prediction(total_lik_A, total_lik_B)
                
                pred_key = f"crumb_{crumb_coord[0]}_{crumb_coord[1]}_audio_{gt_audio_seq[0]}_{gt_audio_seq[4]}"
                predictions[pred_key] = prediction
                
                prediction_data.append({
                    'gt_sequence': gt_audio_seq,
                    'crumb_coord': crumb_coord,
                    'likelihood_A': total_lik_A,
                    'likelihood_B': total_lik_B,
                    'prediction': prediction,
                    'lik_A_visual': lik_A_visual,
                    'lik_B_visual': lik_B_visual,
                    'lik_A_audio': lik_A_audio,
                    'lik_B_audio': lik_B_audio
                })
        
        agent_A_to_steps, agent_A_from_steps = self.audio_processor._extract_step_lengths(agent_A_audio_sequences)
        agent_B_to_steps, agent_B_from_steps = self.audio_processor._extract_step_lengths(agent_B_audio_sequences)
        
        model_A = {'visual': visual_model_A, 'audio': (agent_A_to_steps, agent_A_from_steps)}
        model_B = {'visual': visual_model_B, 'audio': (agent_B_to_steps, agent_B_from_steps)}
        
        if task.param_log_dir:
            save_detective_predictions(prediction_data, task)

        return PredictionResult(
            predictions=predictions,
            model_output_A=model_A,
            model_output_B=model_B,
            prediction_data_for_json=prediction_data
        )
    

def get_evidence_processor(config) -> EvidenceProcessor:
    """Factory function to get the appropriate evidence processor."""
    evidence_type = config.evidence.evidence_type
    if evidence_type == 'visual':
        return VisualEvidenceProcessor(config)
    elif evidence_type == 'audio':
        return AudioEvidenceProcessor(config)
    elif evidence_type == 'multimodal':
        return MultimodalEvidenceProcessor(config)
    else:
        raise ValueError(f"Unsupported evidence type: {evidence_type}")


def save_detective_predictions(prediction_data: List[Dict], task: DetectiveTaskConfig) -> None:
    """Save detective prediction results to JSON file."""
    if not task.param_log_dir or not prediction_data:
        return
        
    logger = logging.getLogger(__name__)
    
    evidence_type = task.config.evidence.evidence_type
    filename = f"{task.trial_name}_{task.agent_type_being_simulated}_{evidence_type}_predictions.json"
    filepath = os.path.join(task.param_log_dir, filename)
    
    try:
        output_data = {
            'trial_name': task.trial_name,
            'agent_type_being_simulated': task.agent_type_being_simulated,
            'evidence_type': evidence_type,
            'predictions': ensure_serializable(prediction_data)
        }
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=4)
        
        logger.info(f"Saved {len(prediction_data)} detective predictions to {filename}")
        
    except Exception as e:
        logger.error(f"Error saving detective predictions to {filename}: {e}", exc_info=True)
