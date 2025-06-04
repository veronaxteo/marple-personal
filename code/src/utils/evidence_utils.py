from abc import ABC
import json
import logging
import os
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from .math_utils import normalized_slider_prediction, smooth_likelihood_grid, smooth_likelihood_grid_connectivity_aware
from .io_utils import ensure_serializable
from src.core.evidence import VisualEvidence, generate_ground_truth_audio_sequences, single_segment_audio_likelihood
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
    """Handles visual evidence processing and predictions"""
    
    def compute_detective_predictions(self, task: DetectiveTaskConfig) -> PredictionResult:
        """Compute visual evidence predictions using new config system"""
        
        self.logger.info(f"Computing VISUAL detective predictions for {task.agent_type_being_simulated} agents")
        
        # Get possible crumb locations
        possible_crumb_coords = task.world.get_valid_kitchen_crumb_coords_world()
        if not possible_crumb_coords:
            self.logger.warning("No valid crumb coordinates found")
            return PredictionResult({}, {}, {}, [])
        
        # Debug: show possible crumb coordinates
        if task.agent_type_being_simulated == 'sophisticated':
            self.logger.info(f"DEBUG: Possible crumb coordinates ({len(possible_crumb_coords)}): {sorted(possible_crumb_coords)}")
        
        # Calculate likelihoods for each possible crumb location
        agent_A_data = task.sampled_data.get('A', {})
        agent_B_data = task.sampled_data.get('B', {})
        
        agent_A_sequences = agent_A_data.get('full_sequences', [])
        agent_A_middle_sequences = agent_A_data.get('middle_sequences', [])
        agent_A_plant_spots = agent_A_data.get('chosen_plant_spots', [])
        
        agent_B_sequences = agent_B_data.get('full_sequences', [])
        agent_B_middle_sequences = agent_B_data.get('middle_sequences', [])
        agent_B_plant_spots = agent_B_data.get('chosen_plant_spots', [])
        
        if not agent_A_sequences or not agent_B_sequences:
            self.logger.warning("Missing agent sequences for visual evidence calculation")
            return PredictionResult({}, {}, {}, [])
        
        # Calculate raw likelihoods for each crumb coordinate
        raw_likelihood_map_A = {}
        raw_likelihood_map_B = {}
        
        for crumb_coord in possible_crumb_coords:
            # Agent A likelihood
            likelihood_A = VisualEvidence.get_visual_evidence_likelihood(
                crumb_coord,
                agent_A_sequences,
                agent_A_middle_sequences,
                task.world,
                task.agent_type_being_simulated,
                agent_A_plant_spots
            )
            raw_likelihood_map_A[crumb_coord] = likelihood_A
            
            # Agent B likelihood  
            likelihood_B = VisualEvidence.get_visual_evidence_likelihood(
                crumb_coord,
                agent_B_sequences,
                agent_B_middle_sequences,
                task.world,
                task.agent_type_being_simulated,
                agent_B_plant_spots
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
                    # 2D grid-based smoothing without furniture awareness
                    # final_likelihood_map_A = smooth_likelihood_grid(raw_likelihood_map_A, task.world, detective_sigma)
                    # final_likelihood_map_B = smooth_likelihood_grid(raw_likelihood_map_B, task.world, detective_sigma)
                    
                    # Connectivity-aware smoothing
                    sigma_steps = max(1, int(detective_sigma))  # Convert sigma to discrete steps
                    final_likelihood_map_A = smooth_likelihood_grid_connectivity_aware(raw_likelihood_map_A, task.world, sigma_steps)
                    final_likelihood_map_B = smooth_likelihood_grid_connectivity_aware(raw_likelihood_map_B, task.world, sigma_steps)
                    
                    # Generate smoothing comparison plots for debugging
                    try:
                        from src.analysis.plotting import plot_smoothing_comparison
                        plot_smoothing_comparison(task.trial_name, task.param_log_dir, raw_likelihood_map_A, raw_likelihood_map_B, task.world, detective_sigma)
                    except ImportError:
                        self.logger.warning("Could not import plot_smoothing_comparison for debugging plots")
        
        # Calculate predictions using final (i.e. possibly smoothed) likelihoods
        prediction_data = []
        for crumb_coord in possible_crumb_coords:
            likelihood_A = final_likelihood_map_A[crumb_coord]
            likelihood_B = final_likelihood_map_B[crumb_coord]
            prediction = normalized_slider_prediction(likelihood_A, likelihood_B)
            
            prediction_data.append({
                'crumb_coord': crumb_coord,
                'likelihood_A': likelihood_A,
                'likelihood_B': likelihood_B,
                'prediction': prediction
            })
        
        # Create prediction dictionary
        predictions = {
            f"crumb_{coord[0]}_{coord[1]}": normalized_slider_prediction(
                final_likelihood_map_A[coord], final_likelihood_map_B[coord]
            )
            for coord in possible_crumb_coords
        }
        
        # Save predictions to file
        save_detective_predictions(prediction_data, task)
        
        return PredictionResult(
            predictions=predictions,
            model_output_A=final_likelihood_map_A,
            model_output_B=final_likelihood_map_B,
            prediction_data_for_json=prediction_data
        )


class AudioEvidenceProcessor(EvidenceProcessor):
    """Handles audio evidence processing and predictions"""
    
    def compute_detective_predictions(self, task: DetectiveTaskConfig) -> PredictionResult:
        """Compute audio evidence predictions using new config system"""
        
        self.logger.info(f"Computing AUDIO detective predictions for {task.agent_type_being_simulated} agents")
        
        # Generate ground truth audio sequences  
        gt_audio_sequences = generate_ground_truth_audio_sequences(task.world, task.config)
        if not gt_audio_sequences:
            self.logger.warning("No ground truth audio sequences generated")
            return PredictionResult({}, ([], []), ([], []), [])
        
        # Extract agent data
        agent_A_data = task.sampled_data.get('A', {})
        agent_B_data = task.sampled_data.get('B', {})
        
        agent_A_audio_sequences = agent_A_data.get('audio_sequences', [])
        agent_B_audio_sequences = agent_B_data.get('audio_sequences', [])
        
        if not agent_A_audio_sequences or not agent_B_audio_sequences:
            self.logger.warning("Missing audio sequences for agents")
            return PredictionResult({}, ([], []), ([], []), [])
        
        # Calculate step lengths for model output
        agent_A_to_steps, agent_A_from_steps = self._extract_step_lengths(agent_A_audio_sequences)
        agent_B_to_steps, agent_B_from_steps = self._extract_step_lengths(agent_B_audio_sequences)
        
        # Calculate predictions for each ground truth sequence
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
        
        # Save predictions to file
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
        gt_fridge_events = gt_sequence[1:4]
        
        total_likelihood = 0.0
        valid_sequences = 0
        sigma_factor = config.evidence.audio_similarity_sigma
        
        for agent_seq in agent_sequences:
            if not (isinstance(agent_seq, list) and len(agent_seq) == 5 and 
                    isinstance(agent_seq[0], int) and isinstance(agent_seq[4], int)):
                continue
            
            # Check if fridge events match
            if gt_fridge_events != agent_seq[1:4]:
                continue
            
            agent_steps_to = agent_seq[0]
            agent_steps_from = agent_seq[4]
            
            # Calculate likelihood for to_fridge and from_fridge segments separately
            lik_to = single_segment_audio_likelihood(gt_steps_to, agent_steps_to, sigma_factor)
            lik_from = single_segment_audio_likelihood(gt_steps_from, agent_steps_from, sigma_factor)
            
            # Combine segment likelihoods
            sequence_likelihood = lik_to * lik_from
            total_likelihood += sequence_likelihood
            valid_sequences += 1
        
        return total_likelihood / valid_sequences if valid_sequences > 0 else 0.0


class MultimodalEvidenceProcessor(EvidenceProcessor):
    """Handles multimodal (visual + audio) evidence processing"""
    
    def __init__(self):
        super().__init__()
        self.visual_processor = VisualEvidenceProcessor()
        self.audio_processor = AudioEvidenceProcessor()
    
    def compute_detective_predictions(self, task: DetectiveTaskConfig) -> PredictionResult:
        """Compute multimodal predictions by combining visual and audio evidence"""
        
        self.logger.info(f"Computing MULTIMODAL detective predictions for {task.agent_type_being_simulated} agents")
        
        # Create separate tasks for each modality to prevent duplicate saves
        visual_task = DetectiveTaskConfig(
            world=task.world,
            sampled_data=task.sampled_data,
            agent_type_being_simulated=task.agent_type_being_simulated,
            trial_name=task.trial_name,
            param_log_dir=None,  # Don't save individual predictions for multimodal
            config=task.config
        )
        
        audio_task = DetectiveTaskConfig(
            world=task.world,
            sampled_data=task.sampled_data,
            agent_type_being_simulated=task.agent_type_being_simulated,
            trial_name=task.trial_name,
            param_log_dir=None,  # Don't save individual predictions for multimodal
            config=task.config
        )
        
        # Get predictions from both modalities (without saving)
        visual_result = self.visual_processor.compute_detective_predictions(visual_task)
        audio_result = self.audio_processor.compute_detective_predictions(audio_task)
        
        # Combine predictions using weighted average
        visual_weight = task.config.evidence.visual_weight
        audio_weight = 1.0 - visual_weight
        
        combined_predictions = {}
        
        # Combine visual predictions
        for key, visual_pred in visual_result.predictions.items():
            combined_predictions[f"multimodal_{key}"] = visual_weight * visual_pred
        
        # Combine audio predictions
        for key, audio_pred in audio_result.predictions.items():
            if f"multimodal_{key}" in combined_predictions:
                combined_predictions[f"multimodal_{key}"] += audio_weight * audio_pred
            else:
                combined_predictions[f"multimodal_{key}"] = audio_weight * audio_pred
        
        # Combine prediction data
        combined_data = []
        if visual_result.prediction_data_for_json:
            combined_data.extend([{**item, 'modality': 'visual', 'weight': visual_weight} for item in visual_result.prediction_data_for_json])
        if audio_result.prediction_data_for_json:
            combined_data.extend([{**item, 'modality': 'audio', 'weight': audio_weight} for item in audio_result.prediction_data_for_json])
        
        # Add combined predictions to data
        for key, combined_pred in combined_predictions.items():
            combined_data.append({
                'prediction_key': key,
                'combined_prediction': combined_pred,
                'visual_weight': visual_weight,
                'audio_weight': audio_weight
            })
        
        # Save multimodal predictions
        save_detective_predictions(combined_data, task)
        
        return PredictionResult(
            predictions=combined_predictions,
            model_output_A={'visual': visual_result.model_output_A, 'audio': audio_result.model_output_A},
            model_output_B={'visual': visual_result.model_output_B, 'audio': audio_result.model_output_B},
            prediction_data_for_json=combined_data
        )


def create_evidence_processor(evidence_type: str) -> EvidenceProcessor:
    """Factory function to create appropriate evidence processor"""
    if evidence_type == 'visual':
        return VisualEvidenceProcessor()
    elif evidence_type == 'audio':
        return AudioEvidenceProcessor()
    elif evidence_type == 'multimodal':
        return MultimodalEvidenceProcessor()
    else:
        raise ValueError(f"Unknown evidence type: {evidence_type}") 


def save_detective_predictions(prediction_data: List[Dict], task: DetectiveTaskConfig) -> None:
    """Save detective prediction results to JSON file."""
    if not task.param_log_dir or not prediction_data:
        return
        
    logger = logging.getLogger(__name__)
    
    # Create filename based on simulation parameters
    evidence_type = task.config.evidence.evidence_type
    filename = f"{task.trial_name}_{task.agent_type_being_simulated}_{evidence_type}_predictions.json"
    filepath = os.path.join(task.param_log_dir, filename)
    
    try:
        # Add metadata to predictions
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
        logger.error(f"Error saving detective predictions to {filename}: {e}")
        
        # Try to save a minimal version
        try:
            minimal_data = {
                'trial_name': task.trial_name,
                'agent_type_being_simulated': task.agent_type_being_simulated,
                'evidence_type': evidence_type,
                'num_predictions': len(prediction_data),
                'error': str(e)
            }
            
            minimal_filename = filename.replace('.json', '_minimal.json')
            minimal_filepath = os.path.join(task.param_log_dir, minimal_filename)
            
            with open(minimal_filepath, 'w') as f:
                json.dump(minimal_data, f, indent=4)
                
            logger.info(f"Saved minimal prediction summary to {minimal_filename}")
            
        except Exception as e2:
            logger.error(f"Error saving minimal predictions: {e2}") 
    