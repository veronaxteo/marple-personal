"""
Utility modules for simulation functionality.

This module provides:
- evidence_utils: Evidence processing and detective prediction utilities
- io_utils: Input/output utilities for file handling and serialization
- math_utils: Mathematical utilities for path analysis and likelihood calculations
- cache: Caching utilities for simulation data and path sequences
"""

from .evidence_utils import (
    EvidenceProcessor,
    VisualEvidenceProcessor, 
    AudioEvidenceProcessor,
    MultimodalEvidenceProcessor,
    create_evidence_processor,
    EvidenceData,
    PredictionResult
)

from .io_utils import (
    ensure_serializable,
    get_json_files,
    create_param_dir,
    save_sampled_paths_to_csv,
    save_grid_to_json
)

from .math_utils import (
    normalized_slider_prediction,
    smooth_likelihood_grid,
    smooth_likelihood_grid_connectivity_aware
)

from .cache import (
    PathSequenceCache,
    SimulationDataCache
)

__all__ = [
    # Evidence processing
    'EvidenceProcessor',
    'VisualEvidenceProcessor',
    'AudioEvidenceProcessor', 
    'MultimodalEvidenceProcessor',
    'create_evidence_processor',
    'EvidenceData',
    'PredictionResult',
    
    # I/O utilities
    'ensure_serializable',
    'get_json_files',
    'create_param_dir',
    'save_sampled_paths_to_csv',
    'save_grid_to_json',
    
    # Math utilities
    'normalized_slider_prediction',
    'smooth_likelihood_grid',
    'smooth_likelihood_grid_connectivity_aware',
    
    # Caching
    'PathSequenceCache',
    'SimulationDataCache'
] 