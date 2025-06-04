"""
Analysis module for simulation visualization and evaluation.

This module provides:
- plotting: Core plotting functionality for heatmaps and visualizations
- evaluation: Analysis functions for path lengths and prediction evaluation
- evaluate: Legacy evaluation module for backwards compatibility
"""

from .plotting import (
    plot_smoothing_comparison,
    plot_suspect_paths_heatmap,
    create_summary_plots,
    plot_detective_predictions_heatmap,
    plot_suspect_crumb_planting_heatmap,
    create_simulation_plots
)

from .evaluation import (
    extract_params_from_path,
    get_evidence_type_from_metadata,
    calculate_avg_path_lengths,
    analyze_gt_audio_lengths_vs_predictions,
    consolidate_evaluation_results
)

# Legacy compatibility - import evaluate module
from . import evaluate

__all__ = [
    # Plotting functions
    'plot_smoothing_comparison',
    'plot_suspect_paths_heatmap',
    'create_summary_plots',
    'plot_detective_predictions_heatmap',
    'plot_suspect_crumb_planting_heatmap',
    'create_simulation_plots',
    
    # Evaluation functions  
    'extract_params_from_path',
    'get_evidence_type_from_metadata',
    'calculate_avg_path_lengths',
    'analyze_gt_audio_lengths_vs_predictions',
    'consolidate_evaluation_results',
    
    # Legacy modules
    'evaluate'
] 