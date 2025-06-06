"""
Analysis module for simulation visualization.

This module provides:
- plot: Core plotting functionality for heatmaps and visualizations
"""

from .plot import (
    plot_smoothing_comparison,
    plot_suspect_paths_heatmap,
    create_summary_plots,
    plot_detective_predictions_heatmap,
    plot_suspect_crumb_planting_heatmap,
    create_simulation_plots
)


__all__ = [
    # Plotting functions
    'plot_smoothing_comparison',
    'plot_suspect_paths_heatmap',
    'create_summary_plots',
    'plot_detective_predictions_heatmap',
    'plot_suspect_crumb_planting_heatmap',
    'create_simulation_plots'
] 