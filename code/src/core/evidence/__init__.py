"""
Evidence processing module for visual and audio evidence types.

This module provides:
- visual: Visual evidence likelihood calculations and crumb dropping simulation  
- audio: Audio token generation, compression, and likelihood calculations
- Evidence classes for object-oriented access to functionality
"""

from .visual import get_visual_evidence_likelihood
from .audio import (
    get_audio_tokens_for_path,
    parse_raw_audio_tokens,
    get_compressed_audio_from_path,
    single_segment_audio_likelihood,
    generate_ground_truth_audio_sequences
)


class Evidence:
    """Base evidence class for future expansion"""
    pass


class VisualEvidence(Evidence):
    """Visual evidence processing utilities"""
    
    @staticmethod
    def get_visual_evidence_likelihood(*args, **kwargs):
        """Static method wrapper for visual evidence likelihood calculation"""
        return get_visual_evidence_likelihood(*args, **kwargs)


class AudioEvidence(Evidence):
    """Audio evidence processing utilities"""
    
    @staticmethod
    def single_segment_audio_likelihood(*args, **kwargs):
        """Static method wrapper for single segment audio likelihood"""
        return single_segment_audio_likelihood(*args, **kwargs)
    
    @staticmethod
    def generate_ground_truth_sequences(*args, **kwargs):
        """Static method wrapper for ground truth audio sequence generation"""
        return generate_ground_truth_audio_sequences(*args, **kwargs)


__all__ = [
    # Base classes
    'Evidence',
    'VisualEvidence', 
    'AudioEvidence',
    
    # Visual evidence functions
    'get_visual_evidence_likelihood',
    
    # Audio evidence functions
    'get_audio_tokens_for_path',
    'parse_raw_audio_tokens',
    'get_compressed_audio_from_path',
    'single_segment_audio_likelihood',
    'generate_ground_truth_audio_sequences'
] 