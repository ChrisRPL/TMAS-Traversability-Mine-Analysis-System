"""Depth estimation modules for TMAS.

This module provides monocular and stereo depth estimation for
obstacle distance calculation and safety zone classification.
"""

from .monocular_depth import MonocularDepthEstimator, create_depth_estimator

__all__ = ["MonocularDepthEstimator", "create_depth_estimator"]
