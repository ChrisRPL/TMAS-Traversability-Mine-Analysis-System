"""Semantic segmentation models for terrain analysis.

This module provides terrain segmentation models for the TMAS system,
enabling pixel-wise classification of terrain types for traversability
cost map generation.
"""

from .terrain_segmenter import TerrainSegmenter

__all__ = ["TerrainSegmenter"]
