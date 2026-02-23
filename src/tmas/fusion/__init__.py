"""Multi-modal fusion modules for TMAS.

This module provides fusion mechanisms for combining RGB and thermal
imaging modalities for enhanced mine detection performance.
"""

from .cross_attention import CrossAttentionFusion

__all__ = ["CrossAttentionFusion"]
