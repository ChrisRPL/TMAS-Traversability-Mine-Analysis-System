"""Uncertainty estimation modules for TMAS.

This module provides uncertainty quantification methods for safety-critical
mine detection, enabling the system to know when it doesn't know.
"""

from .evidential import (
    EvidentialLayer,
    evidential_loss,
    compute_uncertainty
)

__all__ = [
    "EvidentialLayer",
    "evidential_loss",
    "compute_uncertainty"
]
