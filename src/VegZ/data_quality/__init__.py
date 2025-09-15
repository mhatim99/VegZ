"""
Data Quality & Validation Module

This module provides functionality for:
- Flagging records missing spatial or taxonomic information
- Identifying invalid coordinate ranges and transposed coordinates
- Detecting coordinates inconsistent with country boundaries
- Deriving country names from valid coordinates
- Assessing coordinate precision levels
- Extracting and validating collection dates
- Flagging suspicious temporal information
- Detecting records at centroids, urban areas, institutions
- Identifying geographic outliers and duplicate records
"""

from .spatial_validation import *
from .temporal_validation import *

__all__ = [
    'SpatialValidator',
    'TemporalValidator'
]