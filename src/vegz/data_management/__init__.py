"""
Data Management & Preprocessing Module

This module provides functionality for:
- Parsing vegetation survey data from various formats
- Integration with remote sensing APIs
- Standardizing and integrating heterogeneous datasets
- Handling Darwin Core standards
- Data quality handling and transformations
- Coordinate system transformations
"""

from .parsers import *
from .remote_sensing import *
from .standardization import *
from .darwin_core import *
from .transformations import *
from .coordinate_systems import *

__all__ = [
    'VegetationDataParser',
    'TurbovegParser', 
    'RemoteSensingAPI',
    'DataStandardizer',
    'DarwinCoreHandler',
    'DataTransformer',
    'CoordinateTransformer'
]