"""
Data Management & Preprocessing Module

This module provides functionality for:
- Parsing vegetation survey data from various formats
- Integration with remote sensing APIs
- Standardizing and integrating heterogeneous datasets
- Handling Darwin Core standards
- Data quality handling and transformations
- Coordinate system transformations
- Taxonomic name resolution against online databases

Copyright (c) 2025 Mohamed Z. Hatim
"""

from .parsers import *
from .remote_sensing import *
from .standardization import *
from .darwin_core import *
from .transformations import *
from .coordinate_systems import *
from .taxonomic_resolver import (
    TaxonomicResolver,
    resolve_species_names,
    resolve_species_from_file,
    update_species_in_dataframe
)

__all__ = [
    'VegetationDataParser',
    'TurbovegParser',
    'RemoteSensingAPI',
    'DataStandardizer',
    'DarwinCoreHandler',
    'DataTransformer',
    'CoordinateTransformer',
    'TaxonomicResolver',
    'resolve_species_names',
    'resolve_species_from_file',
    'update_species_in_dataframe'
]