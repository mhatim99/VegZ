"""
Data standardization and integration utilities.

Copyright (c) 2025 Mohamed Z. Hatim
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import warnings

# Suppress fuzzywuzzy Levenshtein performance warning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*Using slow pure-python SequenceMatcher.*")
    from fuzzywuzzy import fuzz, process
import re


class DataStandardizer:
    """Main class for standardizing and integrating heterogeneous datasets."""
    
    def __init__(self):
        self.standard_columns = {
            'spatial': ['latitude', 'longitude', 'coordinate_precision', 'coordinate_system'],
            'taxonomic': ['species', 'genus', 'family', 'order', 'class', 'kingdom'],
            'temporal': ['date', 'year', 'month', 'day'],
            'ecological': ['abundance', 'cover', 'frequency', 'biomass'],
            'environmental': ['temperature', 'precipitation', 'elevation', 'slope', 'aspect'],
            'identification': ['plot_id', 'site_id', 'sample_id', 'observer']
        }
        
        self.species_standardizer = SpeciesNameStandardizer()
        self.coordinate_standardizer = CoordinateStandardizer()
        
    def standardize_dataset(self, 
                           df: pd.DataFrame,
                           dataset_type: str = 'vegetation',
                           column_mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Standardize a dataset to common format.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset
        dataset_type : str
            Type of dataset ('vegetation', 'environmental', 'spatial')
        column_mapping : dict, optional
            Custom column name mappings
            
        Returns:
        --------
        pd.DataFrame
            Standardized dataset
        """
        df_std = df.copy()
        
        # Apply custom column mapping if provided
        if column_mapping:
            df_std = df_std.rename(columns=column_mapping)
        
        # Auto-detect and standardize common columns
        df_std = self._auto_standardize_columns(df_std)
        
        # Standardize species names if present
        if 'species' in df_std.columns:
            df_std = self.species_standardizer.standardize_dataframe(df_std)
        
        # Standardize coordinates if present
        if 'latitude' in df_std.columns and 'longitude' in df_std.columns:
            df_std = self.coordinate_standardizer.standardize_coordinates(df_std)
        
        # Standardize dates if present
        if 'date' in df_std.columns:
            df_std = self._standardize_dates(df_std)
        
        # Handle missing values
        df_std = self._handle_missing_values(df_std)
        
        return df_std
    
    def integrate_datasets(self, 
                          datasets: List[pd.DataFrame],
                          join_columns: List[str],
                          dataset_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Integrate multiple standardized datasets.
        
        Parameters:
        -----------
        datasets : list of pd.DataFrame
            List of standardized datasets
        join_columns : list
            Columns to use for joining datasets
        dataset_names : list, optional
            Names for source datasets
            
        Returns:
        --------
        pd.DataFrame
            Integrated dataset
        """
        if not datasets:
            raise ValueError("No datasets provided")
        
        if dataset_names is None:
            dataset_names = [f"dataset_{i}" for i in range(len(datasets))]
        
        # Add source column to each dataset
        for i, df in enumerate(datasets):
            df['source_dataset'] = dataset_names[i]
        
        # Start with first dataset
        result = datasets[0].copy()
        
        # Merge with remaining datasets
        for i, df in enumerate(datasets[1:], 1):
            # Find common columns for merging
            common_cols = [col for col in join_columns if col in result.columns and col in df.columns]
            
            if not common_cols:
                warnings.warn(f"No common columns found for dataset {i}. Concatenating instead.")
                result = pd.concat([result, df], ignore_index=True, sort=False)
            else:
                result = pd.merge(result, df, on=common_cols, how='outer', suffixes=('', f'_{i}'))
        
        return result
    
    def _auto_standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Auto-detect and standardize column names."""
        column_mappings = {
            # Spatial columns
            'lat': 'latitude',
            'y': 'latitude',
            'coord_y': 'latitude',
            'lat_dd': 'latitude',
            'decimal_latitude': 'latitude',
            
            'lon': 'longitude',
            'lng': 'longitude',
            'long': 'longitude',
            'x': 'longitude',
            'coord_x': 'longitude',
            'lon_dd': 'longitude',
            'decimal_longitude': 'longitude',
            
            # Taxonomic columns
            'species_name': 'species',
            'scientific_name': 'species',
            'taxon': 'species',
            'taxa': 'species',
            'binomial': 'species',
            
            'genus_species': 'species',
            'tax_genus': 'genus',
            'tax_family': 'family',
            
            # Ecological columns
            'cover': 'abundance',
            'coverage': 'abundance',
            'percent_cover': 'abundance',
            'pct_cover': 'abundance',
            'abundance_value': 'abundance',
            
            # Identification columns
            'plot': 'plot_id',
            'site': 'site_id',
            'releve': 'plot_id',
            'quadrat': 'plot_id',
            'sample': 'sample_id',
            
            # Temporal columns
            'sampling_date': 'date',
            'survey_date': 'date',
            'collection_date': 'date',
            'obs_date': 'date',
            'eventdate': 'date'
        }
        
        # Apply mappings
        for old_name, new_name in column_mappings.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        return df
    
    def _standardize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize date columns."""
        if 'date' in df.columns:
            # Try to parse dates
            df['date'] = pd.to_datetime(df['date'], errors='coerce', infer_datetime_format=True)
            
            # Extract components
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['day_of_year'] = df['date'].dt.dayofyear
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to column type."""
        for column in df.columns:
            if column in self.standard_columns['spatial']:
                # Don't fill spatial coordinates
                continue
            elif column in self.standard_columns['taxonomic']:
                # Don't fill taxonomic information
                continue
            elif column in self.standard_columns['ecological']:
                # Fill ecological measurements with 0 for absence
                if df[column].dtype in ['int64', 'float64']:
                    df[column] = df[column].fillna(0)
            elif column in self.standard_columns['environmental']:
                # Fill environmental variables with median
                if df[column].dtype in ['int64', 'float64']:
                    df[column] = df[column].fillna(df[column].median())
        
        return df


class SpeciesNameStandardizer:
    """Standardize and clean species names with fuzzy matching."""
    
    def __init__(self):
        self.author_patterns = [
            r'\s+L\.$',  # Linnaeus
            r'\s+\([^)]+\)\s*[A-Z][a-z]+',  # (Author) SecondAuthor
            r'\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$',  # Author names
            r'\s+\d{4}',  # Years
            r'\s+ex\s+[A-Z][a-z]+',  # ex Author
            r'\s+sensu\s+[A-Z][a-z]+',  # sensu Author
        ]
        
        self.infraspecific_markers = ['var.', 'subsp.', 'ssp.', 'f.', 'forma']
        
    def clean_species_name(self, name: str) -> str:
        """
        Clean a single species name.
        
        Parameters:
        -----------
        name : str
            Raw species name
            
        Returns:
        --------
        str
            Cleaned species name
        """
        if pd.isna(name) or not isinstance(name, str):
            return ''
        
        # Basic cleaning
        name = name.strip()
        name = re.sub(r'\s+', ' ', name)  # Multiple spaces to single
        
        # Remove author names
        for pattern in self.author_patterns:
            name = re.sub(pattern, '', name)
        
        # Handle infraspecific names
        for marker in self.infraspecific_markers:
            if marker in name:
                parts = name.split(marker)
                if len(parts) >= 2:
                    genus_species = parts[0].strip()
                    infraspecific = parts[1].strip()
                    name = f"{genus_species} {marker} {infraspecific}"
                break
        
        # Capitalize properly
        words = name.split()
        if len(words) >= 2:
            # Genus capitalized, species lowercase
            words[0] = words[0].capitalize()
            words[1] = words[1].lower()
            
            # Handle infraspecific markers
            for i, word in enumerate(words[2:], 2):
                if word in self.infraspecific_markers:
                    if i + 1 < len(words):
                        words[i + 1] = words[i + 1].lower()
        
        return ' '.join(words).strip()
    
    def standardize_dataframe(self, df: pd.DataFrame, 
                            species_column: str = 'species') -> pd.DataFrame:
        """
        Standardize species names in a dataframe.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        species_column : str
            Name of the species column
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with standardized species names
        """
        if species_column not in df.columns:
            return df
        
        df_clean = df.copy()
        
        # Clean species names
        df_clean[f'{species_column}_original'] = df_clean[species_column]
        df_clean[species_column] = df_clean[species_column].apply(self.clean_species_name)
        
        # Extract genus
        df_clean['genus'] = df_clean[species_column].apply(
            lambda x: x.split()[0] if x and len(x.split()) > 0 else ''
        )
        
        # Extract specific epithet
        df_clean['specific_epithet'] = df_clean[species_column].apply(
            lambda x: x.split()[1] if x and len(x.split()) > 1 else ''
        )
        
        return df_clean
    
    def fuzzy_match_species(self, 
                           query_species: List[str],
                           reference_species: List[str],
                           threshold: int = 80) -> Dict[str, str]:
        """
        Fuzzy match species names against a reference list.
        
        Parameters:
        -----------
        query_species : list
            Species names to match
        reference_species : list
            Reference species list
        threshold : int
            Minimum match score (0-100)
            
        Returns:
        --------
        dict
            Mapping of query species to best matches
        """
        matches = {}
        
        for query in query_species:
            if not query or query == '':
                continue
                
            best_match, score = process.extractOne(query, reference_species)
            
            if score >= threshold:
                matches[query] = best_match
            else:
                matches[query] = query  # Keep original if no good match
        
        return matches


class CoordinateStandardizer:
    """Standardize coordinate data and handle different formats."""
    
    def __init__(self):
        self.coordinate_patterns = {
            'decimal': r'^-?\d+\.?\d*$',
            'dms': r'(\d+)[°d]\s*(\d+)[\'m]\s*([\d.]+)[\"s]?\s*([NSEW])?',
            'dm': r'(\d+)[°d]\s*([\d.]+)[\'m]\s*([NSEW])?'
        }
    
    def standardize_coordinates(self, df: pd.DataFrame,
                              lat_col: str = 'latitude',
                              lon_col: str = 'longitude') -> pd.DataFrame:
        """
        Standardize coordinate formats to decimal degrees.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        lat_col : str
            Latitude column name
        lon_col : str
            Longitude column name
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with standardized coordinates
        """
        df_std = df.copy()
        
        if lat_col in df.columns:
            df_std[lat_col] = df_std[lat_col].apply(self._convert_to_decimal)
            df_std[lat_col] = pd.to_numeric(df_std[lat_col], errors='coerce')
        
        if lon_col in df.columns:
            df_std[lon_col] = df_std[lon_col].apply(self._convert_to_decimal)
            df_std[lon_col] = pd.to_numeric(df_std[lon_col], errors='coerce')
        
        # Validate coordinate ranges
        if lat_col in df_std.columns:
            invalid_lat = (df_std[lat_col] < -90) | (df_std[lat_col] > 90)
            if invalid_lat.any():
                warnings.warn(f"Found {invalid_lat.sum()} invalid latitude values")
                df_std.loc[invalid_lat, lat_col] = np.nan
        
        if lon_col in df_std.columns:
            invalid_lon = (df_std[lon_col] < -180) | (df_std[lon_col] > 180)
            if invalid_lon.any():
                warnings.warn(f"Found {invalid_lon.sum()} invalid longitude values")
                df_std.loc[invalid_lon, lon_col] = np.nan
        
        return df_std
    
    def _convert_to_decimal(self, coord_str) -> float:
        """Convert coordinate string to decimal degrees."""
        if pd.isna(coord_str):
            return np.nan
        
        coord_str = str(coord_str).strip()
        
        # Already decimal
        if re.match(self.coordinate_patterns['decimal'], coord_str):
            return float(coord_str)
        
        # DMS format
        dms_match = re.match(self.coordinate_patterns['dms'], coord_str, re.IGNORECASE)
        if dms_match:
            degrees, minutes, seconds, hemisphere = dms_match.groups()
            decimal = float(degrees) + float(minutes)/60 + float(seconds)/3600
            
            if hemisphere and hemisphere.upper() in ['S', 'W']:
                decimal = -decimal
            
            return decimal
        
        # DM format
        dm_match = re.match(self.coordinate_patterns['dm'], coord_str, re.IGNORECASE)
        if dm_match:
            degrees, minutes, hemisphere = dm_match.groups()
            decimal = float(degrees) + float(minutes)/60
            
            if hemisphere and hemisphere.upper() in ['S', 'W']:
                decimal = -decimal
            
            return decimal
        
        # Try to extract just the numeric part
        numeric = re.findall(r'-?\d+\.?\d*', coord_str)
        if numeric:
            return float(numeric[0])
        
        return np.nan