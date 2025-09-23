"""
Spatial data validation module - comprehensive coordinate and geographic validation.

Copyright (c) 2025 Mohamed Z. Hatim
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import requests
import json
from pathlib import Path

try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    from shapely.ops import transform
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    warnings.warn("GeoPandas not available. Install with: pip install geopandas")

try:
    import pyproj
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False


class SpatialValidator:
    """Comprehensive spatial data validation for biodiversity records."""
    
    def __init__(self):
        """Initialize spatial validator."""
        self.country_boundaries = None
        self.urban_areas = None
        self.institutions = None
        
# Copyright (c) 2025 Mohamed Z. Hatim
        self.coord_ranges = {
            'latitude': (-90, 90),
            'longitude': (-180, 180)
        }
        
# Copyright (c) 2025 Mohamed Z. Hatim
        self.known_centroids = {
            'country_centroids': {
                'Brazil': (-14.2350, -51.9253),
                'United States': (39.8283, -98.5795),
                'Canada': (56.1304, -106.3468),
                'Australia': (-25.2744, 133.7751),
                'India': (20.5937, 78.9629),
                'China': (35.8617, 104.1954),
                'Russia': (61.5240, 105.3188)
            },
            'institution_coordinates': {
# Copyright (c) 2025 Mohamed Z. Hatim
                'Smithsonian': (38.8912, -77.0253),
                'Harvard_Herbaria': (42.3826, -71.1162),
                'Kew_Gardens': (51.4829, -0.2945),
                'Natural_History_Museum_London': (51.4966, -0.1764),
                'MNHN_Paris': (48.8430, 2.3547)
            }
        }
    
    def validate_coordinates(self, df: pd.DataFrame,
                           lat_col: str = 'latitude',
                           lon_col: str = 'longitude') -> Dict[str, Any]:
        """
        Comprehensive coordinate validation.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with coordinate columns
        lat_col : str
            Latitude column name
        lon_col : str
            Longitude column name
            
        Returns:
        --------
        dict
            Validation results and flags
        """
        results = {
            'total_records': len(df),
            'flags': {},
            'valid_coordinates': 0,
            'issues_found': []
        }
        
        if lat_col not in df.columns or lon_col not in df.columns:
            results['issues_found'].append(f"Missing coordinate columns: {lat_col}, {lon_col}")
            return results
        
# Copyright (c) 2025 Mohamed Z. Hatim
        flags_df = pd.DataFrame(index=df.index)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        missing_coords = df[lat_col].isna() | df[lon_col].isna()
        flags_df['missing_coordinates'] = missing_coords
        results['flags']['missing_coordinates'] = missing_coords.sum()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        invalid_lat = (df[lat_col] < -90) | (df[lat_col] > 90)
        invalid_lon = (df[lon_col] < -180) | (df[lon_col] > 180)
        flags_df['invalid_ranges'] = invalid_lat | invalid_lon
        results['flags']['invalid_ranges'] = (invalid_lat | invalid_lon).sum()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        zero_coords = (df[lat_col] == 0) & (df[lon_col] == 0)
        flags_df['zero_coordinates'] = zero_coords
        results['flags']['zero_coordinates'] = zero_coords.sum()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        transposed = self._detect_transposed_coordinates(df, lat_col, lon_col)
        flags_df['potentially_transposed'] = transposed
        results['flags']['potentially_transposed'] = transposed.sum()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        low_precision = self._detect_low_precision(df, lat_col, lon_col)
        flags_df['low_precision'] = low_precision
        results['flags']['low_precision'] = low_precision.sum()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        centroids = self._detect_centroids(df, lat_col, lon_col)
        flags_df['potential_centroids'] = centroids
        results['flags']['potential_centroids'] = centroids.sum()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        urban = self._detect_urban_coordinates(df, lat_col, lon_col)
        flags_df['urban_areas'] = urban
        results['flags']['urban_areas'] = urban.sum()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        institutions = self._detect_institution_coordinates(df, lat_col, lon_col)
        flags_df['near_institutions'] = institutions
        results['flags']['near_institutions'] = institutions.sum()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        all_flags = flags_df.any(axis=1)
        results['valid_coordinates'] = (~all_flags).sum()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        precision_assessment = self._assess_coordinate_precision(df, lat_col, lon_col)
        results['precision_assessment'] = precision_assessment
        
# Copyright (c) 2025 Mohamed Z. Hatim
        results['flags_dataframe'] = flags_df
        
        return results
    
    def _detect_transposed_coordinates(self, df: pd.DataFrame,
                                     lat_col: str, lon_col: str) -> pd.Series:
        """Detect potentially transposed coordinates."""
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
        
# Copyright (c) 2025 Mohamed Z. Hatim
        extreme_lat = (df[lat_col].abs() > 90) & (df[lat_col].abs() <= 180)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        would_be_valid_if_swapped = (
            (df[lat_col].abs() > 90) & (df[lat_col].abs() <= 180) &
            (df[lon_col].abs() <= 90)
        )
        
        return extreme_lat | would_be_valid_if_swapped
    
    def _detect_low_precision(self, df: pd.DataFrame,
                            lat_col: str, lon_col: str,
                            precision_threshold: int = 2) -> pd.Series:
        """Detect coordinates with suspiciously low precision."""
        def count_decimals(value):
            if pd.isna(value):
                return 0
            str_val = str(float(value))
            if '.' in str_val:
                return len(str_val.split('.')[1].rstrip('0'))
            return 0
        
        lat_precision = df[lat_col].apply(count_decimals)
        lon_precision = df[lon_col].apply(count_decimals)
        
        return (lat_precision < precision_threshold) | (lon_precision < precision_threshold)
    
    def _detect_centroids(self, df: pd.DataFrame,
                         lat_col: str, lon_col: str,
                         tolerance: float = 0.01) -> pd.Series:
        """Detect coordinates that match known geographic centroids."""
        centroid_flags = pd.Series(False, index=df.index)
        
        for category, centroids in self.known_centroids.items():
            for name, (cent_lat, cent_lon) in centroids.items():
# Copyright (c) 2025 Mohamed Z. Hatim
                distance = np.sqrt(
                    (df[lat_col] - cent_lat)**2 + (df[lon_col] - cent_lon)**2
                )
                near_centroid = distance < tolerance
                centroid_flags |= near_centroid
        
        return centroid_flags
    
    def _detect_urban_coordinates(self, df: pd.DataFrame,
                                lat_col: str, lon_col: str) -> pd.Series:
        """Detect coordinates in major urban areas."""
# Copyright (c) 2025 Mohamed Z. Hatim
        urban_flags = pd.Series(False, index=df.index)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        major_cities = {
            'New_York': (40.7128, -74.0060, 0.5),
            'London': (51.5074, -0.1278, 0.3),
            'Tokyo': (35.6762, 139.6503, 0.4),
            'Paris': (48.8566, 2.3522, 0.3),
            'Berlin': (52.5200, 13.4050, 0.3),
            'Sydney': (-33.8688, 151.2093, 0.3),
            'San_Francisco': (37.7749, -122.4194, 0.3),
            'Los_Angeles': (34.0522, -118.2437, 0.5)
        }
        
        for city, (city_lat, city_lon, radius) in major_cities.items():
            distance = np.sqrt(
                (df[lat_col] - city_lat)**2 + (df[lon_col] - city_lon)**2
            )
            in_city = distance < radius
            urban_flags |= in_city
        
        return urban_flags
    
    def _detect_institution_coordinates(self, df: pd.DataFrame,
                                      lat_col: str, lon_col: str,
                                      tolerance: float = 0.001) -> pd.Series:
        """Detect coordinates near known institutions."""
        institution_flags = pd.Series(False, index=df.index)
        
        if 'institution_coordinates' in self.known_centroids:
            for inst_name, (inst_lat, inst_lon) in self.known_centroids['institution_coordinates'].items():
                distance = np.sqrt(
                    (df[lat_col] - inst_lat)**2 + (df[lon_col] - inst_lon)**2
                )
                near_institution = distance < tolerance
                institution_flags |= near_institution
        
        return institution_flags
    
    def _assess_coordinate_precision(self, df: pd.DataFrame,
                                   lat_col: str, lon_col: str) -> Dict[str, Any]:
        """Assess coordinate precision levels."""
        def precision_to_uncertainty(decimals):
            """Convert decimal places to approximate uncertainty in meters."""
# Copyright (c) 2025 Mohamed Z. Hatim
            if decimals == 0:
                return 111000  # ~111 km
            elif decimals == 1:
                return 11100   # ~11 km
            elif decimals == 2:
                return 1110    # ~1.1 km
            elif decimals == 3:
                return 111     # ~111 m
            elif decimals == 4:
                return 11      # ~11 m
            elif decimals >= 5:
                return 1       # ~1 m or better
            return None
        
        def count_decimals(value):
            if pd.isna(value):
                return 0
            str_val = str(float(value))
            if '.' in str_val:
                return len(str_val.split('.')[1].rstrip('0'))
            return 0
        
        lat_decimals = df[lat_col].apply(count_decimals)
        lon_decimals = df[lon_col].apply(count_decimals)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        min_precision = np.minimum(lat_decimals, lon_decimals)
        uncertainty = min_precision.apply(precision_to_uncertainty)
        
        precision_summary = {
            'mean_decimal_places': min_precision.mean(),
            'median_decimal_places': min_precision.median(),
            'precision_distribution': min_precision.value_counts().sort_index().to_dict(),
            'estimated_uncertainty_meters': uncertainty.describe().to_dict()
        }
        
        return precision_summary
    
    def derive_country_from_coordinates(self, df: pd.DataFrame,
                                      lat_col: str = 'latitude',
                                      lon_col: str = 'longitude',
                                      country_col: str = 'derived_country') -> pd.DataFrame:
        """
        Derive country names from coordinates using reverse geocoding.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with coordinates
        lat_col : str
            Latitude column name
        lon_col : str
            Longitude column name
        country_col : str
            Output column name for derived countries
            
        Returns:
        --------
        pd.DataFrame
            Dataset with derived country column
        """
        result_df = df.copy()
        result_df[country_col] = np.nan
        
        if not GEOPANDAS_AVAILABLE:
            warnings.warn("GeoPandas not available. Cannot derive countries from coordinates.")
            return result_df
        
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
        
        try:
# Copyright (c) 2025 Mohamed Z. Hatim
            valid_coords = (~df[lat_col].isna()) & (~df[lon_col].isna())
            
            if valid_coords.any():
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
                countries = self._approximate_country_from_coordinates(
                    df.loc[valid_coords, lat_col].values,
                    df.loc[valid_coords, lon_col].values
                )
                
                result_df.loc[valid_coords, country_col] = countries
        
        except Exception as e:
            warnings.warn(f"Error deriving countries: {e}")
        
        return result_df
    
    def _approximate_country_from_coordinates(self, latitudes: np.ndarray, 
                                            longitudes: np.ndarray) -> List[str]:
        """Rough country approximation based on coordinate ranges."""
        countries = []
        
        for lat, lon in zip(latitudes, longitudes):
            country = "Unknown"
            
# Copyright (c) 2025 Mohamed Z. Hatim
            if 25 <= lat <= 49 and -125 <= lon <= -66:
                country = "United States"
            elif 42 <= lat <= 70 and -141 <= lon <= -52:
                country = "Canada"
            elif -44 <= lat <= -10 and 113 <= lon <= 154:
                country = "Australia"
            elif 8 <= lat <= 37 and 68 <= lon <= 97:
                country = "India"
            elif 18 <= lat <= 54 and 73 <= lon <= 135:
                country = "China"
            elif -34 <= lat <= 5 and -74 <= lon <= -32:
                country = "Brazil"
            elif 36 <= lat <= 71 and -9 <= lon <= 32:
                country = "Europe"  # Very rough
            elif -35 <= lat <= 37 and -18 <= lon <= 52:
                country = "Africa"  # Very rough
            
            countries.append(country)
        
        return countries
    
    def detect_geographic_outliers(self, df: pd.DataFrame,
                                 lat_col: str = 'latitude',
                                 lon_col: str = 'longitude',
                                 method: str = 'iqr',
                                 threshold: float = 1.5) -> pd.Series:
        """
        Detect geographic outliers using statistical methods.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with coordinates
        lat_col : str
            Latitude column name
        lon_col : str
            Longitude column name
        method : str
            Outlier detection method ('iqr', 'zscore', 'isolation_forest')
        threshold : float
            Outlier threshold
            
        Returns:
        --------
        pd.Series
            Boolean series indicating outliers
        """
        outlier_flags = pd.Series(False, index=df.index)
        
        valid_coords = (~df[lat_col].isna()) & (~df[lon_col].isna())
        
        if not valid_coords.any():
            return outlier_flags
        
        if method == 'iqr':
# Copyright (c) 2025 Mohamed Z. Hatim
            for col in [lat_col, lon_col]:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_flags |= outliers
        
        elif method == 'zscore':
# Copyright (c) 2025 Mohamed Z. Hatim
            for col in [lat_col, lon_col]:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outliers = pd.Series(False, index=df.index)
                outliers.loc[df[col].notna()] = z_scores > threshold
                outlier_flags |= outliers
        
        elif method == 'isolation_forest':
            try:
                from sklearn.ensemble import IsolationForest
                
                coords = df.loc[valid_coords, [lat_col, lon_col]].values
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_pred = iso_forest.fit_predict(coords)
                
                outliers = pd.Series(False, index=df.index)
                outliers.loc[valid_coords] = outlier_pred == -1
                outlier_flags = outliers
                
            except ImportError:
                warnings.warn("scikit-learn not available for isolation forest method")
                return self.detect_geographic_outliers(df, lat_col, lon_col, method='iqr', threshold=threshold)
        
        return outlier_flags
    
    def validate_coordinate_consistency(self, df: pd.DataFrame,
                                      lat_col: str = 'latitude',
                                      lon_col: str = 'longitude',
                                      country_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate consistency between coordinates and country information.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with coordinates and optionally country information
        lat_col : str
            Latitude column name  
        lon_col : str
            Longitude column name
        country_col : str, optional
            Country column name
            
        Returns:
        --------
        dict
            Consistency validation results
        """
        results = {
            'total_records': len(df),
            'consistency_flags': {},
            'recommendations': []
        }
        
        if country_col and country_col in df.columns:
# Copyright (c) 2025 Mohamed Z. Hatim
            df_with_derived = self.derive_country_from_coordinates(df, lat_col, lon_col)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            stated_countries = df[country_col].fillna('Unknown')
            derived_countries = df_with_derived['derived_country'].fillna('Unknown')
            
            inconsistent = (stated_countries != derived_countries) & \
                          (stated_countries != 'Unknown') & \
                          (derived_countries != 'Unknown')
            
            results['consistency_flags']['country_coordinate_mismatch'] = inconsistent.sum()
            results['inconsistent_records'] = df[inconsistent]
            
            if inconsistent.any():
                results['recommendations'].append(
                    f"Found {inconsistent.sum()} records with country-coordinate inconsistencies"
                )
        
        return results
    
    def generate_spatial_quality_report(self, df: pd.DataFrame,
                                      lat_col: str = 'latitude',
                                      lon_col: str = 'longitude',
                                      country_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive spatial data quality report.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset to validate
        lat_col : str
            Latitude column name
        lon_col : str
            Longitude column name
        country_col : str, optional
            Country column name
            
        Returns:
        --------
        dict
            Comprehensive spatial quality report
        """
        report = {
            'dataset_summary': {
                'total_records': len(df),
                'records_with_coordinates': (~df[lat_col].isna() & ~df[lon_col].isna()).sum()
            }
        }
        
# Copyright (c) 2025 Mohamed Z. Hatim
        coord_validation = self.validate_coordinates(df, lat_col, lon_col)
        report['coordinate_validation'] = coord_validation
        
# Copyright (c) 2025 Mohamed Z. Hatim
        outliers = self.detect_geographic_outliers(df, lat_col, lon_col)
        report['geographic_outliers'] = {
            'count': outliers.sum(),
            'percentage': (outliers.sum() / len(df)) * 100
        }
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if country_col:
            consistency = self.validate_coordinate_consistency(df, lat_col, lon_col, country_col)
            report['coordinate_consistency'] = consistency
        
# Copyright (c) 2025 Mohamed Z. Hatim
        recommendations = []
        
        if coord_validation['flags']['missing_coordinates'] > 0:
            recommendations.append(f"Fill {coord_validation['flags']['missing_coordinates']} missing coordinates")
        
        if coord_validation['flags']['invalid_ranges'] > 0:
            recommendations.append(f"Fix {coord_validation['flags']['invalid_ranges']} coordinates outside valid ranges")
        
        if coord_validation['flags']['potentially_transposed'] > 0:
            recommendations.append(f"Check {coord_validation['flags']['potentially_transposed']} potentially transposed coordinates")
        
        if coord_validation['flags']['low_precision'] > 0:
            recommendations.append(f"Improve precision for {coord_validation['flags']['low_precision']} low-precision coordinates")
        
        report['recommendations'] = recommendations
        
        return report