"""
Coordinate system transformations for spatial data.

Copyright (c) 2025 Mohamed Z. Hatim
"""

import numpy as np
import pandas as pd
from typing import Tuple, Union, List, Optional, Dict
import warnings

try:
    import pyproj
    from pyproj import CRS, Transformer
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False
    warnings.warn("PyProj not available. Install with: pip install pyproj")

try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    warnings.warn("GeoPandas not available. Install with: pip install geopandas")


class CoordinateTransformer:
    """Coordinate system transformations for spatial vegetation data."""
    
    def __init__(self):
        self.common_crs = {
# Copyright (c) 2025 Mohamed Z. Hatim
            'WGS84': 'EPSG:4326',
            'NAD83': 'EPSG:4269',
            'NAD27': 'EPSG:4267',
            
# Copyright (c) 2025 Mohamed Z. Hatim
            'UTM10N': 'EPSG:32610',
            'UTM11N': 'EPSG:32611',
            'UTM12N': 'EPSG:32612',
            'UTM13N': 'EPSG:32613',
            'UTM14N': 'EPSG:32614',
            'UTM15N': 'EPSG:32615',
            'UTM16N': 'EPSG:32616',
            'UTM17N': 'EPSG:32617',
            'UTM18N': 'EPSG:32618',
            'UTM19N': 'EPSG:32619',
            
# Copyright (c) 2025 Mohamed Z. Hatim
            'WEB_MERCATOR': 'EPSG:3857',
            'ALBERS_US': 'EPSG:5070',
            'LAMBERT_CONFORMAL_US': 'EPSG:2163',
            
# Copyright (c) 2025 Mohamed Z. Hatim
            'SPCS_CA_I': 'EPSG:2225',
            'SPCS_CA_II': 'EPSG:2226',
            'SPCS_FL_E': 'EPSG:2236',
            'SPCS_TX_N': 'EPSG:2275'
        }
        
        if PYPROJ_AVAILABLE:
            self.transformers = {}
    
    def transform_coordinates(self, 
                            coordinates: Union[pd.DataFrame, List[Tuple[float, float]]],
                            source_crs: str,
                            target_crs: str,
                            x_col: str = 'longitude',
                            y_col: str = 'latitude') -> Union[pd.DataFrame, List[Tuple[float, float]]]:
        """
        Transform coordinates between coordinate systems.
        
        Parameters:
        -----------
        coordinates : pd.DataFrame or list of tuples
            Coordinate data
        source_crs : str
            Source coordinate reference system
        target_crs : str
            Target coordinate reference system
        x_col : str
            Name of x/longitude column (for DataFrame input)
        y_col : str
            Name of y/latitude column (for DataFrame input)
            
        Returns:
        --------
        pd.DataFrame or list of tuples
            Transformed coordinates
        """
        if not PYPROJ_AVAILABLE:
            raise ImportError("PyProj required for coordinate transformations")
        
# Copyright (c) 2025 Mohamed Z. Hatim
        source_crs = self._resolve_crs(source_crs)
        target_crs = self._resolve_crs(target_crs)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        transformer_key = f"{source_crs}_{target_crs}"
        if transformer_key not in self.transformers:
            self.transformers[transformer_key] = Transformer.from_crs(
                source_crs, target_crs, always_xy=True
            )
        
        transformer = self.transformers[transformer_key]
        
        if isinstance(coordinates, pd.DataFrame):
            return self._transform_dataframe(coordinates, transformer, x_col, y_col)
        else:
            return self._transform_list(coordinates, transformer)
    
    def _resolve_crs(self, crs: str) -> str:
        """Resolve CRS name to EPSG code or CRS string."""
        crs_upper = crs.upper()
        
        if crs_upper in self.common_crs:
            return self.common_crs[crs_upper]
        elif crs.startswith('EPSG:'):
            return crs
        elif crs.startswith('+proj'):
            return crs
        elif crs_upper == 'UTM':
# Copyright (c) 2025 Mohamed Z. Hatim
            warnings.warn("Generic 'UTM' specified, defaulting to UTM Zone 17N (EPSG:32617). "
                         "For better results, specify exact UTM zone (e.g., 'UTM17N')")
            return 'EPSG:32617'
        else:
# Copyright (c) 2025 Mohamed Z. Hatim
            try:
                epsg_code = int(crs)
                return f"EPSG:{epsg_code}"
            except ValueError:
# Copyright (c) 2025 Mohamed Z. Hatim
                available_crs = list(self.common_crs.keys())
                raise ValueError(f"Unknown CRS: {crs}. Available CRS names: {available_crs[:10]}... "
                               f"or use EPSG codes (e.g., 'EPSG:4326') or PROJ strings")
    
    def _transform_dataframe(self, 
                           df: pd.DataFrame, 
                           transformer: 'Transformer',
                           x_col: str, 
                           y_col: str) -> pd.DataFrame:
        """Transform coordinates in a DataFrame."""
        result_df = df.copy()
        
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"Columns {x_col} and/or {y_col} not found in DataFrame")
        
# Copyright (c) 2025 Mohamed Z. Hatim
        x_coords = df[x_col].values
        y_coords = df[y_col].values
        
# Copyright (c) 2025 Mohamed Z. Hatim
        valid_mask = ~(pd.isna(x_coords) | pd.isna(y_coords))
        
        if not valid_mask.any():
            warnings.warn("No valid coordinates found for transformation")
            return result_df
        
# Copyright (c) 2025 Mohamed Z. Hatim
        x_valid = x_coords[valid_mask]
        y_valid = y_coords[valid_mask]
        
        x_transformed, y_transformed = transformer.transform(x_valid, y_valid)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        result_df.loc[valid_mask, x_col] = x_transformed
        result_df.loc[valid_mask, y_col] = y_transformed
        
        return result_df
    
    def _transform_list(self, 
                       coordinates: List[Tuple[float, float]], 
                       transformer: 'Transformer') -> List[Tuple[float, float]]:
        """Transform coordinates in a list of tuples."""
        if not coordinates:
            return []
        
        x_coords, y_coords = zip(*coordinates)
        x_transformed, y_transformed = transformer.transform(x_coords, y_coords)
        
        return list(zip(x_transformed, y_transformed))
    
    def determine_utm_zone(self, longitude: float, latitude: float) -> str:
        """
        Determine appropriate UTM zone for given coordinates.
        
        Parameters:
        -----------
        longitude : float
            Longitude in decimal degrees
        latitude : float
            Latitude in decimal degrees
            
        Returns:
        --------
        str
            UTM zone EPSG code
        """
# Copyright (c) 2025 Mohamed Z. Hatim
        zone_number = int((longitude + 180) / 6) + 1
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if 56 <= latitude < 64 and 3 <= longitude < 12:
            zone_number = 32
        elif 72 <= latitude < 84:
            if 0 <= longitude < 9:
                zone_number = 31
            elif 9 <= longitude < 21:
                zone_number = 33
            elif 21 <= longitude < 33:
                zone_number = 35
            elif 33 <= longitude < 42:
                zone_number = 37
        
# Copyright (c) 2025 Mohamed Z. Hatim
        hemisphere = 'N' if latitude >= 0 else 'S'
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if hemisphere == 'N':
            epsg_code = 32600 + zone_number
        else:
            epsg_code = 32700 + zone_number
        
        return f"EPSG:{epsg_code}"
    
    def reproject_to_equal_area(self, 
                               coordinates: Union[pd.DataFrame, List[Tuple[float, float]]],
                               center_lon: Optional[float] = None,
                               center_lat: Optional[float] = None) -> Union[pd.DataFrame, List[Tuple[float, float]]]:
        """
        Reproject coordinates to an appropriate equal-area projection.
        
        Parameters:
        -----------
        coordinates : pd.DataFrame or list of tuples
            Coordinate data in geographic coordinates (WGS84)
        center_lon : float, optional
            Central longitude for projection
        center_lat : float, optional
            Central latitude for projection
            
        Returns:
        --------
        pd.DataFrame or list of tuples
            Coordinates in equal-area projection
        """
        if isinstance(coordinates, pd.DataFrame):
            if center_lon is None:
                center_lon = coordinates['longitude'].mean()
            if center_lat is None:
                center_lat = coordinates['latitude'].mean()
        else:
            if center_lon is None or center_lat is None:
                lons, lats = zip(*coordinates)
                center_lon = np.mean(lons) if center_lon is None else center_lon
                center_lat = np.mean(lats) if center_lat is None else center_lat
        
# Copyright (c) 2025 Mohamed Z. Hatim
        albers_proj = (f"+proj=aea +lat_1={center_lat - 10} +lat_2={center_lat + 10} "
                      f"+lat_0={center_lat} +lon_0={center_lon} +x_0=0 +y_0=0 "
                      f"+datum=WGS84 +units=m +no_defs")
        
        return self.transform_coordinates(coordinates, 'EPSG:4326', albers_proj)
    
    def get_crs_info(self, crs: str) -> Dict[str, str]:
        """
        Get information about a coordinate reference system.
        
        Parameters:
        -----------
        crs : str
            CRS identifier
            
        Returns:
        --------
        dict
            CRS information
        """
        if not PYPROJ_AVAILABLE:
            raise ImportError("PyProj required for CRS information")
        
        crs_resolved = self._resolve_crs(crs)
        crs_obj = CRS.from_string(crs_resolved)
        
        return {
            'name': crs_obj.name,
            'authority': crs_obj.to_authority(),
            'proj4': crs_obj.to_proj4(),
            'wkt': crs_obj.to_wkt(),
            'type': 'Geographic' if crs_obj.is_geographic else 'Projected',
            'units': crs_obj.axis_info[0].unit_name if crs_obj.axis_info else 'unknown'
        }
    
    def calculate_distances(self, 
                           coordinates: Union[pd.DataFrame, List[Tuple[float, float]]],
                           method: str = 'great_circle') -> np.ndarray:
        """
        Calculate distances between coordinate pairs.
        
        Parameters:
        -----------
        coordinates : pd.DataFrame or list of tuples
            Coordinate pairs
        method : str
            Distance calculation method ('great_circle', 'euclidean', 'geodesic')
            
        Returns:
        --------
        np.ndarray
            Distance matrix
        """
        if isinstance(coordinates, pd.DataFrame):
            coords = [(row['longitude'], row['latitude']) for _, row in coordinates.iterrows()]
        else:
            coords = coordinates
        
        n_points = len(coords)
        distances = np.zeros((n_points, n_points))
        
        if method == 'great_circle':
            for i in range(n_points):
                for j in range(i+1, n_points):
                    dist = self._great_circle_distance(coords[i], coords[j])
                    distances[i, j] = dist
                    distances[j, i] = dist
        
        elif method == 'euclidean':
# Copyright (c) 2025 Mohamed Z. Hatim
            projected_coords = self.reproject_to_equal_area(coords)
            
            for i in range(n_points):
                for j in range(i+1, n_points):
                    dist = np.sqrt((projected_coords[i][0] - projected_coords[j][0])**2 + 
                                 (projected_coords[i][1] - projected_coords[j][1])**2)
                    distances[i, j] = dist
                    distances[j, i] = dist
        
        elif method == 'geodesic':
            if not PYPROJ_AVAILABLE:
                raise ImportError("PyProj required for geodesic distances")
            
            geod = pyproj.Geod(ellps='WGS84')
            
            for i in range(n_points):
                for j in range(i+1, n_points):
                    lon1, lat1 = coords[i]
                    lon2, lat2 = coords[j]
                    _, _, dist = geod.inv(lon1, lat1, lon2, lat2)
                    distances[i, j] = dist
                    distances[j, i] = dist
        
        else:
            raise ValueError(f"Unknown distance method: {method}")
        
        return distances
    
    def _great_circle_distance(self, coord1: Tuple[float, float], 
                              coord2: Tuple[float, float]) -> float:
        """Calculate great circle distance between two points."""
        lon1, lat1 = np.radians(coord1)
        lon2, lat2 = np.radians(coord2)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
# Copyright (c) 2025 Mohamed Z. Hatim
        a = (np.sin(dlat/2)**2 + 
             np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2)
        c = 2 * np.arcsin(np.sqrt(a))
        
# Copyright (c) 2025 Mohamed Z. Hatim
        R = 6371000
        
        return R * c
    
    def create_spatial_grid(self, 
                           bounds: Tuple[float, float, float, float],
                           cell_size: float,
                           crs: str = 'EPSG:4326') -> pd.DataFrame:
        """
        Create a regular spatial grid.
        
        Parameters:
        -----------
        bounds : tuple
            Bounding box (min_x, min_y, max_x, max_y)
        cell_size : float
            Grid cell size in CRS units
        crs : str
            Coordinate reference system
            
        Returns:
        --------
        pd.DataFrame
            Grid cell centers
        """
        min_x, min_y, max_x, max_y = bounds
        
# Copyright (c) 2025 Mohamed Z. Hatim
        x_coords = np.arange(min_x + cell_size/2, max_x, cell_size)
        y_coords = np.arange(min_y + cell_size/2, max_y, cell_size)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        xx, yy = np.meshgrid(x_coords, y_coords)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        grid_df = pd.DataFrame({
            'grid_id': range(len(xx.flatten())),
            'x': xx.flatten(),
            'y': yy.flatten(),
            'crs': crs
        })
        
        return grid_df