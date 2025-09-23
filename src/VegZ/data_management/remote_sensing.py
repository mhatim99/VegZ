"""
Remote sensing integration for vegetation indices and environmental data.

Copyright (c) 2025 Mohamed Z. Hatim
"""

import numpy as np
import pandas as pd
import requests
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import json
import warnings

try:
    import ee
    EE_AVAILABLE = True
except ImportError:
    EE_AVAILABLE = False
    warnings.warn("Google Earth Engine not available. Install with: pip install earthengine-api")


class RemoteSensingAPI:
    """Main class for remote sensing data integration."""
    
    def __init__(self):
        self.apis = {
            'landsat': LandsatAPI(),
            'modis': MODISAPI(),
            'sentinel': SentinelAPI()
        }
        self.ee_initialized = False
        
        if EE_AVAILABLE:
            try:
                ee.Initialize()
                self.ee_initialized = True
            except Exception as e:
                self.ee_initialized = False
# Copyright (c) 2025 Mohamed Z. Hatim
                if "authentication" in str(e).lower() or "credentials" in str(e).lower():
                    warnings.warn(f"Earth Engine authentication required. Run 'earthengine authenticate' in your terminal.")
                else:
                    warnings.warn(f"Earth Engine initialization failed: {e}", category=UserWarning)
    
    def get_vegetation_indices(self, 
                             coordinates: List[Tuple[float, float]],
                             date_range: Tuple[str, str],
                             indices: List[str] = ['NDVI', 'EVI', 'SAVI'],
                             platform: str = 'landsat') -> pd.DataFrame:
        """
        Extract vegetation indices for given coordinates and date range.
        
        Parameters:
        -----------
        coordinates : list of tuples
            List of (latitude, longitude) pairs
        date_range : tuple
            Start and end dates as strings ('YYYY-MM-DD')
        indices : list
            Vegetation indices to calculate
        platform : str
            Satellite platform ('landsat', 'modis', 'sentinel')
            
        Returns:
        --------
        pd.DataFrame
            Vegetation indices data
        """
        if platform not in self.apis:
            raise ValueError(f"Unsupported platform: {platform}")
        
        api = self.apis[platform]
        return api.extract_indices(coordinates, date_range, indices)


class LandsatAPI:
    """Landsat data API interface."""
    
    def __init__(self):
        self.collection_id = 'LANDSAT/LC08/C02/T1_TOA'
        self.available_indices = ['NDVI', 'EVI', 'SAVI', 'MSAVI', 'NDWI', 'NBR']
    
    def extract_indices(self, 
                       coordinates: List[Tuple[float, float]],
                       date_range: Tuple[str, str],
                       indices: List[str]) -> pd.DataFrame:
        """Extract Landsat-based vegetation indices."""
        if not EE_AVAILABLE:
            raise ImportError("Google Earth Engine required for Landsat data")
        
        results = []
        
        for i, (lat, lon) in enumerate(coordinates):
            point = ee.Geometry.Point([lon, lat])
            
# Copyright (c) 2025 Mohamed Z. Hatim
            collection = (ee.ImageCollection(self.collection_id)
                         .filterBounds(point)
                         .filterDate(date_range[0], date_range[1])
                         .filter(ee.Filter.lt('CLOUD_COVER', 20)))
            
            if collection.size().getInfo() == 0:
                continue
            
# Copyright (c) 2025 Mohamed Z. Hatim
            image = collection.median()
            
# Copyright (c) 2025 Mohamed Z. Hatim
            for index in indices:
                if index in self.available_indices:
                    index_image = self._calculate_index(image, index)
                    value = index_image.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=point,
                        scale=30
                    ).getInfo()
                    
                    results.append({
                        'point_id': i,
                        'latitude': lat,
                        'longitude': lon,
                        'index': index,
                        'value': value.get(index, np.nan),
                        'platform': 'landsat',
                        'date_range': f"{date_range[0]} to {date_range[1]}"
                    })
        
        return pd.DataFrame(results)
    
    def _calculate_index(self, image, index_name: str):
        """Calculate specific vegetation index."""
        if index_name == 'NDVI':
            return image.normalizedDifference(['B5', 'B4']).rename('NDVI')
        elif index_name == 'EVI':
            return image.expression(
                '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
                {
                    'NIR': image.select('B5'),
                    'RED': image.select('B4'),
                    'BLUE': image.select('B2')
                }
            ).rename('EVI')
        elif index_name == 'SAVI':
            L = 0.5  # Soil adjustment factor
            return image.expression(
                '((NIR - RED) / (NIR + RED + L)) * (1 + L)',
                {
                    'NIR': image.select('B5'),
                    'RED': image.select('B4'),
                    'L': L
                }
            ).rename('SAVI')
        elif index_name == 'MSAVI':
            return image.expression(
                '(2 * NIR + 1 - sqrt(pow((2 * NIR + 1), 2) - 8 * (NIR - RED))) / 2',
                {
                    'NIR': image.select('B5'),
                    'RED': image.select('B4')
                }
            ).rename('MSAVI')
        elif index_name == 'NDWI':
            return image.normalizedDifference(['B3', 'B5']).rename('NDWI')
        elif index_name == 'NBR':
            return image.normalizedDifference(['B5', 'B7']).rename('NBR')
        else:
            raise ValueError(f"Unknown index: {index_name}")


class MODISAPI:
    """MODIS data API interface."""
    
    def __init__(self):
        self.collection_id = 'MODIS/006/MOD13Q1'
        self.available_indices = ['NDVI', 'EVI']
    
    def extract_indices(self, 
                       coordinates: List[Tuple[float, float]],
                       date_range: Tuple[str, str],
                       indices: List[str]) -> pd.DataFrame:
        """Extract MODIS vegetation indices."""
        if not EE_AVAILABLE:
            raise ImportError("Google Earth Engine required for MODIS data")
        
        results = []
        
        for i, (lat, lon) in enumerate(coordinates):
            point = ee.Geometry.Point([lon, lat])
            
# Copyright (c) 2025 Mohamed Z. Hatim
            collection = (ee.ImageCollection(self.collection_id)
                         .filterBounds(point)
                         .filterDate(date_range[0], date_range[1]))
            
            if collection.size().getInfo() == 0:
                continue
            
# Copyright (c) 2025 Mohamed Z. Hatim
            def extract_values(image):
                date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
                values = image.select(['NDVI', 'EVI']).reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=point,
                    scale=250
                )
                return ee.Feature(None, values.set('date', date))
            
            features = collection.map(extract_values)
            feature_list = features.getInfo()['features']
            
            for feature in feature_list:
                props = feature['properties']
                for index in indices:
                    if index in self.available_indices:
                        value = props.get(index)
                        if value is not None:
                            results.append({
                                'point_id': i,
                                'latitude': lat,
                                'longitude': lon,
                                'index': index,
                                'value': value * 0.0001,  # Scale factor
                                'platform': 'modis',
                                'date': props['date']
                            })
        
        return pd.DataFrame(results)


class SentinelAPI:
    """Sentinel-2 data API interface."""
    
    def __init__(self):
        self.collection_id = 'COPERNICUS/S2_SR'
        self.available_indices = ['NDVI', 'EVI', 'SAVI', 'NDWI', 'NBR']
    
    def extract_indices(self, 
                       coordinates: List[Tuple[float, float]],
                       date_range: Tuple[str, str],
                       indices: List[str]) -> pd.DataFrame:
        """Extract Sentinel-2 vegetation indices."""
        if not EE_AVAILABLE:
            raise ImportError("Google Earth Engine required for Sentinel data")
        
        results = []
        
        for i, (lat, lon) in enumerate(coordinates):
            point = ee.Geometry.Point([lon, lat])
            
# Copyright (c) 2025 Mohamed Z. Hatim
            collection = (ee.ImageCollection(self.collection_id)
                         .filterBounds(point)
                         .filterDate(date_range[0], date_range[1])
                         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
            
            if collection.size().getInfo() == 0:
                continue
            
# Copyright (c) 2025 Mohamed Z. Hatim
            image = collection.median()
            
# Copyright (c) 2025 Mohamed Z. Hatim
            for index in indices:
                if index in self.available_indices:
                    index_image = self._calculate_index(image, index)
                    value = index_image.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=point,
                        scale=10
                    ).getInfo()
                    
                    results.append({
                        'point_id': i,
                        'latitude': lat,
                        'longitude': lon,
                        'index': index,
                        'value': value.get(index, np.nan),
                        'platform': 'sentinel2',
                        'date_range': f"{date_range[0]} to {date_range[1]}"
                    })
        
        return pd.DataFrame(results)
    
    def _calculate_index(self, image, index_name: str):
        """Calculate specific vegetation index for Sentinel-2."""
        if index_name == 'NDVI':
            return image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        elif index_name == 'EVI':
            return image.expression(
                '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
                {
                    'NIR': image.select('B8'),
                    'RED': image.select('B4'),
                    'BLUE': image.select('B2')
                }
            ).rename('EVI')
        elif index_name == 'SAVI':
            L = 0.5
            return image.expression(
                '((NIR - RED) / (NIR + RED + L)) * (1 + L)',
                {
                    'NIR': image.select('B8'),
                    'RED': image.select('B4'),
                    'L': L
                }
            ).rename('SAVI')
        elif index_name == 'NDWI':
            return image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        elif index_name == 'NBR':
            return image.normalizedDifference(['B8', 'B12']).rename('NBR')
        else:
            raise ValueError(f"Unknown index: {index_name}")


class VegetationIndexCalculator:
    """Calculate vegetation indices from band data."""
    
    @staticmethod
    def ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Calculate Normalized Difference Vegetation Index."""
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = (nir - red) / (nir + red)
            ndvi[np.isnan(ndvi)] = 0
            return ndvi
    
    @staticmethod
    def evi(red: np.ndarray, nir: np.ndarray, blue: np.ndarray, 
            G: float = 2.5, C1: float = 6.0, C2: float = 7.5, L: float = 1.0) -> np.ndarray:
        """Calculate Enhanced Vegetation Index."""
        with np.errstate(divide='ignore', invalid='ignore'):
            evi = G * ((nir - red) / (nir + C1 * red - C2 * blue + L))
            evi[np.isnan(evi)] = 0
            return evi
    
    @staticmethod
    def savi(red: np.ndarray, nir: np.ndarray, L: float = 0.5) -> np.ndarray:
        """Calculate Soil Adjusted Vegetation Index."""
        with np.errstate(divide='ignore', invalid='ignore'):
            savi = ((nir - red) / (nir + red + L)) * (1 + L)
            savi[np.isnan(savi)] = 0
            return savi
    
    @staticmethod
    def msavi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Calculate Modified Soil Adjusted Vegetation Index."""
        with np.errstate(divide='ignore', invalid='ignore'):
            msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1)**2 - 8 * (nir - red))) / 2
            msavi[np.isnan(msavi)] = 0
            return msavi
    
    @staticmethod
    def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Calculate Normalized Difference Water Index."""
        with np.errstate(divide='ignore', invalid='ignore'):
            ndwi = (green - nir) / (green + nir)
            ndwi[np.isnan(ndwi)] = 0
            return ndwi
    
    @staticmethod
    def nbr(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
        """Calculate Normalized Burn Ratio."""
        with np.errstate(divide='ignore', invalid='ignore'):
            nbr = (nir - swir) / (nir + swir)
            nbr[np.isnan(nbr)] = 0
            return nbr