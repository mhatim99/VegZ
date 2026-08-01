"""
Remote sensing integration for vegetation indices and environmental data.

Copyright (c) 2025 Mohamed Z. Hatim
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings

from .._compat import optional_import

# Optional; resolved without warning at import time so that `import VegZ`
# stays quiet for users who never touch Earth Engine.
ee = optional_import('ee')
EE_AVAILABLE = ee is not None


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
                             indices: Optional[List[str]] = None,
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

        if indices is None:
            indices = ['NDVI', 'EVI', 'SAVI']

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
            # `point` is bound as a default argument: a bare closure over the
            # loop variable would make every mapped function use the *last*
            # point once the loop finishes.
            def extract_values(image, point=point):
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
    """
    Calculate vegetation indices from band reflectance values.

    All methods accept scalars, lists or NumPy arrays of any matching shape and
    return a float array (or a float scalar for scalar input). Bands must have
    identical shapes: silently broadcasting a mismatched pair produces a
    plausible-looking raster that is simply wrong.

    Division by zero yields ``fill`` (NaN by default) rather than an infinity,
    and the normalised-difference indices are clipped to their theoretical
    ``[-1, 1]`` range.
    """

    #: Indices available through :meth:`compute`, with the bands each needs.
    available_indices = {
        'ndvi': ('red', 'nir'),
        'evi': ('red', 'nir', 'blue'),
        'evi2': ('red', 'nir'),
        'savi': ('red', 'nir'),
        'msavi': ('red', 'nir'),
        'ndwi': ('green', 'nir'),
        'nbr': ('nir', 'swir'),
        'gndvi': ('green', 'nir'),
        'ndre': ('red_edge', 'nir'),
        'arvi': ('red', 'nir', 'blue'),
    }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _as_bands(**bands) -> Tuple[Dict[str, np.ndarray], bool]:
        """Coerce inputs to float arrays and check their shapes agree."""
        converted = {}
        for name, value in bands.items():
            array = np.asarray(value, dtype=float)
            converted[name] = array

        shapes = {name: array.shape for name, array in converted.items()}
        distinct = set(shapes.values())
        if len(distinct) > 1:
            raise ValueError(
                "All bands must have the same shape; got "
                + ', '.join(f'{name}={shape}' for name, shape in shapes.items())
            )

        scalar = next(iter(distinct)) == ()
        if scalar:
            converted = {name: np.atleast_1d(array)
                         for name, array in converted.items()}
        return converted, scalar

    @staticmethod
    def _finish(values: np.ndarray, scalar: bool, fill: float,
                clip: Optional[Tuple[float, float]] = None):
        """
        Clip valid results, substitute the fill value, and unwrap scalars.

        Clipping comes first so that a sentinel fill outside the index's own
        range (-999, say) survives: filling first would clip the sentinel back
        into range and make it indistinguishable from a real measurement. NaN
        passes through np.clip untouched, so valid pixels are unaffected.
        """
        if clip is not None:
            with np.errstate(invalid='ignore'):
                values = np.clip(values, clip[0], clip[1])
        values = np.where(np.isfinite(values), values, fill)
        return float(values[0]) if scalar else values

    @staticmethod
    def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        """Elementwise division with zero denominators mapped to NaN."""
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(denominator != 0, numerator / denominator, np.nan)

    @classmethod
    def compute(cls, index_name: str, fill: float = np.nan, **bands):
        """
        Calculate an index by name.

        Parameters
        ----------
        index_name : str
            One of :attr:`available_indices`.
        fill : float
            Value substituted where the index is undefined.
        **bands
            The bands the index requires, e.g. ``red=..., nir=...``.
        """
        key = index_name.lower()
        if key not in cls.available_indices:
            raise ValueError(
                f"Unknown index '{index_name}'. Available: "
                f"{', '.join(sorted(cls.available_indices))}"
            )

        required = cls.available_indices[key]
        missing = [band for band in required if band not in bands]
        if missing:
            raise ValueError(
                f"Index '{key}' requires the band(s) {missing}; "
                f"got {sorted(bands)}"
            )

        extras = {k: v for k, v in bands.items() if k not in required}
        return getattr(cls, key)(**{band: bands[band] for band in required},
                                 fill=fill, **extras)

    # ------------------------------------------------------------------
    # Indices
    # ------------------------------------------------------------------

    @classmethod
    def ndvi(cls, red, nir, fill: float = np.nan):
        """Normalized Difference Vegetation Index: (NIR - Red) / (NIR + Red)."""
        b, scalar = cls._as_bands(red=red, nir=nir)
        value = cls._ratio(b['nir'] - b['red'], b['nir'] + b['red'])
        return cls._finish(value, scalar, fill, clip=(-1.0, 1.0))

    @classmethod
    def evi(cls, red, nir, blue, G: float = 2.5, C1: float = 6.0,
            C2: float = 7.5, L: float = 1.0, fill: float = np.nan):
        """Enhanced Vegetation Index (Huete et al. 2002)."""
        b, scalar = cls._as_bands(red=red, nir=nir, blue=blue)
        denominator = b['nir'] + C1 * b['red'] - C2 * b['blue'] + L
        value = G * cls._ratio(b['nir'] - b['red'], denominator)
        return cls._finish(value, scalar, fill)

    @classmethod
    def evi2(cls, red, nir, G: float = 2.5, L: float = 1.0,
             C: float = 2.4, fill: float = np.nan):
        """Two-band EVI (Jiang et al. 2008), for sensors without a blue band."""
        b, scalar = cls._as_bands(red=red, nir=nir)
        value = G * cls._ratio(b['nir'] - b['red'], b['nir'] + C * b['red'] + L)
        return cls._finish(value, scalar, fill)

    @classmethod
    def savi(cls, red, nir, L: float = 0.5, fill: float = np.nan):
        """Soil Adjusted Vegetation Index (Huete 1988)."""
        b, scalar = cls._as_bands(red=red, nir=nir)
        value = cls._ratio(b['nir'] - b['red'], b['nir'] + b['red'] + L) * (1 + L)
        return cls._finish(value, scalar, fill)

    @classmethod
    def msavi(cls, red, nir, fill: float = np.nan):
        """
        Modified Soil Adjusted Vegetation Index (Qi et al. 1994).

        The discriminant can go negative for physically implausible band
        combinations; those pixels return ``fill`` rather than a spurious value
        from the square root of a negative number.
        """
        b, scalar = cls._as_bands(red=red, nir=nir)
        term = 2 * b['nir'] + 1
        discriminant = term ** 2 - 8 * (b['nir'] - b['red'])
        with np.errstate(invalid='ignore'):
            value = np.where(discriminant >= 0,
                             (term - np.sqrt(np.maximum(discriminant, 0))) / 2,
                             np.nan)
        return cls._finish(value, scalar, fill)

    @classmethod
    def ndwi(cls, green, nir, fill: float = np.nan):
        """Normalized Difference Water Index (McFeeters 1996)."""
        b, scalar = cls._as_bands(green=green, nir=nir)
        value = cls._ratio(b['green'] - b['nir'], b['green'] + b['nir'])
        return cls._finish(value, scalar, fill, clip=(-1.0, 1.0))

    @classmethod
    def nbr(cls, nir, swir, fill: float = np.nan):
        """Normalized Burn Ratio."""
        b, scalar = cls._as_bands(nir=nir, swir=swir)
        value = cls._ratio(b['nir'] - b['swir'], b['nir'] + b['swir'])
        return cls._finish(value, scalar, fill, clip=(-1.0, 1.0))

    @classmethod
    def gndvi(cls, green, nir, fill: float = np.nan):
        """Green NDVI - more sensitive to chlorophyll than NDVI."""
        b, scalar = cls._as_bands(green=green, nir=nir)
        value = cls._ratio(b['nir'] - b['green'], b['nir'] + b['green'])
        return cls._finish(value, scalar, fill, clip=(-1.0, 1.0))

    @classmethod
    def ndre(cls, red_edge, nir, fill: float = np.nan):
        """Normalized Difference Red Edge index."""
        b, scalar = cls._as_bands(red_edge=red_edge, nir=nir)
        value = cls._ratio(b['nir'] - b['red_edge'], b['nir'] + b['red_edge'])
        return cls._finish(value, scalar, fill, clip=(-1.0, 1.0))

    @classmethod
    def arvi(cls, red, nir, blue, gamma: float = 1.0, fill: float = np.nan):
        """Atmospherically Resistant Vegetation Index (Kaufman & Tanre 1992)."""
        b, scalar = cls._as_bands(red=red, nir=nir, blue=blue)
        corrected = b['red'] - gamma * (b['blue'] - b['red'])
        value = cls._ratio(b['nir'] - corrected, b['nir'] + corrected)
        return cls._finish(value, scalar, fill, clip=(-1.0, 1.0))
