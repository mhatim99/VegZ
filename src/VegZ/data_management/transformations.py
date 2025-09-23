"""
Data transformation methods for ecological analysis.

Copyright (c) 2025 Mohamed Z. Hatim
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, List
from scipy import stats
import warnings


class DataTransformer:
    """Main class for data transformations in ecological analysis."""
    
    def __init__(self):
        self.transformation_methods = {
            'hellinger': self.hellinger_transform,
            'chord': self.chord_transform,
            'log_chord': self.log_chord_transform,
            'wisconsin': self.wisconsin_transform,
            'chi_square': self.chi_square_transform,
            'log': self.log_transform,
            'sqrt': self.sqrt_transform,
            'arcsine': self.arcsine_transform,
            'standardize': self.standardize_transform,
            'normalize': self.normalize_transform
        }
    
    def transform(self, data: Union[pd.DataFrame, np.ndarray],
                  method: str = 'hellinger',
                  **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """
        Apply transformation to ecological data.
        
        Parameters:
        -----------
        data : pd.DataFrame or np.ndarray
            Species abundance/composition data
        method : str
            Transformation method
        **kwargs : dict
            Additional parameters for transformation
            
        Returns:
        --------
        pd.DataFrame or np.ndarray
            Transformed data
        """
        if method not in self.transformation_methods:
            raise ValueError(f"Unknown transformation method: {method}")
        
        transform_func = self.transformation_methods[method]
        return transform_func(data, **kwargs)
    
    def hellinger_transform(self, data: Union[pd.DataFrame, np.ndarray],
                           **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """
        Hellinger transformation for species abundance data.
        
        Formula: sqrt(p_ij / p_i+)
        where p_ij is the abundance of species j in sample i
        and p_i+ is the total abundance in sample i
        """
        if isinstance(data, pd.DataFrame):
# Copyright (c) 2025 Mohamed Z. Hatim
            numeric_data = data.select_dtypes(include=[np.number])
            transformed_data = data.copy()
        else:
            numeric_data = data.copy()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        numeric_data = np.maximum(numeric_data, 0)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        numeric_values = numeric_data.values if isinstance(numeric_data, pd.DataFrame) else numeric_data
        row_sums = np.sum(numeric_values, axis=1)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        row_sums[row_sums == 0] = 1
        
# Copyright (c) 2025 Mohamed Z. Hatim
        proportions = numeric_values / row_sums[:, np.newaxis]
        hellinger_data = np.sqrt(proportions)
        
        if isinstance(data, pd.DataFrame):
            transformed_data[numeric_data.columns] = hellinger_data
            return transformed_data
        else:
            return hellinger_data
    
    def chord_transform(self, data: Union[pd.DataFrame, np.ndarray],
                       **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """
        Chord transformation normalizes samples to unit length.
        
        Formula: x_ij / sqrt(sum(x_ij^2))
        """
        if isinstance(data, pd.DataFrame):
            numeric_data = data.select_dtypes(include=[np.number])
            transformed_data = data.copy()
        else:
            numeric_data = data.copy()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        numeric_data = np.maximum(numeric_data, 0)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        norms = np.sqrt(np.sum(numeric_data**2, axis=1))
        norms[norms == 0] = 1  # Avoid division by zero
        
# Copyright (c) 2025 Mohamed Z. Hatim
        chord_data = numeric_data / norms[:, np.newaxis]
        
        if isinstance(data, pd.DataFrame):
            transformed_data[numeric_data.columns] = chord_data
            return transformed_data
        else:
            return chord_data
    
    def log_chord_transform(self, data: Union[pd.DataFrame, np.ndarray],
                           **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """
        Log-chord transformation: log transform followed by chord transformation.
        """
# Copyright (c) 2025 Mohamed Z. Hatim
        log_data = self.log_transform(data, **kwargs)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        return self.chord_transform(log_data)
    
    def wisconsin_transform(self, data: Union[pd.DataFrame, np.ndarray],
                           **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """
        Wisconsin double standardization.
        
        First standardizes species (columns) to unit maxima,
        then standardizes samples (rows) to unit totals.
        """
        if isinstance(data, pd.DataFrame):
            numeric_data = data.select_dtypes(include=[np.number])
            transformed_data = data.copy()
        else:
            numeric_data = data.copy()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        numeric_data = np.maximum(numeric_data, 0)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        column_maxs = np.max(numeric_data, axis=0)
        column_maxs[column_maxs == 0] = 1  # Avoid division by zero
        species_std = numeric_data / column_maxs
        
# Copyright (c) 2025 Mohamed Z. Hatim
        species_values = species_std.values if isinstance(species_std, pd.DataFrame) else species_std
        row_sums = np.sum(species_values, axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        wisconsin_data = species_values / row_sums[:, np.newaxis]
        
        if isinstance(data, pd.DataFrame):
            transformed_data[numeric_data.columns] = wisconsin_data
            return transformed_data
        else:
            return wisconsin_data
    
    def chi_square_transform(self, data: Union[pd.DataFrame, np.ndarray],
                            **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """
        Chi-square transformation for correspondence analysis.
        
        Formula: sqrt(n * p_ij / (p_i+ * p_+j))
        """
        if isinstance(data, pd.DataFrame):
            numeric_data = data.select_dtypes(include=[np.number])
            transformed_data = data.copy()
        else:
            numeric_data = data.copy()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        numeric_data = np.maximum(numeric_data, 0)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        total_sum = np.sum(numeric_data)
        if total_sum == 0:
            warnings.warn("Total sum is zero, returning original data")
            return data
        
# Copyright (c) 2025 Mohamed Z. Hatim
        row_sums = np.sum(numeric_data, axis=1)
        col_sums = np.sum(numeric_data, axis=0)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        expected = np.outer(row_sums, col_sums) / total_sum
        expected[expected == 0] = 1  # Avoid division by zero
        
# Copyright (c) 2025 Mohamed Z. Hatim
        chi_sq_data = np.sqrt(numeric_data * total_sum / expected)
        
        if isinstance(data, pd.DataFrame):
            transformed_data[numeric_data.columns] = chi_sq_data
            return transformed_data
        else:
            return chi_sq_data
    
    def log_transform(self, data: Union[pd.DataFrame, np.ndarray],
                     base: str = 'natural',
                     constant: float = 1.0,
                     **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """
        Logarithmic transformation.
        
        Parameters:
        -----------
        base : str
            'natural' (ln), 'log10', or 'log2'
        constant : float
            Constant to add before transformation (log(x + c))
        """
        if isinstance(data, pd.DataFrame):
            numeric_data = data.select_dtypes(include=[np.number])
            transformed_data = data.copy()
        else:
            numeric_data = data.copy()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        adjusted_data = numeric_data + constant
        adjusted_data = np.maximum(adjusted_data, 1e-10)  # Avoid log(0)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if base == 'natural':
            log_data = np.log(adjusted_data)
        elif base == 'log10':
            log_data = np.log10(adjusted_data)
        elif base == 'log2':
            log_data = np.log2(adjusted_data)
        else:
            raise ValueError(f"Unknown logarithm base: {base}")
        
        if isinstance(data, pd.DataFrame):
            transformed_data[numeric_data.columns] = log_data
            return transformed_data
        else:
            return log_data
    
    def sqrt_transform(self, data: Union[pd.DataFrame, np.ndarray],
                       **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """Square root transformation."""
        if isinstance(data, pd.DataFrame):
            numeric_data = data.select_dtypes(include=[np.number])
            transformed_data = data.copy()
        else:
            numeric_data = data.copy()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        numeric_data = np.maximum(numeric_data, 0)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        sqrt_data = np.sqrt(numeric_data)
        
        if isinstance(data, pd.DataFrame):
            transformed_data[numeric_data.columns] = sqrt_data
            return transformed_data
        else:
            return sqrt_data
    
    def arcsine_transform(self, data: Union[pd.DataFrame, np.ndarray],
                         **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """
        Arcsine square root transformation for proportional data.
        
        Formula: arcsin(sqrt(p))
        """
        if isinstance(data, pd.DataFrame):
            numeric_data = data.select_dtypes(include=[np.number])
            transformed_data = data.copy()
        else:
            numeric_data = data.copy()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        numeric_data = np.clip(numeric_data, 0, 1)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        arcsine_data = np.arcsin(np.sqrt(numeric_data))
        
        if isinstance(data, pd.DataFrame):
            transformed_data[numeric_data.columns] = arcsine_data
            return transformed_data
        else:
            return arcsine_data
    
    def standardize_transform(self, data: Union[pd.DataFrame, np.ndarray],
                             method: str = 'zscore',
                             **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """
        Standardization transformations.
        
        Parameters:
        -----------
        method : str
            'zscore' (mean=0, std=1), 'robust' (median=0, MAD=1)
        """
        if isinstance(data, pd.DataFrame):
            numeric_data = data.select_dtypes(include=[np.number])
            transformed_data = data.copy()
        else:
            numeric_data = data.copy()
        
        if method == 'zscore':
# Copyright (c) 2025 Mohamed Z. Hatim
            means = np.mean(numeric_data, axis=0)
            stds = np.std(numeric_data, axis=0, ddof=1)
            stds[stds == 0] = 1  # Avoid division by zero
            standardized_data = (numeric_data - means) / stds
            
        elif method == 'robust':
# Copyright (c) 2025 Mohamed Z. Hatim
            medians = np.median(numeric_data, axis=0)
            mads = stats.median_abs_deviation(numeric_data, axis=0)
            mads[mads == 0] = 1  # Avoid division by zero
            standardized_data = (numeric_data - medians) / mads
            
        else:
            raise ValueError(f"Unknown standardization method: {method}")
        
        if isinstance(data, pd.DataFrame):
            transformed_data[numeric_data.columns] = standardized_data
            return transformed_data
        else:
            return standardized_data
    
    def normalize_transform(self, data: Union[pd.DataFrame, np.ndarray],
                           method: str = 'minmax',
                           **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """
        Normalization transformations.
        
        Parameters:
        -----------
        method : str
            'minmax' (0-1 scaling), 'unit_vector' (unit length)
        """
        if isinstance(data, pd.DataFrame):
            numeric_data = data.select_dtypes(include=[np.number])
            transformed_data = data.copy()
        else:
            numeric_data = data.copy()
        
        if method == 'minmax':
# Copyright (c) 2025 Mohamed Z. Hatim
            mins = np.min(numeric_data, axis=0)
            maxs = np.max(numeric_data, axis=0)
            ranges = maxs - mins
            ranges[ranges == 0] = 1  # Avoid division by zero
            normalized_data = (numeric_data - mins) / ranges
            
        elif method == 'unit_vector':
# Copyright (c) 2025 Mohamed Z. Hatim
            return self.chord_transform(data, **kwargs)
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        if isinstance(data, pd.DataFrame):
            transformed_data[numeric_data.columns] = normalized_data
            return transformed_data
        else:
            return normalized_data
    
    def inverse_transform(self, data: Union[pd.DataFrame, np.ndarray],
                         method: str,
                         original_params: Optional[dict] = None) -> Union[pd.DataFrame, np.ndarray]:
        """
        Apply inverse transformation (where possible).
        
        Parameters:
        -----------
        data : pd.DataFrame or np.ndarray
            Transformed data
        method : str
            Original transformation method
        original_params : dict, optional
            Parameters from original transformation
            
        Returns:
        --------
        pd.DataFrame or np.ndarray
            Inverse transformed data
        """
        if original_params is None:
            original_params = {}
        
        if method == 'log':
            base = original_params.get('base', 'natural')
            constant = original_params.get('constant', 1.0)
            
            if isinstance(data, pd.DataFrame):
                numeric_data = data.select_dtypes(include=[np.number])
                inverse_data = data.copy()
            else:
                numeric_data = data.copy()
            
            if base == 'natural':
                exp_data = np.exp(numeric_data) - constant
            elif base == 'log10':
                exp_data = np.power(10, numeric_data) - constant
            elif base == 'log2':
                exp_data = np.power(2, numeric_data) - constant
            else:
                raise ValueError(f"Unknown logarithm base: {base}")
            
            if isinstance(data, pd.DataFrame):
                inverse_data[numeric_data.columns] = exp_data
                return inverse_data
            else:
                return exp_data
        
        elif method == 'sqrt':
            if isinstance(data, pd.DataFrame):
                numeric_data = data.select_dtypes(include=[np.number])
                inverse_data = data.copy()
                inverse_data[numeric_data.columns] = np.square(numeric_data)
                return inverse_data
            else:
                return np.square(data)
        
        elif method == 'arcsine':
            if isinstance(data, pd.DataFrame):
                numeric_data = data.select_dtypes(include=[np.number])
                inverse_data = data.copy()
                inverse_data[numeric_data.columns] = np.square(np.sin(numeric_data))
                return inverse_data
            else:
                return np.square(np.sin(data))
        
        else:
            warnings.warn(f"Inverse transformation not implemented for method: {method}")
            return data