"""
Comprehensive spatial analysis module for vegetation mapping and landscape ecology.

Copyright (c) 2025 Mohamed Z. Hatim
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any
from scipy import stats, ndimage
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata, RBFInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import warnings

from ._compat import optional_import

# Optional geospatial stack. Imported lazily and *without* warning at import
# time: a library should not emit warnings just for being imported.
gpd = optional_import('geopandas')
rasterio = optional_import('rasterio')
SPATIAL_LIBS_AVAILABLE = gpd is not None and rasterio is not None


class SpatialAnalyzer:
    """Comprehensive spatial analysis for vegetation and landscape data."""
    
    def __init__(self):
        """Initialize spatial analyzer."""
        self.interpolation_methods = {
            'idw': self._inverse_distance_weighting,
            'kriging': self._simple_kriging,
            'rbf': self._radial_basis_function,
            'nearest': self._nearest_neighbor,
            'linear': self._linear_interpolation,
            'cubic': self._cubic_interpolation
        }
        
        self.landscape_metrics = {
            'patch_density': self._patch_density,
            'edge_density': self._edge_density,
            'mean_patch_size': self._mean_patch_size,
            'patch_size_cv': self._patch_size_coefficient_variation,
            'largest_patch_index': self._largest_patch_index,
            'landscape_shape_index': self._landscape_shape_index,
            'contagion': self._contagion_index,
            'shannon_diversity': self._landscape_shannon_diversity,
            'simpson_diversity': self._landscape_simpson_diversity,
            'evenness': self._landscape_evenness
        }
    
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
    
    def spatial_interpolation(self, data: pd.DataFrame,
                            x_col: str = 'longitude',
                            y_col: str = 'latitude', 
                            z_col: str = 'response',
                            method: str = 'idw',
                            grid_resolution: float = 0.01,
                            max_grid_cells: int = 1_000_000,
                            **kwargs) -> Dict[str, Any]:
        """
        Spatial interpolation of vegetation data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Point data with coordinates and response values
        x_col : str
            X coordinate column name
        y_col : str
            Y coordinate column name
        z_col : str
            Response variable column name
        method : str
            Interpolation method
        grid_resolution : float
            Grid cell resolution
        max_grid_cells : int
            Safety limit on the number of interpolation grid cells. A default
            resolution that suits degrees will silently allocate billions of
            cells for projected coordinates, so this fails fast instead.
        **kwargs
            Additional parameters for interpolation methods
            
        Returns:
        --------
        dict
            Interpolation results including grid and statistics
        """
# Copyright (c) 2025 Mohamed Z. Hatim
        clean_data = data.dropna(subset=[x_col, y_col, z_col])
        
        if len(clean_data) < 3:
            raise ValueError("Need at least 3 valid data points for interpolation")
        
# Copyright (c) 2025 Mohamed Z. Hatim
        points = clean_data[[x_col, y_col]].values
        values = clean_data[z_col].values
        
# Copyright (c) 2025 Mohamed Z. Hatim
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        x_buffer = (x_max - x_min) * 0.1
        y_buffer = (y_max - y_min) * 0.1
        
        if grid_resolution <= 0:
            raise ValueError("grid_resolution must be positive")

        x_grid = np.arange(x_min - x_buffer, x_max + x_buffer, grid_resolution)
        y_grid = np.arange(y_min - y_buffer, y_max + y_buffer, grid_resolution)

        n_cells = x_grid.size * y_grid.size
        if n_cells > max_grid_cells:
            raise ValueError(
                f"grid_resolution={grid_resolution} over this extent would "
                f"produce {n_cells:,} grid cells (limit {max_grid_cells:,}). "
                "Use a coarser resolution or raise max_grid_cells."
            )
        if n_cells == 0:
            raise ValueError("The requested grid is empty; check grid_resolution")

        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if method not in self.interpolation_methods:
            raise ValueError(f"Unknown interpolation method: {method}")
        
        interp_func = self.interpolation_methods[method]
        interpolated_values = interp_func(points, values, grid_points, **kwargs)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        Z_grid = interpolated_values.reshape(X_grid.shape)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        stats_dict = self._calculate_interpolation_stats(
            points, values, grid_points, interpolated_values, method
        )
        
        results = {
            'x_grid': x_grid,
            'y_grid': y_grid,
            'X_grid': X_grid,
            'Y_grid': Y_grid,
            'Z_grid': Z_grid,
            'interpolated_values': interpolated_values,
            'grid_points': grid_points,
            'original_points': points,
            'original_values': values,
            'method': method,
            'grid_resolution': grid_resolution,
            'statistics': stats_dict
        }
        
        return results
    
    def _inverse_distance_weighting(self, points: np.ndarray, values: np.ndarray,
                                  grid_points: np.ndarray, power: float = 2,
                                  **kwargs) -> np.ndarray:
        """Inverse distance weighting interpolation."""
        interpolated = np.zeros(len(grid_points))
        
        for i, grid_point in enumerate(grid_points):
            distances = np.sqrt(np.sum((points - grid_point)**2, axis=1))
            
# Copyright (c) 2025 Mohamed Z. Hatim
            if np.any(distances == 0):
                zero_idx = np.where(distances == 0)[0][0]
                interpolated[i] = values[zero_idx]
            else:
                weights = 1 / (distances ** power)
                interpolated[i] = np.sum(weights * values) / np.sum(weights)
        
        return interpolated
    
    def _simple_kriging(self, points: np.ndarray, values: np.ndarray,
                       grid_points: np.ndarray, variogram_model: str = 'gaussian',
                       **kwargs) -> np.ndarray:
        """Simple kriging interpolation (simplified implementation)."""
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
        
        try:
            from scipy.spatial.distance import pdist, squareform
            
# Copyright (c) 2025 Mohamed Z. Hatim
            point_distances = squareform(pdist(points))
            
# Copyright (c) 2025 Mohamed Z. Hatim
            nugget = kwargs.get('nugget', 0.1)
            sill = kwargs.get('sill', np.var(values))
            range_param = kwargs.get('range', np.max(point_distances) / 3)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            def variogram(h):
                if variogram_model == 'exponential':
                    return nugget + sill * (1 - np.exp(-h / range_param))
                elif variogram_model == 'gaussian':
                    return nugget + sill * (1 - np.exp(-(h**2) / (range_param**2)))
                else:  # spherical
                    h = np.minimum(h, range_param)
                    return nugget + sill * (1.5 * h / range_param - 0.5 * (h / range_param)**3)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            gamma = variogram(point_distances)
            cov_matrix = sill - gamma + np.eye(len(points)) * nugget
            
# Copyright (c) 2025 Mohamed Z. Hatim
            interpolated = np.zeros(len(grid_points))
            
            for i, grid_point in enumerate(grid_points):
                distances_to_grid = np.sqrt(np.sum((points - grid_point)**2, axis=1))
                gamma_grid = variogram(distances_to_grid)
                cov_grid = sill - gamma_grid
                
                try:
                    weights = np.linalg.solve(cov_matrix, cov_grid)
                    interpolated[i] = np.sum(weights * values)
                except np.linalg.LinAlgError:
# Copyright (c) 2025 Mohamed Z. Hatim
                    if np.any(distances_to_grid == 0):
                        zero_idx = np.where(distances_to_grid == 0)[0][0]
                        interpolated[i] = values[zero_idx]
                    else:
                        weights = 1 / (distances_to_grid ** 2)
                        interpolated[i] = np.sum(weights * values) / np.sum(weights)
            
            return interpolated
            
        except Exception as e:
            warnings.warn(f"Kriging failed, using IDW: {e}")
            return self._inverse_distance_weighting(points, values, grid_points)
    
    def _radial_basis_function(self, points: np.ndarray, values: np.ndarray,
                              grid_points: np.ndarray, function: str = 'thin_plate_spline',
                              **kwargs) -> np.ndarray:
        """Radial basis function interpolation."""
        try:
# Copyright (c) 2025 Mohamed Z. Hatim
            if function == 'thin_plate_spline':
                rbf = RBFInterpolator(points, values, kernel='thin_plate_spline')
            elif function == 'multiquadric':
                rbf = RBFInterpolator(points, values, kernel='multiquadric')
            elif function == 'gaussian':
                rbf = RBFInterpolator(points, values, kernel='gaussian')
            else:
                rbf = RBFInterpolator(points, values, kernel='linear')
            
            return rbf(grid_points)
            
        except Exception as e:
            warnings.warn(f"RBF interpolation failed, using IDW: {e}")
            return self._inverse_distance_weighting(points, values, grid_points)
    
    def _nearest_neighbor(self, points: np.ndarray, values: np.ndarray,
                         grid_points: np.ndarray, **kwargs) -> np.ndarray:
        """Nearest neighbor interpolation."""
        distances = cdist(grid_points, points)
        nearest_indices = np.argmin(distances, axis=1)
        return values[nearest_indices]
    
    def _linear_interpolation(self, points: np.ndarray, values: np.ndarray,
                             grid_points: np.ndarray, **kwargs) -> np.ndarray:
        """Linear interpolation using scipy.interpolate.griddata."""
        return griddata(points, values, grid_points, method='linear', fill_value=np.nan)
    
    def _cubic_interpolation(self, points: np.ndarray, values: np.ndarray,
                            grid_points: np.ndarray, **kwargs) -> np.ndarray:
        """Cubic interpolation using scipy.interpolate.griddata."""
        return griddata(points, values, grid_points, method='cubic', fill_value=np.nan)
    
    def _calculate_interpolation_stats(self, points: np.ndarray, values: np.ndarray,
                                     grid_points: np.ndarray, interpolated_values: np.ndarray,
                                     method: str) -> Dict[str, float]:
        """Calculate interpolation quality statistics."""
# Copyright (c) 2025 Mohamed Z. Hatim
        try:
            cv_errors = []
            for i in range(len(points)):
# Copyright (c) 2025 Mohamed Z. Hatim
                train_points = np.delete(points, i, axis=0)
                train_values = np.delete(values, i)
                test_point = points[i:i+1]
                test_value = values[i]
                
                if method in self.interpolation_methods:
                    interp_func = self.interpolation_methods[method]
                    predicted = interp_func(train_points, train_values, test_point)[0]
                    cv_errors.append((predicted - test_value)**2)
            
            cv_rmse = np.sqrt(np.mean(cv_errors)) if cv_errors else np.nan
        except (ValueError, np.linalg.LinAlgError, KeyError):
            # Leave-one-out can fail for degenerate point configurations;
            # report that rather than letting it abort the interpolation.
            cv_rmse = np.nan
        
# Copyright (c) 2025 Mohamed Z. Hatim
        stats_dict = {
            'cv_rmse': cv_rmse,
            'min_interpolated': np.nanmin(interpolated_values),
            'max_interpolated': np.nanmax(interpolated_values),
            'mean_interpolated': np.nanmean(interpolated_values),
            'std_interpolated': np.nanstd(interpolated_values),
            'n_grid_points': len(interpolated_values),
            'n_data_points': len(points),
            'method_used': method
        }
        
        return stats_dict
    
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
    
    def habitat_suitability_modeling(self, presence_data: pd.DataFrame,
                                   environmental_data: pd.DataFrame,
                                   x_col: str = 'longitude',
                                   y_col: str = 'latitude',
                                   response_col: str = 'presence',
                                   method: str = 'random_forest',
                                   **kwargs) -> Dict[str, Any]:
        """
        Habitat suitability modeling for species distribution.
        
        Parameters:
        -----------
        presence_data : pd.DataFrame
            Species presence/abundance data with coordinates
        environmental_data : pd.DataFrame
            Environmental predictor variables
        x_col, y_col : str
            Coordinate column names
        response_col : str
            Response variable (presence/absence or abundance)
        method : str
            Modeling method
        **kwargs
            Additional parameters
            
        Returns:
        --------
        dict
            Habitat suitability model results
        """
# Copyright (c) 2025 Mohamed Z. Hatim
        merged_data = pd.merge(presence_data, environmental_data, 
                              on=[x_col, y_col], how='inner')
        
# Copyright (c) 2025 Mohamed Z. Hatim
        env_cols = [col for col in environmental_data.columns 
                   if col not in [x_col, y_col]]
        X = merged_data[env_cols].dropna()
        y = merged_data[response_col].loc[X.index]
        
        if len(X) < 10:
            raise ValueError("Insufficient data points for modeling")
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if method == 'random_forest':
            model_results = self._fit_random_forest_hsm(X, y, **kwargs)
        elif method == 'glm':
            model_results = self._fit_glm_hsm(X, y, **kwargs)
        elif method == 'maxent':
            model_results = self._fit_maxent_hsm(X, y, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
# Copyright (c) 2025 Mohamed Z. Hatim
        var_importance = self._calculate_variable_importance(
            model_results['model'], X, y, method
        )
        
# Copyright (c) 2025 Mohamed Z. Hatim
        prediction_map = None
        if 'prediction_grid' in kwargs:
            prediction_map = self._generate_prediction_map(
                model_results['model'], kwargs['prediction_grid'], env_cols
            )
        
        results = {
            'model': model_results['model'],
            'performance_metrics': model_results['metrics'],
            'variable_importance': var_importance,
            'prediction_map': prediction_map,
            'environmental_variables': env_cols,
            'n_data_points': len(X),
            'method': method
        }
        
        return results
    
    def _fit_random_forest_hsm(self, X: pd.DataFrame, y: pd.Series,
                              n_estimators: int = 100, **kwargs) -> Dict[str, Any]:
        """Fit Random Forest habitat suitability model."""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, r2_score
        
# Copyright (c) 2025 Mohamed Z. Hatim
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if np.all(np.isin(y, [0, 1])):  # Binary classification
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import roc_auc_score
            
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_pred_proba),
                'cv_score': cross_val_score(model, X, y, cv=5).mean()
            }
        else:  # Regression
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            metrics = {
                'r2': r2_score(y_test, y_pred),
                'rmse': np.sqrt(np.mean((y_test - y_pred)**2)),
                'cv_score': cross_val_score(model, X, y, cv=5, 
                                          scoring='r2').mean()
            }
        
        return {
            'model': model,
            'metrics': metrics
        }
    
    def _fit_glm_hsm(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """Fit Generalized Linear Model for habitat suitability."""
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if np.all(np.isin(y, [0, 1])):  # Binary
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(max_iter=1000))
            ])
            
            pipeline.fit(X, y)
            y_pred_proba = pipeline.predict_proba(X)[:, 1]
            
            from sklearn.metrics import roc_auc_score
            metrics = {
                'auc': roc_auc_score(y, y_pred_proba),
                'cv_score': cross_val_score(pipeline, X, y, cv=5).mean()
            }
        else:  # Continuous
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', LinearRegression())
            ])
            
            pipeline.fit(X, y)
            
            metrics = {
                'r2': pipeline.score(X, y),
                'cv_score': cross_val_score(pipeline, X, y, cv=5,
                                          scoring='r2').mean()
            }
        
        return {
            'model': pipeline,
            'metrics': metrics
        }
    
    def _fit_maxent_hsm(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """Simplified MaxEnt-like model using logistic regression."""
# Copyright (c) 2025 Mohamed Z. Hatim
        warnings.warn("Using simplified MaxEnt (logistic regression)")
        return self._fit_glm_hsm(X, y, **kwargs)
    
    def _calculate_variable_importance(self, model, X: pd.DataFrame, y: pd.Series,
                                     method: str) -> pd.Series:
        """Calculate variable importance."""
        if method == 'random_forest':
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            else:
                importance = np.zeros(len(X.columns))
        else:
# Copyright (c) 2025 Mohamed Z. Hatim
            importance = np.zeros(len(X.columns))
            baseline_score = model.score(X, y)
            
            for i, col in enumerate(X.columns):
                X_permuted = X.copy()
                X_permuted[col] = np.random.permutation(X_permuted[col])
                permuted_score = model.score(X_permuted, y)
                importance[i] = baseline_score - permuted_score
        
        return pd.Series(importance, index=X.columns, name='importance').sort_values(ascending=False)
    
    def _generate_prediction_map(self, model, prediction_grid: pd.DataFrame,
                               env_cols: List[str]) -> np.ndarray:
        """Generate spatial prediction map."""
        grid_env = prediction_grid[env_cols]
        
        if hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(grid_env)[:, 1]
        else:
            predictions = model.predict(grid_env)
        
        return predictions
    
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
    
    def fragmentation_analysis(self, landscape_data: np.ndarray,
                             patch_types: Optional[List] = None,
                             cell_size: float = 1.0) -> Dict[str, Any]:
        """
        Analyze landscape fragmentation and calculate landscape metrics.
        
        Parameters:
        -----------
        landscape_data : np.ndarray
            2D array representing landscape with different patch types
        patch_types : list, optional
            List of patch type values to analyze
        cell_size : float
            Size of each cell in the landscape
            
        Returns:
        --------
        dict
            Fragmentation analysis results
        """
        if patch_types is None:
            patch_types = np.unique(landscape_data)
            patch_types = patch_types[patch_types != 0]  # Exclude background
        
        results = {}
        
        for patch_type in patch_types:
# Copyright (c) 2025 Mohamed Z. Hatim
            binary_mask = (landscape_data == patch_type).astype(int)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            metrics = {}
            for metric_name, metric_func in self.landscape_metrics.items():
                try:
                    metrics[metric_name] = metric_func(binary_mask, landscape_data, cell_size)
                except Exception as e:
                    warnings.warn(f"Error calculating {metric_name}: {e}")
                    metrics[metric_name] = np.nan
            
# Copyright (c) 2025 Mohamed Z. Hatim
            fragmentation_metrics = self._calculate_fragmentation_metrics(
                binary_mask, cell_size
            )
            metrics.update(fragmentation_metrics)
            
            results[f'patch_type_{patch_type}'] = metrics
        
# Copyright (c) 2025 Mohamed Z. Hatim
        landscape_metrics = self._calculate_landscape_level_metrics(
            landscape_data, patch_types, cell_size
        )
        results['landscape_level'] = landscape_metrics
        
        return results
    
    def _patch_density(self, binary_mask: np.ndarray, landscape: np.ndarray,
                      cell_size: float) -> float:
        """Calculate patch density (patches per 100 ha)."""
        labeled_patches, n_patches = ndimage.label(binary_mask)
        landscape_area = binary_mask.size * (cell_size ** 2)
        return (n_patches / landscape_area) * 10000  # per hectare
    
    @staticmethod
    def _total_edge_length(binary_mask: np.ndarray, cell_size: float,
                           count_boundary: bool = True) -> float:
        """
        Total edge length of a patch class, in linear units.

        Counts the cell *interfaces* between the class and its surroundings
        (each contributing one cell side). A gradient-magnitude proxy, which is
        what a naive implementation uses, counts cells rather than sides and so
        is neither a length nor correctly scaled.
        """
        mask = binary_mask.astype(bool)
        n_interfaces = 0

        # Vertical interfaces between horizontally adjacent cells.
        n_interfaces += int(np.sum(mask[:, :-1] != mask[:, 1:]))
        # Horizontal interfaces between vertically adjacent cells.
        n_interfaces += int(np.sum(mask[:-1, :] != mask[1:, :]))

        if count_boundary:
            # Cells of the class touching the edge of the raster.
            n_interfaces += int(mask[:, 0].sum() + mask[:, -1].sum()
                                + mask[0, :].sum() + mask[-1, :].sum())

        return n_interfaces * cell_size

    def _edge_density(self, binary_mask: np.ndarray, landscape: np.ndarray,
                      cell_size: float) -> float:
        """Edge density: total edge length per unit landscape area."""
        total_edge = self._total_edge_length(binary_mask, cell_size)
        landscape_area = binary_mask.size * (cell_size ** 2)
        return total_edge / landscape_area if landscape_area > 0 else 0.0
    
    def _mean_patch_size(self, binary_mask: np.ndarray, landscape: np.ndarray,
                        cell_size: float) -> float:
        """Calculate mean patch size."""
        labeled_patches, n_patches = ndimage.label(binary_mask)
        
        if n_patches == 0:
            return 0
        
        patch_sizes = []
        for patch_id in range(1, n_patches + 1):
            patch_area = np.sum(labeled_patches == patch_id) * (cell_size ** 2)
            patch_sizes.append(patch_area)
        
        return np.mean(patch_sizes)
    
    def _patch_size_coefficient_variation(self, binary_mask: np.ndarray,
                                        landscape: np.ndarray, cell_size: float) -> float:
        """Calculate coefficient of variation of patch sizes."""
        labeled_patches, n_patches = ndimage.label(binary_mask)
        
        if n_patches <= 1:
            return 0
        
        patch_sizes = []
        for patch_id in range(1, n_patches + 1):
            patch_area = np.sum(labeled_patches == patch_id) * (cell_size ** 2)
            patch_sizes.append(patch_area)
        
        return (np.std(patch_sizes) / np.mean(patch_sizes)) * 100
    
    def _largest_patch_index(self, binary_mask: np.ndarray, landscape: np.ndarray,
                            cell_size: float) -> float:
        """Calculate largest patch index (percentage of landscape)."""
        labeled_patches, n_patches = ndimage.label(binary_mask)
        
        if n_patches == 0:
            return 0
        
        patch_sizes = []
        for patch_id in range(1, n_patches + 1):
            patch_area = np.sum(labeled_patches == patch_id)
            patch_sizes.append(patch_area)
        
        largest_patch = max(patch_sizes) if patch_sizes else 0
        total_landscape = binary_mask.size
        
        return (largest_patch / total_landscape) * 100
    
    def _landscape_shape_index(self, binary_mask: np.ndarray, landscape: np.ndarray,
                               cell_size: float) -> float:
        """
        Landscape shape index: observed edge relative to the most compact shape
        of the same area (LSI = 1 for a perfect square, larger when convoluted).
        """
        # Must compare a *length* with a *length*; the previous version divided
        # an edge density by a perimeter, mixing units.
        total_edge = self._total_edge_length(binary_mask, cell_size)
        total_area = float(np.sum(binary_mask)) * (cell_size ** 2)

        if total_area <= 0:
            return 0.0

        min_edge = 4 * np.sqrt(total_area)  # perimeter of an equal-area square
        return total_edge / min_edge if min_edge > 0 else 0.0

    def _contagion_index(self, binary_mask: np.ndarray, landscape: np.ndarray,
                         cell_size: float) -> float:
        """
        Contagion index (O'Neill et al. 1988), as a percentage.

        ``CONTAG = [1 + sum_ij (P_i * q_ij) ln(P_i * q_ij) / (2 ln m)] * 100``
        where ``P_i`` is the proportion of the landscape in class i, ``q_ij``
        the proportion of class i's cell adjacencies that are with class j, and
        ``m`` the number of classes. Two details matter: the term is
        ``P_i * q_ij`` (not the raw joint adjacency proportion, which only
        coincides when all classes have equal area), and the denominator is
        ``2 ln m`` (with ``ln m`` the index is not bounded by 100).

        CONTAG is 100 for a landscape of a single class and approaches 0 when
        classes are equally abundant and randomly interspersed.
        """
        patch_types = np.unique(landscape)
        n_types = len(patch_types)

        if n_types <= 1:
            return 100.0

        # Map class values onto 0..m-1 so adjacencies can be tallied with bincount.
        lookup = {value: idx for idx, value in enumerate(patch_types)}
        coded = np.vectorize(lookup.get)(landscape).astype(int)

        def tally(a, b):
            flat = a.ravel() * n_types + b.ravel()
            counts = np.bincount(flat, minlength=n_types * n_types)
            return counts.reshape(n_types, n_types).astype(float)

        # All horizontal and vertical neighbour pairs, counted in both
        # directions so the adjacency matrix is symmetric.
        horizontal = tally(coded[:, :-1], coded[:, 1:])
        vertical = tally(coded[:-1, :], coded[1:, :])
        adjacencies = horizontal + horizontal.T + vertical + vertical.T

        if adjacencies.sum() == 0:
            return 0.0

        # P_i: proportion of the landscape in each class.
        class_counts = np.array([np.sum(coded == i) for i in range(n_types)], dtype=float)
        p_i = class_counts / class_counts.sum()

        # q_ij: proportion of class i's adjacencies that are with class j.
        row_totals = adjacencies.sum(axis=1)
        row_totals_safe = np.where(row_totals > 0, row_totals, 1.0)
        q_ij = adjacencies / row_totals_safe[:, None]

        terms = p_i[:, None] * q_ij
        nonzero = terms > 0
        entropy_term = float(np.sum(terms[nonzero] * np.log(terms[nonzero])))

        contagion = 1 + entropy_term / (2 * np.log(n_types))
        return float(np.clip(contagion, 0.0, 1.0) * 100)
    
    def _landscape_shannon_diversity(self, binary_mask: np.ndarray, landscape: np.ndarray,
                                   cell_size: float) -> float:
        """Calculate landscape Shannon diversity."""
        patch_types, counts = np.unique(landscape, return_counts=True)
        proportions = counts / np.sum(counts)
        
        return -np.sum(proportions * np.log(proportions + 1e-10))
    
    def _landscape_simpson_diversity(self, binary_mask: np.ndarray, landscape: np.ndarray,
                                   cell_size: float) -> float:
        """Calculate landscape Simpson diversity."""
        patch_types, counts = np.unique(landscape, return_counts=True)
        proportions = counts / np.sum(counts)
        
        return 1 - np.sum(proportions ** 2)
    
    def _landscape_evenness(self, binary_mask: np.ndarray, landscape: np.ndarray,
                           cell_size: float) -> float:
        """Calculate landscape evenness."""
        shannon_div = self._landscape_shannon_diversity(binary_mask, landscape, cell_size)
        n_types = len(np.unique(landscape))
        
        if n_types <= 1:
            return 1
        
        return shannon_div / np.log(n_types)
    
    def _calculate_fragmentation_metrics(self, binary_mask: np.ndarray,
                                       cell_size: float) -> Dict[str, float]:
        """Calculate additional fragmentation metrics."""
        labeled_patches, n_patches = ndimage.label(binary_mask)
        
        metrics = {
            'number_of_patches': n_patches,
            'total_area': np.sum(binary_mask) * (cell_size ** 2),
            'percentage_of_landscape': (np.sum(binary_mask) / binary_mask.size) * 100
        }
        
        if n_patches > 0:
            compactness_values = []

            for patch_id in range(1, n_patches + 1):
                patch = (labeled_patches == patch_id)
                patch_area = float(np.sum(patch)) * (cell_size ** 2)

                # True perimeter from cell interfaces, in the same linear units
                # as the area's square root, so the ratio is dimensionless.
                perimeter = self._total_edge_length(patch, cell_size)

                if perimeter > 0:
                    compactness = (4 * np.pi * patch_area) / (perimeter ** 2)
                    compactness_values.append(compactness)

            metrics['mean_compactness'] = (float(np.mean(compactness_values))
                                           if compactness_values else 0.0)

        return metrics
    
    def _calculate_landscape_level_metrics(self, landscape: np.ndarray,
                                         patch_types: List, cell_size: float) -> Dict[str, float]:
        """Calculate landscape-level metrics."""
        metrics = {
            'total_landscape_area': landscape.size * (cell_size ** 2),
            'number_of_patch_types': len(patch_types),
            'shannon_diversity_index': self._landscape_shannon_diversity(None, landscape, cell_size),
            'simpson_diversity_index': self._landscape_simpson_diversity(None, landscape, cell_size),
            'evenness_index': self._landscape_evenness(None, landscape, cell_size)
        }
        
        return metrics
    
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
    
    def spatial_autocorrelation(self, data: pd.DataFrame,
                              x_col: str = 'longitude',
                              y_col: str = 'latitude',
                              response_col: str = 'response',
                              method: str = 'morans_i') -> Dict[str, Any]:
        """
        Calculate spatial autocorrelation statistics.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Spatial data
        x_col, y_col : str
            Coordinate columns
        response_col : str
            Response variable
        method : str
            Autocorrelation method
            
        Returns:
        --------
        dict
            Spatial autocorrelation results
        """
        clean_data = data.dropna(subset=[x_col, y_col, response_col])
        
        if len(clean_data) < 3:
            raise ValueError("Need at least 3 points for spatial autocorrelation")
        
        points = clean_data[[x_col, y_col]].values
        values = clean_data[response_col].values
        
        if method == 'morans_i':
            return self._morans_i(points, values)
        elif method == 'gearys_c':
            return self._gearys_c(points, values)
        elif method == 'variogram':
            return self._calculate_variogram(points, values)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def _spatial_weights(points: np.ndarray,
                         distance_threshold: Optional[float],
                         row_standardize: bool):
        """Binary (optionally row-standardised) distance-band weights matrix."""
        distances = cdist(points, points)
        off_diagonal = distances > 0

        if distance_threshold is None:
            distance_threshold = float(np.mean(distances[off_diagonal])) \
                if off_diagonal.any() else 0.0

        W = ((distances <= distance_threshold) & off_diagonal).astype(float)

        if row_standardize:
            row_sums = W.sum(axis=1)
            nonzero = row_sums > 0
            W[nonzero] = W[nonzero] / row_sums[nonzero][:, np.newaxis]

        return W, distance_threshold

    def _morans_i(self, points: np.ndarray, values: np.ndarray,
                  distance_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Moran's I spatial autocorrelation with a randomisation-based z test.

        ``I = (n / S0) * sum_ij w_ij z_i z_j / sum_i z_i^2`` where
        ``S0 = sum_ij w_ij``. The ``n / S0`` factor is required whenever any
        site has no neighbours within the distance band (row standardisation
        alone does not make ``S0 = n``). The variance uses the standard
        randomisation-assumption formula rather than a placeholder.
        """
        n = len(points)
        W, distance_threshold = self._spatial_weights(points, distance_threshold, True)

        s0 = float(W.sum())
        deviations = np.asarray(values, dtype=float) - np.mean(values)
        denominator = float(np.sum(deviations ** 2))

        if denominator == 0 or s0 == 0 or n < 3:
            return {
                'morans_i': 0.0, 'expected_i': -1 / (n - 1) if n > 1 else np.nan,
                'variance': np.nan, 'z_score': 0.0, 'p_value': 1.0,
                'distance_threshold': distance_threshold, 's0': s0,
                'interpretation': 'random',
            }

        numerator = float(np.sum(W * np.outer(deviations, deviations)))
        morans_i = (n / s0) * (numerator / denominator)

        expected_i = -1.0 / (n - 1)
        variance = self._morans_variance(W, deviations, n, s0)

        if variance > 0:
            z_score = (morans_i - expected_i) / np.sqrt(variance)
            p_value = float(2 * stats.norm.sf(abs(z_score)))
        else:  # pragma: no cover - degenerate weights
            z_score, p_value = 0.0, 1.0

        return {
            'morans_i': float(morans_i),
            'expected_i': expected_i,
            'variance': variance,
            'z_score': float(z_score),
            'p_value': p_value,
            'distance_threshold': distance_threshold,
            's0': s0,
            'interpretation': ('positive' if morans_i > expected_i
                               else 'negative' if morans_i < expected_i else 'random')
        }

    @staticmethod
    def _morans_variance(W: np.ndarray, deviations: np.ndarray,
                         n: int, s0: float) -> float:
        """Variance of Moran's I under the randomisation assumption."""
        s1 = 0.5 * float(np.sum((W + W.T) ** 2))
        s2 = float(np.sum((W.sum(axis=1) + W.sum(axis=0)) ** 2))

        m2 = float(np.sum(deviations ** 2)) / n
        m4 = float(np.sum(deviations ** 4)) / n
        if m2 == 0:
            return 0.0
        b2 = m4 / (m2 ** 2)  # kurtosis

        a = n * ((n ** 2 - 3 * n + 3) * s1 - n * s2 + 3 * s0 ** 2)
        b = b2 * ((n ** 2 - n) * s1 - 2 * n * s2 + 6 * s0 ** 2)
        denom = (n - 1) * (n - 2) * (n - 3) * s0 ** 2
        if denom == 0:
            return 0.0

        e_i2 = (a - b) / denom
        return float(e_i2 - (-1.0 / (n - 1)) ** 2)

    def _gearys_c(self, points: np.ndarray, values: np.ndarray,
                  distance_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Geary's C spatial autocorrelation.

        ``C = ((n - 1) * sum_ij w_ij (x_i - x_j)^2) / (2 * S0 * sum_i z_i^2)``.
        The ``n - 1`` numerator (not ``n``) is what makes the expected value
        exactly 1 under no autocorrelation. Values below 1 indicate positive
        autocorrelation.
        """
        n = len(points)
        W, distance_threshold = self._spatial_weights(points, distance_threshold, False)

        values = np.asarray(values, dtype=float)
        deviations = values - values.mean()
        sum_sq_dev = float(np.sum(deviations ** 2))
        s0 = float(W.sum())

        if s0 == 0 or sum_sq_dev == 0 or n < 3:
            return {
                'gearys_c': 1.0, 'expected_c': 1.0, 'z_score': 0.0, 'p_value': 1.0,
                'distance_threshold': distance_threshold, 's0': s0,
                'interpretation': 'random',
            }

        squared_differences = (values[:, None] - values[None, :]) ** 2
        numerator = float(np.sum(W * squared_differences))

        gearys_c = ((n - 1) * numerator) / (2 * s0 * sum_sq_dev)

        # Normality-assumption variance (Cliff & Ord 1981).
        s1 = 0.5 * float(np.sum((W + W.T) ** 2))
        s2 = float(np.sum((W.sum(axis=1) + W.sum(axis=0)) ** 2))
        variance = ((2 * s1 + s2) * (n - 1) - 4 * s0 ** 2) / (2 * (n + 1) * s0 ** 2)

        if variance > 0:
            z_score = (gearys_c - 1.0) / np.sqrt(variance)
            p_value = float(2 * stats.norm.sf(abs(z_score)))
        else:  # pragma: no cover
            z_score, p_value = 0.0, 1.0

        return {
            'gearys_c': float(gearys_c),
            'expected_c': 1.0,
            'variance': float(variance),
            'z_score': float(z_score),
            'p_value': p_value,
            'distance_threshold': distance_threshold,
            's0': s0,
            'interpretation': ('positive' if gearys_c < 1
                               else 'negative' if gearys_c > 1 else 'random')
        }
    
    def _calculate_variogram(self, points: np.ndarray, values: np.ndarray,
                           n_lags: int = 20) -> Dict[str, Any]:
        """Calculate empirical variogram."""
        distances = cdist(points, points)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        triu_indices = np.triu_indices(len(points), k=1)
        dist_pairs = distances[triu_indices]
        squared_diffs = (values[triu_indices[0]] - values[triu_indices[1]]) ** 2
        
# Copyright (c) 2025 Mohamed Z. Hatim
        max_dist = np.max(dist_pairs)
        lag_bins = np.linspace(0, max_dist, n_lags + 1)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        lag_centers = []
        variogram_values = []
        n_pairs = []
        
        for i in range(n_lags):
            lag_mask = (dist_pairs >= lag_bins[i]) & (dist_pairs < lag_bins[i + 1])
            
            if np.sum(lag_mask) > 0:
                lag_center = (lag_bins[i] + lag_bins[i + 1]) / 2
                variogram_val = np.mean(squared_diffs[lag_mask]) / 2  # Semivariance
                n_pair = np.sum(lag_mask)
                
                lag_centers.append(lag_center)
                variogram_values.append(variogram_val)
                n_pairs.append(n_pair)
        
        return {
            'lag_centers': np.array(lag_centers),
            'variogram_values': np.array(variogram_values),
            'n_pairs': np.array(n_pairs),
            'max_distance': max_dist
        }