"""
Comprehensive spatial analysis module for vegetation mapping and landscape ecology.

Copyright (c) 2025 Mohamed Z. Hatim
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Tuple, Optional, Any
from scipy import stats, ndimage
from scipy.spatial import distance_matrix, ConvexHull, Voronoi
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata, RBFInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
import warnings

try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon, MultiPolygon
    from shapely.ops import unary_union
    import rasterio
    from rasterio.features import shapes, geometry_mask
    from rasterio.transform import from_bounds
    SPATIAL_LIBS_AVAILABLE = True
except ImportError:
    SPATIAL_LIBS_AVAILABLE = False
    warnings.warn("Spatial libraries not available. Install geopandas, rasterio for full functionality.")


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
        
        x_grid = np.arange(x_min - x_buffer, x_max + x_buffer, grid_resolution)
        y_grid = np.arange(y_min - y_buffer, y_max + y_buffer, grid_resolution)
        
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
        except:
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
    
    def _edge_density(self, binary_mask: np.ndarray, landscape: np.ndarray,
                     cell_size: float) -> float:
        """Calculate edge density (edge length per area)."""
# Copyright (c) 2025 Mohamed Z. Hatim
        edges = np.gradient(binary_mask.astype(float))
        edge_magnitude = np.sqrt(edges[0]**2 + edges[1]**2)
        total_edge = np.sum(edge_magnitude > 0) * cell_size
        
        landscape_area = binary_mask.size * (cell_size ** 2)
        return total_edge / landscape_area
    
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
        """Calculate landscape shape index."""
# Copyright (c) 2025 Mohamed Z. Hatim
        total_edge = self._edge_density(binary_mask, landscape, cell_size)
        total_area = np.sum(binary_mask) * (cell_size ** 2)
        
        if total_area == 0:
            return 0
        
# Copyright (c) 2025 Mohamed Z. Hatim
        min_edge = 4 * np.sqrt(total_area)  # Square perimeter
        return total_edge / min_edge if min_edge > 0 else 0
    
    def _contagion_index(self, binary_mask: np.ndarray, landscape: np.ndarray,
                        cell_size: float) -> float:
        """Calculate contagion index."""
# Copyright (c) 2025 Mohamed Z. Hatim
        patch_types = np.unique(landscape)
        n_types = len(patch_types)
        
        if n_types <= 1:
            return 100
        
# Copyright (c) 2025 Mohamed Z. Hatim
        adjacencies = np.zeros((n_types, n_types))
        
        for i in range(landscape.shape[0] - 1):
            for j in range(landscape.shape[1] - 1):
                type1 = landscape[i, j]
                type2 = landscape[i+1, j]
                type3 = landscape[i, j+1]
                
                idx1 = np.where(patch_types == type1)[0][0]
                idx2 = np.where(patch_types == type2)[0][0]
                idx3 = np.where(patch_types == type3)[0][0]
                
                adjacencies[idx1, idx2] += 1
                adjacencies[idx1, idx3] += 1
        
# Copyright (c) 2025 Mohamed Z. Hatim
        total_adjacencies = np.sum(adjacencies)
        if total_adjacencies == 0:
            return 0
        
        proportions = adjacencies / total_adjacencies
        contagion = 1 + np.sum(proportions * np.log(proportions + 1e-10)) / np.log(n_types)
        
        return contagion * 100
    
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
# Copyright (c) 2025 Mohamed Z. Hatim
            compactness_values = []
            
            for patch_id in range(1, n_patches + 1):
                patch = (labeled_patches == patch_id)
                patch_area = np.sum(patch)
                
# Copyright (c) 2025 Mohamed Z. Hatim
                patch_float = patch.astype(float)
                edges = np.gradient(patch_float)
                perimeter = np.sum(np.sqrt(edges[0]**2 + edges[1]**2) > 0)
                
                if perimeter > 0:
# Copyright (c) 2025 Mohamed Z. Hatim
                    compactness = (4 * np.pi * patch_area) / (perimeter ** 2)
                    compactness_values.append(compactness)
            
            metrics['mean_compactness'] = np.mean(compactness_values) if compactness_values else 0
        
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
    
    def _morans_i(self, points: np.ndarray, values: np.ndarray,
                 distance_threshold: Optional[float] = None) -> Dict[str, Any]:
        """Calculate Moran's I spatial autocorrelation."""
        n = len(points)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        distances = distance_matrix(points, points)
        
        if distance_threshold is None:
# Copyright (c) 2025 Mohamed Z. Hatim
            distance_threshold = np.mean(distances[distances > 0])
        
# Copyright (c) 2025 Mohamed Z. Hatim
        W = (distances <= distance_threshold) & (distances > 0)
        W = W.astype(float)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        row_sums = np.sum(W, axis=1)
        W[row_sums > 0] = W[row_sums > 0] / row_sums[row_sums > 0][:, np.newaxis]
        
# Copyright (c) 2025 Mohamed Z. Hatim
        mean_val = np.mean(values)
        deviations = values - mean_val
        
        numerator = np.sum(W * np.outer(deviations, deviations))
        denominator = np.sum(deviations ** 2)
        
        if denominator == 0:
            morans_i = 0
        else:
            morans_i = numerator / denominator
        
# Copyright (c) 2025 Mohamed Z. Hatim
        expected_i = -1 / (n - 1)
        
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
        z_score = (morans_i - expected_i) / np.sqrt(1 / n)  # Simplified standard error
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return {
            'morans_i': morans_i,
            'expected_i': expected_i,
            'z_score': z_score,
            'p_value': p_value,
            'distance_threshold': distance_threshold,
            'interpretation': 'positive' if morans_i > expected_i else 'negative' if morans_i < expected_i else 'random'
        }
    
    def _gearys_c(self, points: np.ndarray, values: np.ndarray,
                 distance_threshold: Optional[float] = None) -> Dict[str, Any]:
        """Calculate Geary's C spatial autocorrelation."""
        n = len(points)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        distances = distance_matrix(points, points)
        
        if distance_threshold is None:
            distance_threshold = np.mean(distances[distances > 0])
        
        W = (distances <= distance_threshold) & (distances > 0)
        W = W.astype(float)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        numerator = 0
        denominator = 0
        mean_val = np.mean(values)
        
        for i in range(n):
            for j in range(n):
                if W[i, j] > 0:
                    numerator += W[i, j] * (values[i] - values[j]) ** 2
                    denominator += W[i, j]
        
        variance = np.var(values)
        
        if denominator == 0 or variance == 0:
            gearys_c = 1
        else:
            gearys_c = numerator / (2 * denominator * variance)
        
        return {
            'gearys_c': gearys_c,
            'distance_threshold': distance_threshold,
            'interpretation': 'positive' if gearys_c < 1 else 'negative' if gearys_c > 1 else 'random'
        }
    
    def _calculate_variogram(self, points: np.ndarray, values: np.ndarray,
                           n_lags: int = 20) -> Dict[str, Any]:
        """Calculate empirical variogram."""
        distances = distance_matrix(points, points)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        triu_indices = np.triu_indices(len(points), k=1)
        dist_pairs = distances[triu_indices]
        value_pairs = np.array([(values[i], values[j]) for i, j in zip(triu_indices[0], triu_indices[1])])
        
# Copyright (c) 2025 Mohamed Z. Hatim
        squared_diffs = (value_pairs[:, 0] - value_pairs[:, 1]) ** 2
        
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