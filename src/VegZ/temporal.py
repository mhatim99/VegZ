"""
Comprehensive temporal analysis module for vegetation and ecological data.

Copyright (c) 2025 Mohamed Z. Hatim
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Tuple, Optional, Any, Callable
from scipy import stats, optimize, signal
from scipy.interpolate import UnivariateSpline, interp1d
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import warnings
from datetime import datetime, timedelta


class TemporalAnalyzer:
    """Comprehensive temporal analysis for ecological time series data."""
    
    def __init__(self):
        """Initialize temporal analyzer."""
        self.phenology_models = {
            'sigmoid': self._sigmoid_model,
            'double_sigmoid': self._double_sigmoid_model,
            'gaussian': self._gaussian_model,
            'beta': self._beta_model,
            'weibull': self._weibull_model
        }
        
        self.trend_methods = {
            'linear': self._linear_trend,
            'polynomial': self._polynomial_trend,
            'spline': self._spline_trend,
            'lowess': self._lowess_trend,
            'mann_kendall': self._mann_kendall_trend
        }
        
        self.decomposition_methods = {
            'classical': self._classical_decomposition,
            'stl': self._stl_decomposition,
            'x11': self._x11_decomposition
        }
    
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
    
    def phenology_modeling(self, data: pd.DataFrame,
                          time_col: str = 'date',
                          response_col: str = 'response',
                          model_type: str = 'sigmoid',
                          species_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Model phenological patterns for species or communities.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Time series data with date and response variables
        time_col : str
            Name of time/date column
        response_col : str
            Name of response variable (e.g., flowering, leaf emergence)
        model_type : str
            Type of phenological model to fit
        species_col : str, optional
            Column for grouping by species
            
        Returns:
        --------
        dict
            Phenology modeling results
        """
        if time_col not in data.columns or response_col not in data.columns:
            raise ValueError(f"Required columns {time_col} and {response_col} not found")
        
# Copyright (c) 2025 Mohamed Z. Hatim
        data_clean = data.copy()
        if not pd.api.types.is_numeric_dtype(data_clean[time_col]):
            dates = pd.to_datetime(data_clean[time_col])
            data_clean['day_of_year'] = dates.dt.dayofyear
            time_var = 'day_of_year'
        else:
            time_var = time_col
        
        results = {}
        
        if species_col and species_col in data.columns:
# Copyright (c) 2025 Mohamed Z. Hatim
            for species in data_clean[species_col].unique():
                species_data = data_clean[data_clean[species_col] == species]
                species_results = self._fit_phenology_model(
                    species_data[time_var].values,
                    species_data[response_col].values,
                    model_type
                )
                results[species] = species_results
        else:
# Copyright (c) 2025 Mohamed Z. Hatim
            results['combined'] = self._fit_phenology_model(
                data_clean[time_var].values,
                data_clean[response_col].values,
                model_type
            )
        
        return {
            'model_type': model_type,
            'results': results,
            'data_summary': {
                'n_observations': len(data_clean),
                'time_range': (data_clean[time_var].min(), data_clean[time_var].max()),
                'response_range': (data_clean[response_col].min(), data_clean[response_col].max())
            }
        }
    
    def _fit_phenology_model(self, x: np.ndarray, y: np.ndarray, 
                           model_type: str) -> Dict[str, Any]:
        """Fit individual phenology model."""
        if model_type not in self.phenology_models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_func = self.phenology_models[model_type]
        
        try:
# Copyright (c) 2025 Mohamed Z. Hatim
            popt, pcov = optimize.curve_fit(
                model_func, x, y,
                maxfev=5000,
                bounds=self._get_model_bounds(model_type, x, y)
            )
            
# Copyright (c) 2025 Mohamed Z. Hatim
            y_pred = model_func(x, *popt)
            r_squared = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            
# Copyright (c) 2025 Mohamed Z. Hatim
            param_errors = np.sqrt(np.diag(pcov))
            
# Copyright (c) 2025 Mohamed Z. Hatim
            pheno_params = self._extract_phenology_parameters(model_type, popt, x)
            
            results = {
                'parameters': popt,
                'parameter_errors': param_errors,
                'covariance_matrix': pcov,
                'r_squared': r_squared,
                'rmse': rmse,
                'predicted_values': y_pred,
                'phenological_parameters': pheno_params,
                'model_function': model_func,
                'success': True
            }
            
        except Exception as e:
            warnings.warn(f"Model fitting failed: {e}")
            results = {
                'parameters': None,
                'error': str(e),
                'success': False
            }
        
        return results
    
    def _sigmoid_model(self, x: np.ndarray, a: float, b: float, 
                      c: float, d: float) -> np.ndarray:
        """Sigmoid (logistic) phenology model."""
        return a / (1 + np.exp(-b * (x - c))) + d
    
    def _double_sigmoid_model(self, x: np.ndarray, a1: float, b1: float, c1: float,
                             a2: float, b2: float, c2: float, d: float) -> np.ndarray:
        """Double sigmoid model for growing season."""
        rise = a1 / (1 + np.exp(-b1 * (x - c1)))
        fall = a2 / (1 + np.exp(b2 * (x - c2)))
        return rise - fall + d
    
    def _gaussian_model(self, x: np.ndarray, a: float, mu: float, 
                       sigma: float, d: float) -> np.ndarray:
        """Gaussian phenology model."""
        return a * np.exp(-0.5 * ((x - mu) / sigma)**2) + d
    
    def _beta_model(self, x: np.ndarray, a: float, alpha: float, 
                   beta: float, t1: float, t2: float) -> np.ndarray:
        """Beta function phenology model."""
# Copyright (c) 2025 Mohamed Z. Hatim
        x_norm = (x - t1) / (t2 - t1)
        x_norm = np.clip(x_norm, 0, 1)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        return a * (x_norm**(alpha-1)) * ((1-x_norm)**(beta-1))
    
    def _weibull_model(self, x: np.ndarray, a: float, k: float, 
                      lambda_: float, d: float) -> np.ndarray:
        """Weibull phenology model."""
        return a * (k / lambda_) * ((x / lambda_)**(k-1)) * np.exp(-((x / lambda_)**k)) + d
    
    def _get_model_bounds(self, model_type: str, x: np.ndarray, 
                         y: np.ndarray) -> Tuple[List, List]:
        """Get parameter bounds for optimization."""
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        y_range = y_max - y_min
        
        if model_type == 'sigmoid':
            lower = [0, 0, x_min, y_min]
            upper = [y_range * 2, 1, x_max, y_max]
        elif model_type == 'double_sigmoid':
            lower = [0, 0, x_min, 0, 0, x_min, y_min]
            upper = [y_range * 2, 1, x_max, y_range * 2, 1, x_max, y_max]
        elif model_type == 'gaussian':
            lower = [0, x_min, 1, y_min]
            upper = [y_range * 2, x_max, (x_max - x_min), y_max]
        elif model_type == 'beta':
            lower = [0, 0.1, 0.1, x_min, x_min]
            upper = [y_range * 2, 10, 10, x_max, x_max]
        elif model_type == 'weibull':
            lower = [0, 0.1, 1, y_min]
            upper = [y_range * 2, 10, (x_max - x_min), y_max]
        else:
# Copyright (c) 2025 Mohamed Z. Hatim
            lower = [-np.inf] * 4
            upper = [np.inf] * 4
        
        return lower, upper
    
    def _extract_phenology_parameters(self, model_type: str, params: np.ndarray,
                                    x: np.ndarray) -> Dict[str, float]:
        """Extract meaningful phenological parameters."""
        pheno_params = {}
        
        if model_type == 'sigmoid':
            a, b, c, d = params
            pheno_params['onset'] = c
            pheno_params['amplitude'] = a
            pheno_params['rate'] = b
            pheno_params['baseline'] = d
            
        elif model_type == 'double_sigmoid':
            a1, b1, c1, a2, b2, c2, d = params
            pheno_params['green_up'] = c1
            pheno_params['senescence'] = c2
            pheno_params['peak_amplitude'] = a1 - a2 + d
            pheno_params['growing_season_length'] = c2 - c1
            
        elif model_type == 'gaussian':
            a, mu, sigma, d = params
            pheno_params['peak_time'] = mu
            pheno_params['peak_amplitude'] = a + d
            pheno_params['duration'] = 2 * sigma
            pheno_params['baseline'] = d
            
        elif model_type == 'beta':
            a, alpha, beta_param, t1, t2 = params
# Copyright (c) 2025 Mohamed Z. Hatim
            if alpha > 1 and beta_param > 1:
                peak_norm = (alpha - 1) / (alpha + beta_param - 2)
                pheno_params['peak_time'] = t1 + peak_norm * (t2 - t1)
            pheno_params['season_start'] = t1
            pheno_params['season_end'] = t2
            pheno_params['peak_amplitude'] = a
            
        return pheno_params
    
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
    
    def trend_detection(self, data: pd.DataFrame,
                       time_col: str = 'date',
                       response_col: str = 'response',
                       method: str = 'linear',
                       **kwargs) -> Dict[str, Any]:
        """
        Detect trends in ecological time series.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Time series data
        time_col : str
            Time column name
        response_col : str
            Response variable column
        method : str
            Trend detection method
        **kwargs
            Additional parameters for specific methods
            
        Returns:
        --------
        dict
            Trend analysis results
        """
        if method not in self.trend_methods:
            raise ValueError(f"Unknown trend method: {method}")
        
# Copyright (c) 2025 Mohamed Z. Hatim
        data_clean = data.dropna(subset=[time_col, response_col])
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if not pd.api.types.is_numeric_dtype(data_clean[time_col]):
            time_numeric = pd.to_datetime(data_clean[time_col]).map(pd.Timestamp.toordinal)
        else:
            time_numeric = data_clean[time_col]
        
        x = time_numeric.values
        y = data_clean[response_col].values
        
# Copyright (c) 2025 Mohamed Z. Hatim
        trend_func = self.trend_methods[method]
        results = trend_func(x, y, **kwargs)
        
        results.update({
            'method': method,
            'data_summary': {
                'n_observations': len(data_clean),
                'time_span': x.max() - x.min(),
                'response_range': (y.min(), y.max())
            }
        })
        
        return results
    
    def _linear_trend(self, x: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Linear trend analysis."""
# Copyright (c) 2025 Mohamed Z. Hatim
        reg = LinearRegression()
        X = x.reshape(-1, 1)
        reg.fit(X, y)
        
        y_pred = reg.predict(X)
        slope = reg.coef_[0]
        intercept = reg.intercept_
        
# Copyright (c) 2025 Mohamed Z. Hatim
        r_squared = reg.score(X, y)
        n = len(y)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        residuals = y - y_pred
        mse = np.sum(residuals**2) / (n - 2)
        se_slope = np.sqrt(mse / np.sum((x - x.mean())**2))
        t_stat = slope / se_slope
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'p_value': p_value,
            't_statistic': t_stat,
            'predicted_values': y_pred,
            'residuals': residuals,
            'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'no trend',
            'trend_magnitude': abs(slope)
        }
    
    def _polynomial_trend(self, x: np.ndarray, y: np.ndarray, 
                         degree: int = 2, **kwargs) -> Dict[str, Any]:
        """Polynomial trend analysis."""
# Copyright (c) 2025 Mohamed Z. Hatim
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(x.reshape(-1, 1))
        
        reg = LinearRegression()
        reg.fit(X_poly, y)
        
        y_pred = reg.predict(X_poly)
        r_squared = reg.score(X_poly, y)
        
        return {
            'coefficients': reg.coef_,
            'intercept': reg.intercept_,
            'degree': degree,
            'r_squared': r_squared,
            'predicted_values': y_pred,
            'polynomial_features': poly_features
        }
    
    def _spline_trend(self, x: np.ndarray, y: np.ndarray, 
                     smoothing: float = None, **kwargs) -> Dict[str, Any]:
        """Spline trend analysis."""
# Copyright (c) 2025 Mohamed Z. Hatim
        if smoothing is None:
            smoothing = len(x)
        
        spline = UnivariateSpline(x, y, s=smoothing)
        y_pred = spline(x)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        x_fine = np.linspace(x.min(), x.max(), len(x) * 10)
        y_fine = spline(x_fine)
        derivatives = spline.derivative()(x_fine)
        
        return {
            'spline_object': spline,
            'predicted_values': y_pred,
            'smoothing_parameter': smoothing,
            'derivatives': derivatives,
            'fine_x': x_fine,
            'fine_y': y_fine,
            'mean_derivative': np.mean(derivatives),
            'trend_direction': 'increasing' if np.mean(derivatives) > 0 else 'decreasing' if np.mean(derivatives) < 0 else 'no trend'
        }
    
    def _lowess_trend(self, x: np.ndarray, y: np.ndarray, 
                     frac: float = 0.3, **kwargs) -> Dict[str, Any]:
        """LOWESS trend analysis."""
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            
# Copyright (c) 2025 Mohamed Z. Hatim
            smoothed = lowess(y, x, frac=frac, return_sorted=True)
            x_smooth = smoothed[:, 0]
            y_smooth = smoothed[:, 1]
            
# Copyright (c) 2025 Mohamed Z. Hatim
            interp_func = interp1d(x_smooth, y_smooth, 
                                 bounds_error=False, fill_value='extrapolate')
            y_pred = interp_func(x)
            
            return {
                'predicted_values': y_pred,
                'smooth_x': x_smooth,
                'smooth_y': y_smooth,
                'fraction': frac
            }
            
        except ImportError:
            warnings.warn("statsmodels not available, using simple moving average")
# Copyright (c) 2025 Mohamed Z. Hatim
            window = max(3, int(len(x) * frac))
            y_pred = pd.Series(y).rolling(window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
            
            return {
                'predicted_values': y_pred,
                'window_size': window,
                'method': 'moving_average_fallback'
            }
    
    def _mann_kendall_trend(self, x: np.ndarray, y: np.ndarray, 
                           alpha: float = 0.05, **kwargs) -> Dict[str, Any]:
        """Mann-Kendall trend test."""
        n = len(y)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if y[j] > y[i]:
                    s += 1
                elif y[j] < y[i]:
                    s -= 1
        
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
        unique_vals, counts = np.unique(y, return_counts=True)
        tie_adjustment = np.sum(counts * (counts - 1) * (2 * counts + 5))
        
        var_s = (n * (n - 1) * (2 * n + 5) - tie_adjustment) / 18
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
# Copyright (c) 2025 Mohamed Z. Hatim
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if p_value < alpha:
            if s > 0:
                trend = 'increasing'
            else:
                trend = 'decreasing'
        else:
            trend = 'no significant trend'
        
# Copyright (c) 2025 Mohamed Z. Hatim
        slopes = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                if x[j] != x[i]:
                    slopes.append((y[j] - y[i]) / (x[j] - x[i]))
        
        sens_slope = np.median(slopes) if slopes else 0
        
        return {
            's_statistic': s,
            'z_statistic': z,
            'p_value': p_value,
            'trend': trend,
            'sens_slope': sens_slope,
            'alpha': alpha,
            'significant': p_value < alpha
        }
    
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
    
    def seasonal_decomposition(self, data: pd.DataFrame,
                             time_col: str = 'date',
                             response_col: str = 'response',
                             method: str = 'classical',
                             period: Optional[int] = None,
                             **kwargs) -> Dict[str, Any]:
        """
        Decompose time series into trend, seasonal, and residual components.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Time series data
        time_col : str
            Time column name
        response_col : str
            Response variable column
        method : str
            Decomposition method
        period : int, optional
            Seasonal period
        **kwargs
            Additional parameters
            
        Returns:
        --------
        dict
            Decomposition results
        """
# Copyright (c) 2025 Mohamed Z. Hatim
        data_clean = data.dropna(subset=[time_col, response_col]).copy()
        data_clean[time_col] = pd.to_datetime(data_clean[time_col])
        data_clean = data_clean.sort_values(time_col)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        ts = pd.Series(
            data_clean[response_col].values,
            index=data_clean[time_col]
        )
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if period is None:
            period = self._detect_seasonality(ts)
        
        if method not in self.decomposition_methods:
            raise ValueError(f"Unknown decomposition method: {method}")
        
        decomp_func = self.decomposition_methods[method]
        return decomp_func(ts, period=period, **kwargs)
    
    def _classical_decomposition(self, ts: pd.Series, period: int,
                               model: str = 'additive', **kwargs) -> Dict[str, Any]:
        """Classical seasonal decomposition."""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
# Copyright (c) 2025 Mohamed Z. Hatim
            decomposition = seasonal_decompose(ts, model=model, period=period)
            
            return {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'observed': decomposition.observed,
                'model': model,
                'period': period,
                'method': 'classical'
            }
            
        except ImportError:
            warnings.warn("statsmodels not available, using simple decomposition")
# Copyright (c) 2025 Mohamed Z. Hatim
            return self._simple_decomposition(ts, period)
    
    def _stl_decomposition(self, ts: pd.Series, period: int, **kwargs) -> Dict[str, Any]:
        """STL (Seasonal and Trend decomposition using Loess) decomposition."""
        try:
            from statsmodels.tsa.seasonal import STL
            
# Copyright (c) 2025 Mohamed Z. Hatim
            stl = STL(ts, seasonal=kwargs.get('seasonal', 7), period=period)
            decomposition = stl.fit()
            
            return {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'observed': decomposition.observed,
                'period': period,
                'method': 'stl'
            }
            
        except ImportError:
            warnings.warn("statsmodels not available, using simple decomposition")
            return self._simple_decomposition(ts, period)
    
    def _x11_decomposition(self, ts: pd.Series, period: int, **kwargs) -> Dict[str, Any]:
        """X-11 seasonal adjustment (simplified version)."""
# Copyright (c) 2025 Mohamed Z. Hatim
        return self._classical_decomposition(ts, period, model='multiplicative')
    
    def _simple_decomposition(self, ts: pd.Series, period: int) -> Dict[str, Any]:
        """Simple fallback decomposition method."""
# Copyright (c) 2025 Mohamed Z. Hatim
        trend = ts.rolling(window=period, center=True).mean()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        detrended = ts - trend
        seasonal_means = detrended.groupby(detrended.index.dayofyear % period).mean()
        seasonal = detrended.copy()
        
        for i in range(len(seasonal)):
            day_of_cycle = seasonal.index[i].dayofyear % period
            if day_of_cycle in seasonal_means.index:
                seasonal.iloc[i] = seasonal_means[day_of_cycle]
            else:
                seasonal.iloc[i] = 0
        
# Copyright (c) 2025 Mohamed Z. Hatim
        residual = ts - trend - seasonal
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual,
            'observed': ts,
            'period': period,
            'method': 'simple'
        }
    
    def _detect_seasonality(self, ts: pd.Series) -> int:
        """Auto-detect seasonal period."""
# Copyright (c) 2025 Mohamed Z. Hatim
        if isinstance(ts.index, pd.DatetimeIndex):
            freq = pd.infer_freq(ts.index)
            if freq:
                if 'D' in freq:
                    return 365
                elif 'M' in freq:
                    return 12
                elif 'W' in freq:
                    return 52
                elif 'Q' in freq:
                    return 4
        
# Copyright (c) 2025 Mohamed Z. Hatim
        return max(4, min(len(ts) // 4, 365))
    
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
    
    def climate_vegetation_response(self, vegetation_data: pd.DataFrame,
                                  climate_data: pd.DataFrame,
                                  time_col: str = 'date',
                                  veg_col: str = 'response',
                                  climate_vars: List[str] = None,
                                  lag_periods: List[int] = [0, 1, 2, 3]) -> Dict[str, Any]:
        """
        Analyze vegetation response to climate variables.
        
        Parameters:
        -----------
        vegetation_data : pd.DataFrame
            Vegetation time series data
        climate_data : pd.DataFrame
            Climate data
        time_col : str
            Time column name
        veg_col : str
            Vegetation response column
        climate_vars : list
            Climate variables to analyze
        lag_periods : list
            Lag periods to test
            
        Returns:
        --------
        dict
            Climate-vegetation response analysis
        """
# Copyright (c) 2025 Mohamed Z. Hatim
        veg_clean = vegetation_data.dropna(subset=[time_col, veg_col])
        climate_clean = climate_data.dropna(subset=[time_col])
        
# Copyright (c) 2025 Mohamed Z. Hatim
        merged = pd.merge(veg_clean, climate_clean, on=time_col, how='inner')
        
        if climate_vars is None:
# Copyright (c) 2025 Mohamed Z. Hatim
            climate_vars = merged.select_dtypes(include=[np.number]).columns.tolist()
            climate_vars = [col for col in climate_vars if col != veg_col]
        
        results = {}
        
        for climate_var in climate_vars:
            if climate_var not in merged.columns:
                continue
            
            var_results = {
                'correlations': {},
                'best_lag': None,
                'best_correlation': 0,
                'response_curve': None
            }
            
# Copyright (c) 2025 Mohamed Z. Hatim
            for lag in lag_periods:
                if lag == 0:
                    climate_lagged = merged[climate_var]
                    veg_response = merged[veg_col]
                else:
# Copyright (c) 2025 Mohamed Z. Hatim
                    climate_lagged = merged[climate_var].shift(lag)
                    veg_response = merged[veg_col]
                    
# Copyright (c) 2025 Mohamed Z. Hatim
                    valid_mask = ~(climate_lagged.isna() | veg_response.isna())
                    climate_lagged = climate_lagged[valid_mask]
                    veg_response = veg_response[valid_mask]
                
                if len(climate_lagged) < 3:
                    continue
                
# Copyright (c) 2025 Mohamed Z. Hatim
                correlation = np.corrcoef(climate_lagged, veg_response)[0, 1]
                if np.isnan(correlation):
                    correlation = 0
                
                var_results['correlations'][lag] = correlation
                
# Copyright (c) 2025 Mohamed Z. Hatim
                if abs(correlation) > abs(var_results['best_correlation']):
                    var_results['best_correlation'] = correlation
                    var_results['best_lag'] = lag
            
# Copyright (c) 2025 Mohamed Z. Hatim
            if var_results['best_lag'] is not None:
                best_lag = var_results['best_lag']
                
                if best_lag == 0:
                    x_vals = merged[climate_var].values
                    y_vals = merged[veg_col].values
                else:
                    climate_lagged = merged[climate_var].shift(best_lag)
                    valid_mask = ~(climate_lagged.isna() | merged[veg_col].isna())
                    x_vals = climate_lagged[valid_mask].values
                    y_vals = merged[veg_col][valid_mask].values
                
                if len(x_vals) > 3:
# Copyright (c) 2025 Mohamed Z. Hatim
                    response_curve = self._fit_response_curve(x_vals, y_vals)
                    var_results['response_curve'] = response_curve
            
            results[climate_var] = var_results
        
        return {
            'climate_variables': climate_vars,
            'lag_periods_tested': lag_periods,
            'results': results,
            'data_summary': {
                'n_observations': len(merged),
                'time_span': merged[time_col].max() - merged[time_col].min()
            }
        }
    
    def _fit_response_curve(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fit response curve to climate-vegetation relationship."""
# Copyright (c) 2025 Mohamed Z. Hatim
        curve_types = {
            'linear': lambda x, a, b: a * x + b,
            'quadratic': lambda x, a, b, c: a * x**2 + b * x + c,
            'exponential': lambda x, a, b: a * np.exp(b * x),
            'logarithmic': lambda x, a, b: a * np.log(x) + b if np.all(x > 0) else a * x + b
        }
        
        best_curve = None
        best_r2 = -np.inf
        best_params = None
        
        for curve_name, curve_func in curve_types.items():
            try:
                if curve_name == 'logarithmic' and np.any(x <= 0):
                    continue
                    
                popt, _ = optimize.curve_fit(curve_func, x, y, maxfev=1000)
                y_pred = curve_func(x, *popt)
                r2 = r2_score(y, y_pred)
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_curve = curve_name
                    best_params = popt
                    
            except:
                continue
        
        return {
            'best_curve_type': best_curve,
            'parameters': best_params,
            'r_squared': best_r2,
            'curve_function': curve_types[best_curve] if best_curve else None
        }
    
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
    
    def growth_curve_fitting(self, data: pd.DataFrame,
                           time_col: str = 'time',
                           size_col: str = 'size',
                           species_col: Optional[str] = None,
                           curve_type: str = 'logistic') -> Dict[str, Any]:
        """
        Fit growth curves to species data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Growth data
        time_col : str
            Time variable
        size_col : str
            Size/biomass variable
        species_col : str, optional
            Species grouping column
        curve_type : str
            Type of growth curve
            
        Returns:
        --------
        dict
            Growth curve fitting results
        """
        growth_models = {
            'logistic': self._logistic_growth,
            'gompertz': self._gompertz_growth,
            'von_bertalanffy': self._von_bertalanffy_growth,
            'exponential': self._exponential_growth,
            'power': self._power_growth
        }
        
        if curve_type not in growth_models:
            raise ValueError(f"Unknown curve type: {curve_type}")
        
        results = {}
        
        if species_col and species_col in data.columns:
# Copyright (c) 2025 Mohamed Z. Hatim
            for species in data[species_col].unique():
                species_data = data[data[species_col] == species]
                if len(species_data) < 4:  # Need minimum points for fitting
                    continue
                    
                species_results = self._fit_growth_curve(
                    species_data[time_col].values,
                    species_data[size_col].values,
                    growth_models[curve_type]
                )
                results[species] = species_results
        else:
# Copyright (c) 2025 Mohamed Z. Hatim
            results['combined'] = self._fit_growth_curve(
                data[time_col].values,
                data[size_col].values,
                growth_models[curve_type]
            )
        
        return {
            'curve_type': curve_type,
            'results': results,
            'model_function': growth_models[curve_type]
        }
    
    def _fit_growth_curve(self, t: np.ndarray, size: np.ndarray, 
                         model_func: Callable) -> Dict[str, Any]:
        """Fit individual growth curve."""
        try:
# Copyright (c) 2025 Mohamed Z. Hatim
            initial_guess = self._get_growth_initial_guess(t, size, model_func)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            popt, pcov = optimize.curve_fit(
                model_func, t, size,
                p0=initial_guess,
                maxfev=5000
            )
            
# Copyright (c) 2025 Mohamed Z. Hatim
            size_pred = model_func(t, *popt)
            r_squared = r2_score(size, size_pred)
            rmse = np.sqrt(mean_squared_error(size, size_pred))
            
# Copyright (c) 2025 Mohamed Z. Hatim
            growth_params = self._extract_growth_parameters(model_func, popt, t)
            
            return {
                'parameters': popt,
                'parameter_covariance': pcov,
                'r_squared': r_squared,
                'rmse': rmse,
                'predicted_values': size_pred,
                'growth_parameters': growth_params,
                'success': True
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'success': False
            }
    
    def _logistic_growth(self, t: np.ndarray, K: float, r: float, t0: float) -> np.ndarray:
        """Logistic growth model."""
        return K / (1 + np.exp(-r * (t - t0)))
    
    def _gompertz_growth(self, t: np.ndarray, K: float, r: float, t0: float) -> np.ndarray:
        """Gompertz growth model."""
        return K * np.exp(-np.exp(-r * (t - t0)))
    
    def _von_bertalanffy_growth(self, t: np.ndarray, Linf: float, K: float, t0: float) -> np.ndarray:
        """von Bertalanffy growth model."""
        return Linf * (1 - np.exp(-K * (t - t0)))**3
    
    def _exponential_growth(self, t: np.ndarray, N0: float, r: float) -> np.ndarray:
        """Exponential growth model."""
        return N0 * np.exp(r * t)
    
    def _power_growth(self, t: np.ndarray, a: float, b: float) -> np.ndarray:
        """Power growth model."""
        return a * (t ** b)
    
    def _get_growth_initial_guess(self, t: np.ndarray, size: np.ndarray, 
                                 model_func: Callable) -> List[float]:
        """Get initial parameter guesses for growth curves."""
        t_range = t.max() - t.min()
        size_max = size.max()
        size_min = size.min()
        
        if model_func == self._logistic_growth:
            return [size_max * 1.2, 0.1, t.mean()]
        elif model_func == self._gompertz_growth:
            return [size_max * 1.2, 0.1, t.mean()]
        elif model_func == self._von_bertalanffy_growth:
            return [size_max * 1.2, 0.1, t.min()]
        elif model_func == self._exponential_growth:
            return [size_min, 0.1]
        elif model_func == self._power_growth:
            return [1.0, 1.0]
        else:
            return [1.0] * 3  # Default
    
    def _extract_growth_parameters(self, model_func: Callable, params: np.ndarray,
                                 t: np.ndarray) -> Dict[str, float]:
        """Extract meaningful growth parameters."""
        growth_params = {}
        
        if model_func == self._logistic_growth:
            K, r, t0 = params
            growth_params['carrying_capacity'] = K
            growth_params['growth_rate'] = r
            growth_params['inflection_point'] = t0
            growth_params['max_growth_rate_time'] = t0
            
        elif model_func == self._gompertz_growth:
            K, r, t0 = params
            growth_params['asymptotic_size'] = K
            growth_params['growth_rate'] = r
            growth_params['inflection_point'] = t0
            
        elif model_func == self._exponential_growth:
            N0, r = params
            growth_params['initial_size'] = N0
            growth_params['growth_rate'] = r
            
        return growth_params