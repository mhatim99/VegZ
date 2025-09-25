"""
Environmental modeling module with GAMs and gradient analysis.

Copyright (c) 2025 Mohamed Z. Hatim
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Tuple, Optional, Any, Callable
from scipy import stats, optimize
from scipy.interpolate import UnivariateSpline, BSpline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import warnings


class EnvironmentalModeler:
    """Environmental modeling and gradient analysis for ecological data."""
    
    def __init__(self):
        """Initialize environmental modeler."""
        self.gam_smoothers = {
            'spline': self._spline_smoother,
            'lowess': self._lowess_smoother,
            'polynomial': self._polynomial_smoother,
            'gaussian_process': self._gaussian_process_smoother
        }
        
        self.gradient_methods = {
            'cca': self._cca_gradient_analysis,
            'dca': self._dca_gradient_analysis,
            'rda': self._rda_gradient_analysis,
            'pca_env': self._pca_environmental_gradient
        }
        
        self.response_curves = {
            'gaussian': self._gaussian_response,
            'skewed_gaussian': self._skewed_gaussian_response,
            'beta': self._beta_response,
            'linear': self._linear_response,
            'threshold': self._threshold_response,
            'unimodal': self._unimodal_response
        }
    
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
    
    def fit_gam(self, data: pd.DataFrame,
                response_col: str,
                predictor_cols: List[str],
                smoother_types: Optional[Dict[str, str]] = None,
                family: str = 'gaussian',
                **kwargs) -> Dict[str, Any]:
        """
        Fit Generalized Additive Model.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        response_col : str
            Response variable column name
        predictor_cols : list
            Predictor variable column names
        smoother_types : dict, optional
            Smoother type for each predictor
        family : str
            Distribution family ('gaussian', 'binomial', 'poisson')
        **kwargs
            Additional parameters
            
        Returns:
        --------
        dict
            GAM results including model, smoothers, and diagnostics
        """
# Copyright (c) 2025 Mohamed Z. Hatim
        use_cols = [response_col] + predictor_cols
        clean_data = data[use_cols].dropna()
        
        if len(clean_data) < 10:
            raise ValueError("Insufficient data for GAM fitting")
        
        y = clean_data[response_col].values
        X = clean_data[predictor_cols]
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if smoother_types is None:
            smoother_types = {col: 'spline' for col in predictor_cols}
        
# Copyright (c) 2025 Mohamed Z. Hatim
        gam_results = {
            'smoothers': {},
            'linear_terms': {},
            'response_variable': response_col,
            'predictor_variables': predictor_cols,
            'family': family,
            'n_observations': len(clean_data)
        }
        
# Copyright (c) 2025 Mohamed Z. Hatim
        y_pred = np.mean(y) * np.ones_like(y)  # Start with intercept
        
# Copyright (c) 2025 Mohamed Z. Hatim
        for predictor in predictor_cols:
            smoother_type = smoother_types.get(predictor, 'spline')
            
            if smoother_type not in self.gam_smoothers:
                warnings.warn(f"Unknown smoother type {smoother_type}, using spline")
                smoother_type = 'spline'
            
            smoother_func = self.gam_smoothers[smoother_type]
            
# Copyright (c) 2025 Mohamed Z. Hatim
            smoother_result = smoother_func(
                X[predictor].values, y, **kwargs
            )
            
            gam_results['smoothers'][predictor] = {
                'type': smoother_type,
                'smoother': smoother_result['smoother'],
                'fitted_values': smoother_result['fitted_values'],
                'edf': smoother_result.get('edf', 1),  # Effective degrees of freedom
                'lambda': smoother_result.get('lambda', None),  # Smoothing parameter
                'r_squared': smoother_result.get('r_squared', 0)
            }
            
# Copyright (c) 2025 Mohamed Z. Hatim
            y_pred += smoother_result['fitted_values'] - np.mean(smoother_result['fitted_values'])
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if family == 'binomial':
# Copyright (c) 2025 Mohamed Z. Hatim
            y_pred_prob = 1 / (1 + np.exp(-y_pred))
            gam_results['fitted_probabilities'] = y_pred_prob
            y_pred = y_pred_prob
        elif family == 'poisson':
# Copyright (c) 2025 Mohamed Z. Hatim
            y_pred = np.exp(y_pred)
        
        gam_results['fitted_values'] = y_pred
        
# Copyright (c) 2025 Mohamed Z. Hatim
        diagnostics = self._calculate_gam_diagnostics(y, y_pred, gam_results, family)
        gam_results['diagnostics'] = diagnostics
        
# Copyright (c) 2025 Mohamed Z. Hatim
        anova_results = self._gam_anova(y, gam_results)
        gam_results['anova'] = anova_results
        
        return gam_results
    
    def _spline_smoother(self, x: np.ndarray, y: np.ndarray, 
                        smoothing_factor: Optional[float] = None,
                        **kwargs) -> Dict[str, Any]:
        """Fit spline smoother."""
# Copyright (c) 2025 Mohamed Z. Hatim
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        
        if len(x_clean) < 3:
# Copyright (c) 2025 Mohamed Z. Hatim
            return self._linear_smoother(x, y)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if smoothing_factor is None:
# Copyright (c) 2025 Mohamed Z. Hatim
            smoothing_candidates = np.logspace(-3, 1, 20)
            best_score = -np.inf
            best_s = smoothing_candidates[0]
            
            for s in smoothing_candidates:
                try:
                    spline = UnivariateSpline(x_clean, y_clean, s=s)
                    y_pred = spline(x_clean)
                    score = r2_score(y_clean, y_pred)
                    
                    if score > best_score:
                        best_score = score
                        best_s = s
                except:
                    continue
            
            smoothing_factor = best_s
        
# Copyright (c) 2025 Mohamed Z. Hatim
        try:
            spline = UnivariateSpline(x_clean, y_clean, s=smoothing_factor)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            fitted_values = np.full_like(x, np.mean(y_clean))
            fitted_values[valid_mask] = spline(x_clean)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            edf = min(len(x_clean), max(2, len(x_clean) / (1 + smoothing_factor)))
            
            return {
                'smoother': spline,
                'fitted_values': fitted_values,
                'edf': edf,
                'lambda': smoothing_factor,
                'r_squared': r2_score(y_clean, spline(x_clean))
            }
            
        except Exception as e:
            warnings.warn(f"Spline fitting failed: {e}, using linear smoother")
            return self._linear_smoother(x, y)
    
    def _lowess_smoother(self, x: np.ndarray, y: np.ndarray,
                        frac: float = 0.3, **kwargs) -> Dict[str, Any]:
        """Fit LOWESS smoother."""
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            
            valid_mask = ~(np.isnan(x) | np.isnan(y))
            x_clean = x[valid_mask]
            y_clean = y[valid_mask]
            
            if len(x_clean) < 3:
                return self._linear_smoother(x, y)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            smoothed = lowess(y_clean, x_clean, frac=frac, return_sorted=True)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            from scipy.interpolate import interp1d
            interp_func = interp1d(smoothed[:, 0], smoothed[:, 1], 
                                 bounds_error=False, fill_value='extrapolate')
            
            fitted_values = np.full_like(x, np.mean(y_clean))
            fitted_values[valid_mask] = interp_func(x_clean)
            
            return {
                'smoother': interp_func,
                'fitted_values': fitted_values,
                'edf': len(x_clean) * frac,
                'frac': frac,
                'r_squared': r2_score(y_clean, interp_func(x_clean))
            }
            
        except ImportError:
            warnings.warn("statsmodels not available, using spline smoother")
            return self._spline_smoother(x, y, **kwargs)
    
    def _polynomial_smoother(self, x: np.ndarray, y: np.ndarray,
                           degree: int = 3, **kwargs) -> Dict[str, Any]:
        """Fit polynomial smoother."""
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        
        if len(x_clean) < degree + 1:
            degree = max(1, len(x_clean) - 1)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(x_clean.reshape(-1, 1))
        
        reg = LinearRegression()
        reg.fit(X_poly, y_clean)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        fitted_values = np.full_like(x, np.mean(y_clean))
        X_all_poly = poly_features.transform(x[valid_mask].reshape(-1, 1))
        fitted_values[valid_mask] = reg.predict(X_all_poly)
        
        return {
            'smoother': (poly_features, reg),
            'fitted_values': fitted_values,
            'edf': degree + 1,
            'degree': degree,
            'r_squared': reg.score(X_poly, y_clean)
        }
    
    def _gaussian_process_smoother(self, x: np.ndarray, y: np.ndarray,
                                 **kwargs) -> Dict[str, Any]:
        """Gaussian Process smoother (simplified implementation)."""
# Copyright (c) 2025 Mohamed Z. Hatim
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel
            
            valid_mask = ~(np.isnan(x) | np.isnan(y))
            x_clean = x[valid_mask].reshape(-1, 1)
            y_clean = y[valid_mask]
            
            if len(x_clean) < 3:
                return self._linear_smoother(x, y)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            kernel = ConstantKernel(1.0) * RBF(1.0)
            gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
            
            gp.fit(x_clean, y_clean)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            fitted_values = np.full_like(x, np.mean(y_clean))
            fitted_values[valid_mask], _ = gp.predict(x_clean, return_std=True)
            
            return {
                'smoother': gp,
                'fitted_values': fitted_values,
                'edf': len(x_clean) / 2,  # Rough approximation
                'r_squared': gp.score(x_clean, y_clean)
            }
            
        except ImportError:
            warnings.warn("scikit-learn GP not available, using spline smoother")
            return self._spline_smoother(x, y, **kwargs)
    
    def _linear_smoother(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Linear smoother (fallback)."""
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        
        if len(x_clean) < 2:
            fitted_values = np.full_like(x, np.mean(y) if len(y) > 0 else 0)
            return {
                'smoother': lambda xi: np.mean(y) if len(y) > 0 else 0,
                'fitted_values': fitted_values,
                'edf': 1,
                'r_squared': 0
            }
        
# Copyright (c) 2025 Mohamed Z. Hatim
        reg = LinearRegression()
        reg.fit(x_clean.reshape(-1, 1), y_clean)
        
        fitted_values = np.full_like(x, np.mean(y_clean))
        fitted_values[valid_mask] = reg.predict(x_clean.reshape(-1, 1))
        
        return {
            'smoother': reg,
            'fitted_values': fitted_values,
            'edf': 2,
            'r_squared': reg.score(x_clean.reshape(-1, 1), y_clean)
        }
    
    def _calculate_gam_diagnostics(self, y: np.ndarray, y_pred: np.ndarray,
                                  gam_results: Dict[str, Any], family: str) -> Dict[str, Any]:
        """Calculate GAM diagnostic statistics."""
        n = len(y)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if family == 'gaussian':
            deviance = np.sum((y - y_pred) ** 2)
            null_deviance = np.sum((y - np.mean(y)) ** 2)
        elif family == 'binomial':
# Copyright (c) 2025 Mohamed Z. Hatim
            y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
            deviance = -2 * np.sum(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
            null_deviance = -2 * np.sum(y * np.log(np.mean(y)) + (1 - y) * np.log(1 - np.mean(y)))
        elif family == 'poisson':
# Copyright (c) 2025 Mohamed Z. Hatim
            y_pred_clipped = np.clip(y_pred, 1e-15, np.inf)
            deviance = 2 * np.sum(y * np.log(y / y_pred_clipped) - (y - y_pred_clipped))
            null_deviance = 2 * np.sum(y * np.log(y / np.mean(y)) - (y - np.mean(y)))
        
# Copyright (c) 2025 Mohamed Z. Hatim
        total_edf = sum(smoother['edf'] for smoother in gam_results['smoothers'].values())
        
# Copyright (c) 2025 Mohamed Z. Hatim
        aic = deviance + 2 * total_edf
        bic = deviance + np.log(n) * total_edf
        
# Copyright (c) 2025 Mohamed Z. Hatim
        explained_deviance = 1 - (deviance / null_deviance) if null_deviance > 0 else 0
        
# Copyright (c) 2025 Mohamed Z. Hatim
        residuals = y - y_pred
        
# Copyright (c) 2025 Mohamed Z. Hatim
        leverage = total_edf / n  # Rough approximation
        cooks_d = (residuals ** 2) * leverage / (1 - leverage)
        
        return {
            'deviance': deviance,
            'null_deviance': null_deviance,
            'explained_deviance': explained_deviance,
            'aic': aic,
            'bic': bic,
            'total_edf': total_edf,
            'residuals': residuals,
            'cooks_distance': cooks_d,
            'rmse': np.sqrt(np.mean(residuals ** 2)),
            'r_squared': r2_score(y, y_pred) if family == 'gaussian' else explained_deviance
        }
    
    def _gam_anova(self, y: np.ndarray, gam_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate ANOVA-like decomposition for GAM."""
        anova_results = {}
        
        total_ss = np.sum((y - np.mean(y)) ** 2)
        
        for predictor, smoother_info in gam_results['smoothers'].items():
# Copyright (c) 2025 Mohamed Z. Hatim
            fitted_component = smoother_info['fitted_values'] - np.mean(smoother_info['fitted_values'])
            component_ss = np.sum(fitted_component ** 2)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            edf = smoother_info['edf']
            residual_df = len(y) - gam_results['diagnostics']['total_edf']
            residual_ms = gam_results['diagnostics']['deviance'] / residual_df if residual_df > 0 else 1
            
            f_stat = (component_ss / edf) / residual_ms if residual_ms > 0 else 0
            p_value = 1 - stats.f.cdf(f_stat, edf, residual_df) if residual_df > 0 else 0.5
            
            anova_results[predictor] = {
                'sum_of_squares': component_ss,
                'edf': edf,
                'mean_square': component_ss / edf if edf > 0 else 0,
                'f_statistic': f_stat,
                'p_value': p_value,
                'variance_explained': component_ss / total_ss if total_ss > 0 else 0
            }
        
        return anova_results
    
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
    
    def environmental_gradient_analysis(self, species_data: pd.DataFrame,
                                      env_data: pd.DataFrame,
                                      method: str = 'cca',
                                      **kwargs) -> Dict[str, Any]:
        """
        Analyze species responses along environmental gradients.
        
        Parameters:
        -----------
        species_data : pd.DataFrame
            Species abundance matrix
        env_data : pd.DataFrame
            Environmental variables
        method : str
            Gradient analysis method
        **kwargs
            Additional parameters
            
        Returns:
        --------
        dict
            Gradient analysis results
        """
        if method not in self.gradient_methods:
            raise ValueError(f"Unknown gradient method: {method}")
        
        gradient_func = self.gradient_methods[method]
        return gradient_func(species_data, env_data, **kwargs)
    
    def _cca_gradient_analysis(self, species_data: pd.DataFrame,
                              env_data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """CCA-based gradient analysis."""
# Copyright (c) 2025 Mohamed Z. Hatim
        from .multivariate import MultivariateAnalyzer
        
        mv_analyzer = MultivariateAnalyzer()
        cca_results = mv_analyzer.cca_analysis(species_data, env_data)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        gradient_results = {
            'method': 'CCA',
            'axis_scores': cca_results['site_scores'],
            'species_scores': cca_results['species_scores'],
            'environmental_scores': cca_results['env_scores'],
            'eigenvalues': cca_results['eigenvalues'],
            'explained_variance': cca_results['explained_variance_ratio']
        }
        
# Copyright (c) 2025 Mohamed Z. Hatim
        optima = self._calculate_species_optima(
            species_data, cca_results['site_scores']
        )
        gradient_results['species_optima'] = optima
        
        return gradient_results
    
    def _dca_gradient_analysis(self, species_data: pd.DataFrame,
                              env_data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """DCA-based gradient analysis."""
        from .multivariate import MultivariateAnalyzer
        
        mv_analyzer = MultivariateAnalyzer()
        dca_results = mv_analyzer.dca_analysis(species_data)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        axis_env_correlations = {}
        for axis in dca_results['site_scores'].columns:
            axis_values = dca_results['site_scores'][axis]
            correlations = {}
            
            for env_var in env_data.columns:
                if env_data[env_var].dtype in ['float64', 'int64']:
# Copyright (c) 2025 Mohamed Z. Hatim
                    common_idx = axis_values.index.intersection(env_data.index)
                    if len(common_idx) > 3:
                        corr = np.corrcoef(
                            axis_values.loc[common_idx],
                            env_data.loc[common_idx, env_var]
                        )[0, 1]
                        correlations[env_var] = corr if not np.isnan(corr) else 0
            
            axis_env_correlations[axis] = correlations
        
        gradient_results = {
            'method': 'DCA',
            'axis_scores': dca_results['site_scores'],
            'species_scores': dca_results['species_scores'],
            'eigenvalues': dca_results['eigenvalues'],
            'gradient_lengths': dca_results['gradient_lengths'],
            'axis_environment_correlations': axis_env_correlations
        }
        
        return gradient_results
    
    def _rda_gradient_analysis(self, species_data: pd.DataFrame,
                              env_data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """RDA-based gradient analysis."""
        from .multivariate import MultivariateAnalyzer
        
        mv_analyzer = MultivariateAnalyzer()
        rda_results = mv_analyzer.rda_analysis(species_data, env_data)
        
        gradient_results = {
            'method': 'RDA',
            'axis_scores': rda_results['site_scores'],
            'species_scores': rda_results['species_scores'],
            'environmental_scores': rda_results['env_scores'],
            'eigenvalues': rda_results['eigenvalues'],
            'explained_variance': rda_results['explained_variance_ratio']
        }
        
        return gradient_results
    
    def _pca_environmental_gradient(self, species_data: pd.DataFrame,
                                   env_data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """PCA of environmental variables for gradient analysis."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
# Copyright (c) 2025 Mohamed Z. Hatim
        scaler = StandardScaler()
        env_scaled = scaler.fit_transform(env_data.select_dtypes(include=[np.number]))
        
# Copyright (c) 2025 Mohamed Z. Hatim
        pca = PCA()
        env_scores = pca.fit_transform(env_scaled)
        
        gradient_results = {
            'method': 'PCA_Environmental',
            'environmental_scores': pd.DataFrame(
                env_scores,
                index=env_data.index,
                columns=[f'PC{i+1}' for i in range(env_scores.shape[1])]
            ),
            'loadings': pd.DataFrame(
                pca.components_.T,
                index=env_data.select_dtypes(include=[np.number]).columns,
                columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])]
            ),
            'explained_variance': pca.explained_variance_ratio_,
            'eigenvalues': pca.explained_variance_
        }
        
        return gradient_results
    
    def _calculate_species_optima(self, species_data: pd.DataFrame,
                                 gradient_scores: pd.DataFrame) -> pd.DataFrame:
        """Calculate species optima along gradients."""
        optima = pd.DataFrame(index=species_data.columns,
                            columns=gradient_scores.columns)
        
        for axis in gradient_scores.columns:
            for species in species_data.columns:
# Copyright (c) 2025 Mohamed Z. Hatim
                abundances = species_data[species]
                gradient_vals = gradient_scores[axis]
                
# Copyright (c) 2025 Mohamed Z. Hatim
                common_idx = abundances.index.intersection(gradient_vals.index)
                
                if len(common_idx) > 0:
                    abund_aligned = abundances.loc[common_idx]
                    grad_aligned = gradient_vals.loc[common_idx]
                    
# Copyright (c) 2025 Mohamed Z. Hatim
                    nonzero_mask = abund_aligned > 0
                    
                    if nonzero_mask.any():
                        weights = abund_aligned[nonzero_mask]
                        values = grad_aligned[nonzero_mask]
                        
                        optimum = np.average(values, weights=weights)
                        optima.loc[species, axis] = optimum
        
        return optima
    
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
    
    def species_response_curves(self, species_data: pd.Series,
                               environmental_var: pd.Series,
                               curve_type: str = 'gaussian',
                               **kwargs) -> Dict[str, Any]:
        """
        Fit species response curves along environmental gradients.
        
        Parameters:
        -----------
        species_data : pd.Series
            Species abundance/occurrence data
        environmental_var : pd.Series
            Environmental variable
        curve_type : str
            Type of response curve to fit
        **kwargs
            Additional parameters
            
        Returns:
        --------
        dict
            Response curve fitting results
        """
# Copyright (c) 2025 Mohamed Z. Hatim
        common_idx = species_data.index.intersection(environmental_var.index)
        species_aligned = species_data.loc[common_idx]
        env_aligned = environmental_var.loc[common_idx]
        
# Copyright (c) 2025 Mohamed Z. Hatim
        valid_mask = ~(species_aligned.isna() | env_aligned.isna())
        species_clean = species_aligned[valid_mask].values
        env_clean = env_aligned[valid_mask].values
        
        if len(species_clean) < 4:
            raise ValueError("Insufficient data points for curve fitting")
        
        if curve_type not in self.response_curves:
            raise ValueError(f"Unknown curve type: {curve_type}")
        
        curve_func = self.response_curves[curve_type]
        
        try:
# Copyright (c) 2025 Mohamed Z. Hatim
            popt, pcov = optimize.curve_fit(
                curve_func, env_clean, species_clean,
                maxfev=5000
            )
            
# Copyright (c) 2025 Mohamed Z. Hatim
            fitted_values = curve_func(env_clean, *popt)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            r_squared = r2_score(species_clean, fitted_values)
            rmse = np.sqrt(mean_squared_error(species_clean, fitted_values))
            
# Copyright (c) 2025 Mohamed Z. Hatim
            ecological_params = self._extract_response_parameters(
                curve_type, popt, env_clean
            )
            
# Copyright (c) 2025 Mohamed Z. Hatim
            env_range = np.linspace(env_clean.min(), env_clean.max(), 100)
            predicted_curve = curve_func(env_range, *popt)
            
            results = {
                'curve_type': curve_type,
                'parameters': popt,
                'parameter_covariance': pcov,
                'r_squared': r_squared,
                'rmse': rmse,
                'fitted_values': fitted_values,
                'ecological_parameters': ecological_params,
                'prediction_range': env_range,
                'predicted_curve': predicted_curve,
                'curve_function': curve_func,
                'success': True
            }
            
        except Exception as e:
            warnings.warn(f"Curve fitting failed: {e}")
            results = {
                'curve_type': curve_type,
                'error': str(e),
                'success': False
            }
        
        return results
    
    def _gaussian_response(self, x: np.ndarray, amplitude: float, 
                          optimum: float, tolerance: float, baseline: float) -> np.ndarray:
        """Gaussian response curve."""
        return amplitude * np.exp(-0.5 * ((x - optimum) / tolerance) ** 2) + baseline
    
    def _skewed_gaussian_response(self, x: np.ndarray, amplitude: float,
                                 optimum: float, tolerance: float,
                                 skewness: float, baseline: float) -> np.ndarray:
        """Skewed Gaussian response curve."""
        z = (x - optimum) / tolerance
        gaussian = np.exp(-0.5 * z ** 2)
        skew_factor = 1 + skewness * z
        return amplitude * gaussian * skew_factor + baseline
    
    def _beta_response(self, x: np.ndarray, amplitude: float,
                      alpha: float, beta: float,
                      x_min: float, x_max: float) -> np.ndarray:
        """Beta function response curve."""
# Copyright (c) 2025 Mohamed Z. Hatim
        x_norm = (x - x_min) / (x_max - x_min)
        x_norm = np.clip(x_norm, 1e-10, 1 - 1e-10)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        beta_func = (x_norm ** (alpha - 1)) * ((1 - x_norm) ** (beta - 1))
        
# Copyright (c) 2025 Mohamed Z. Hatim
        max_beta = ((alpha - 1) / (alpha + beta - 2)) ** (alpha - 1) * \
                   ((beta - 1) / (alpha + beta - 2)) ** (beta - 1)
        
        if max_beta > 0:
            beta_func = beta_func / max_beta
        
        return amplitude * beta_func
    
    def _linear_response(self, x: np.ndarray, slope: float, intercept: float) -> np.ndarray:
        """Linear response curve."""
        return slope * x + intercept
    
    def _threshold_response(self, x: np.ndarray, amplitude: float,
                           threshold: float, steepness: float, baseline: float) -> np.ndarray:
        """Threshold response curve."""
        return amplitude / (1 + np.exp(-steepness * (x - threshold))) + baseline
    
    def _unimodal_response(self, x: np.ndarray, amplitude: float,
                          optimum: float, tolerance: float, baseline: float) -> np.ndarray:
        """Unimodal response curve (same as Gaussian)."""
        return self._gaussian_response(x, amplitude, optimum, tolerance, baseline)
    
    def _extract_response_parameters(self, curve_type: str, params: np.ndarray,
                                   env_data: np.ndarray) -> Dict[str, float]:
        """Extract ecological parameters from fitted curves."""
        ecological_params = {}
        
        if curve_type == 'gaussian':
            amplitude, optimum, tolerance, baseline = params
            ecological_params['optimum'] = optimum
            ecological_params['tolerance'] = abs(tolerance)
            ecological_params['amplitude'] = amplitude
            ecological_params['baseline'] = baseline
            ecological_params['niche_width'] = 2 * abs(tolerance)  # 2 standard deviations
            
        elif curve_type == 'skewed_gaussian':
            amplitude, optimum, tolerance, skewness, baseline = params
            ecological_params['optimum'] = optimum
            ecological_params['tolerance'] = abs(tolerance)
            ecological_params['skewness'] = skewness
            ecological_params['amplitude'] = amplitude
            ecological_params['baseline'] = baseline
            
        elif curve_type == 'beta':
            amplitude, alpha, beta_param, x_min, x_max = params
# Copyright (c) 2025 Mohamed Z. Hatim
            if alpha > 1 and beta_param > 1:
                mode = x_min + (alpha - 1) / (alpha + beta_param - 2) * (x_max - x_min)
                ecological_params['optimum'] = mode
            ecological_params['amplitude'] = amplitude
            ecological_params['range_min'] = x_min
            ecological_params['range_max'] = x_max
            ecological_params['niche_width'] = x_max - x_min
            
        elif curve_type == 'linear':
            slope, intercept = params
            ecological_params['slope'] = slope
            ecological_params['intercept'] = intercept
            ecological_params['trend'] = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'neutral'
            
        elif curve_type == 'threshold':
            amplitude, threshold, steepness, baseline = params
            ecological_params['threshold'] = threshold
            ecological_params['steepness'] = steepness
            ecological_params['amplitude'] = amplitude
            ecological_params['baseline'] = baseline
        
        return ecological_params
    
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
    
    def climate_vegetation_response_modeling(self, vegetation_data: pd.DataFrame,
                                           climate_data: pd.DataFrame,
                                           response_col: str,
                                           climate_vars: List[str],
                                           model_type: str = 'gam') -> Dict[str, Any]:
        """
        Model vegetation responses to climate variables.
        
        Parameters:
        -----------
        vegetation_data : pd.DataFrame
            Vegetation response data
        climate_data : pd.DataFrame
            Climate predictor variables
        response_col : str
            Response variable column
        climate_vars : list
            Climate variables to include
        model_type : str
            Type of model to fit
            
        Returns:
        --------
        dict
            Climate-vegetation response model results
        """
# Copyright (c) 2025 Mohamed Z. Hatim
        merged_data = pd.merge(vegetation_data, climate_data, 
                              left_index=True, right_index=True, how='inner')
        
        if model_type == 'gam':
# Copyright (c) 2025 Mohamed Z. Hatim
            results = self.fit_gam(
                merged_data, response_col, climate_vars,
                family='gaussian'
            )
            
        elif model_type == 'response_curves':
# Copyright (c) 2025 Mohamed Z. Hatim
            results = {'individual_responses': {}}
            
            for climate_var in climate_vars:
                if climate_var in merged_data.columns:
                    try:
                        curve_result = self.species_response_curves(
                            merged_data[response_col],
                            merged_data[climate_var],
                            curve_type='gaussian'
                        )
                        results['individual_responses'][climate_var] = curve_result
                    except Exception as e:
                        warnings.warn(f"Failed to fit curve for {climate_var}: {e}")
        
        elif model_type == 'multiple_regression':
# Copyright (c) 2025 Mohamed Z. Hatim
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            from sklearn.pipeline import Pipeline
            
            X = merged_data[climate_vars]
            y = merged_data[response_col]
            
# Copyright (c) 2025 Mohamed Z. Hatim
            poly_reg = Pipeline([
                ('poly', PolynomialFeatures(degree=2, interaction_only=True)),
                ('reg', LinearRegression())
            ])
            
            poly_reg.fit(X, y)
            
            results = {
                'model': poly_reg,
                'r_squared': poly_reg.score(X, y),
                'climate_variables': climate_vars,
                'model_type': 'multiple_regression'
            }
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return results