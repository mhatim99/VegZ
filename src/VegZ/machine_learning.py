"""
Machine Learning and Predictive Analysis Module

This module provides comprehensive machine learning capabilities for vegetation data,
including species identification, predictive modeling, and ecological insights.

Copyright (c) 2025 Mohamed Z. Hatim
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn.ensemble import IsolationForest
    ISOLATION_FOREST_AVAILABLE = True
except ImportError:
    ISOLATION_FOREST_AVAILABLE = False
    warnings.warn("IsolationForest not available, some anomaly detection methods will be limited")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available, using alternative gradient boosting methods")

try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    ITERATIVE_IMPUTER_AVAILABLE = True
except ImportError:
    ITERATIVE_IMPUTER_AVAILABLE = False
    warnings.warn("IterativeImputer not available, using SimpleImputer for missing data")


class MachineLearningAnalyzer:
    """
    Comprehensive machine learning analyzer for vegetation data.
    
    Provides species identification, predictive modeling, and ecological insights
    using advanced machine learning techniques.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the MachineLearningAnalyzer.
        
        Parameters
        ----------
        random_state : int, optional
            Random state for reproducibility, by default 42
        """
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.model_performance = {}
        
    def prepare_data(self, 
                    data: pd.DataFrame, 
                    target_column: str,
                    feature_columns: Optional[List[str]] = None,
                    handle_missing: str = 'impute',
                    scale_features: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for machine learning analysis.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        target_column : str
            Name of target column
        feature_columns : List[str], optional
            List of feature columns. If None, use all except target
        handle_missing : str, optional
            How to handle missing data ('drop', 'impute'), by default 'impute'
        scale_features : bool, optional
            Whether to scale features, by default True
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, List[str]]
            Features, target, and feature names
        """
# Copyright (c) 2025 Mohamed Z. Hatim
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        
        X = data[feature_columns].copy()
        y = data[target_column].copy()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if handle_missing == 'drop':
            complete_cases = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[complete_cases]
            y = y[complete_cases]
        elif handle_missing == 'impute':
# Copyright (c) 2025 Mohamed Z. Hatim
            if ITERATIVE_IMPUTER_AVAILABLE:
                imputer = IterativeImputer(random_state=self.random_state)
                X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
            else:
                X = X.fillna(X.mean())
            
# Copyright (c) 2025 Mohamed Z. Hatim
            y = y.fillna(y.mean() if y.dtype.kind in 'biufc' else y.mode()[0])
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if scale_features:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers['main'] = scaler
        else:
            X_scaled = X.values
        
        return X_scaled, y.values, list(X.columns)
    
    def species_identification(self, 
                             data: pd.DataFrame,
                             morphological_features: List[str],
                             species_column: str,
                             test_size: float = 0.3,
                             model_types: List[str] = None) -> Dict[str, Any]:
        """
        Implement species identification using morphological features.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with morphological features and species labels
        morphological_features : List[str]
            List of morphological feature columns
        species_column : str
            Column containing species labels
        test_size : float, optional
            Test set proportion, by default 0.3
        model_types : List[str], optional
            List of models to try, by default ['rf', 'svm', 'mlp']
            
        Returns
        -------
        Dict[str, Any]
            Species identification results
        """
        if model_types is None:
            model_types = ['rf', 'svm', 'mlp']
        
# Copyright (c) 2025 Mohamed Z. Hatim
        X, y, feature_names = self.prepare_data(data, species_column, morphological_features)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.encoders['species'] = le
        
# Copyright (c) 2025 Mohamed Z. Hatim
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=self.random_state, stratify=y_encoded
        )
        
        results = {
            'models': {},
            'performance': {},
            'feature_importance': {},
            'predictions': {},
            'feature_names': feature_names,
            'species_names': le.classes_
        }
        
# Copyright (c) 2025 Mohamed Z. Hatim
        models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'svm': SVC(kernel='rbf', random_state=self.random_state),
            'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=self.random_state, max_iter=500)
        }
        
        for model_name in model_types:
            if model_name not in models:
                continue
                
            model = models[model_name]
            
# Copyright (c) 2025 Mohamed Z. Hatim
            model.fit(X_train, y_train)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
# Copyright (c) 2025 Mohamed Z. Hatim
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(feature_names, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                importance = dict(zip(feature_names, np.abs(model.coef_).mean(axis=0)))
            else:
                importance = {}
            
            results['models'][model_name] = model
            results['performance'][model_name] = {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': pd.crosstab(y_test, y_pred, margins=True)
            }
            results['feature_importance'][model_name] = importance
            results['predictions'][model_name] = {
                'y_true': le.inverse_transform(y_test),
                'y_pred': le.inverse_transform(y_pred),
                'probabilities': y_pred_proba
            }
        
# Copyright (c) 2025 Mohamed Z. Hatim
        best_model = max(results['performance'].items(), key=lambda x: x[1]['accuracy'])
        results['best_model'] = best_model[0]
        self.models['species_identification'] = results
        
        return results
    
    def habitat_suitability_modeling(self,
                                   data: pd.DataFrame,
                                   species_column: str,
                                   environmental_features: List[str],
                                   model_type: str = 'rf',
                                   cross_validation: bool = True) -> Dict[str, Any]:
        """
        Model habitat suitability for species.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with species presence/absence and environmental variables
        species_column : str
            Column containing species presence/absence (0/1)
        environmental_features : List[str]
            List of environmental feature columns
        model_type : str, optional
            Model type ('rf', 'gbm', 'logistic'), by default 'rf'
        cross_validation : bool, optional
            Whether to perform cross-validation, by default True
            
        Returns
        -------
        Dict[str, Any]
            Habitat suitability modeling results
        """
# Copyright (c) 2025 Mohamed Z. Hatim
        X, y, feature_names = self.prepare_data(data, species_column, environmental_features)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if model_type == 'rf':
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        elif model_type == 'gbm':
            model = GradientBoostingRegressor(random_state=self.random_state)
        elif model_type == 'logistic':
            model = LogisticRegression(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
# Copyright (c) 2025 Mohamed Z. Hatim
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        
# Copyright (c) 2025 Mohamed Z. Hatim
        model.fit(X_train, y_train)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            suitability_proba = model.predict_proba(X_test)[:, 1]
        else:
            suitability_proba = model.predict(X_test)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if model_type in ['rf', 'logistic']:
            accuracy = accuracy_score(y_test, y_pred)
            performance = {'accuracy': accuracy}
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            performance = {'mse': mse, 'r2': r2}
        
# Copyright (c) 2025 Mohamed Z. Hatim
        cv_scores = None
        if cross_validation:
            cv_scores = cross_val_score(model, X, y, cv=5, random_state=self.random_state)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(feature_names, model.feature_importances_))
        else:
            importance = {}
        
        results = {
            'model': model,
            'performance': performance,
            'cv_scores': cv_scores,
            'feature_importance': importance,
            'predictions': {
                'y_true': y_test,
                'y_pred': y_pred,
                'suitability_probability': suitability_proba
            },
            'feature_names': feature_names
        }
        
        self.models['habitat_suitability'] = results
        return results
    
    def biomass_prediction(self,
                          data: pd.DataFrame,
                          biomass_column: str,
                          predictor_features: List[str],
                          model_type: str = 'rf',
                          optimize_hyperparameters: bool = True) -> Dict[str, Any]:
        """
        Predict vegetation biomass using environmental and structural variables.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with biomass measurements and predictor variables
        biomass_column : str
            Column containing biomass values
        predictor_features : List[str]
            List of predictor feature columns
        model_type : str, optional
            Model type ('rf', 'gbm', 'linear'), by default 'rf'
        optimize_hyperparameters : bool, optional
            Whether to optimize hyperparameters, by default True
            
        Returns
        -------
        Dict[str, Any]
            Biomass prediction results
        """
# Copyright (c) 2025 Mohamed Z. Hatim
        X, y, feature_names = self.prepare_data(data, biomass_column, predictor_features)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if model_type == 'rf':
            model = RandomForestRegressor(random_state=self.random_state)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        elif model_type == 'gbm':
            if LIGHTGBM_AVAILABLE:
                model = lgb.LGBMRegressor(random_state=self.random_state)
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            else:
                model = GradientBoostingRegressor(random_state=self.random_state)
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
        elif model_type == 'linear':
            model = Ridge()
            param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if optimize_hyperparameters:
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            model.fit(X_train, y_train)
            best_model = model
            best_params = {}
        
# Copyright (c) 2025 Mohamed Z. Hatim
        y_pred = best_model.predict(X_test)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred))
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if hasattr(best_model, 'feature_importances_'):
            importance = dict(zip(feature_names, best_model.feature_importances_))
        elif hasattr(best_model, 'coef_'):
            importance = dict(zip(feature_names, np.abs(best_model.coef_)))
        else:
            importance = {}
        
        results = {
            'model': best_model,
            'best_parameters': best_params,
            'performance': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            },
            'feature_importance': importance,
            'predictions': {
                'y_true': y_test,
                'y_pred': y_pred
            },
            'feature_names': feature_names
        }
        
        self.models['biomass_prediction'] = results
        return results
    
    def ecological_anomaly_detection(self,
                                   data: pd.DataFrame,
                                   feature_columns: List[str],
                                   contamination: float = 0.1,
                                   method: str = 'isolation_forest') -> Dict[str, Any]:
        """
        Detect ecological anomalies in vegetation data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        feature_columns : List[str]
            List of feature columns to use for anomaly detection
        contamination : float, optional
            Expected proportion of outliers, by default 0.1
        method : str, optional
            Anomaly detection method ('isolation_forest', 'dbscan'), by default 'isolation_forest'
            
        Returns
        -------
        Dict[str, Any]
            Anomaly detection results
        """
# Copyright (c) 2025 Mohamed Z. Hatim
        X = data[feature_columns].copy()
        X = X.fillna(X.mean())  # Simple imputation for anomaly detection
        
# Copyright (c) 2025 Mohamed Z. Hatim
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if method == 'isolation_forest' and ISOLATION_FOREST_AVAILABLE:
            detector = IsolationForest(contamination=contamination, random_state=self.random_state)
            anomaly_labels = detector.fit_predict(X_scaled)
            anomaly_scores = detector.decision_function(X_scaled)
        elif method == 'dbscan':
            detector = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = detector.fit_predict(X_scaled)
# Copyright (c) 2025 Mohamed Z. Hatim
            anomaly_labels = np.where(cluster_labels == -1, -1, 1)
            anomaly_scores = np.zeros(len(X_scaled))  # DBSCAN doesn't provide scores
        else:
            raise ValueError(f"Unknown or unavailable method: {method}")
        
# Copyright (c) 2025 Mohamed Z. Hatim
        is_anomaly = anomaly_labels == -1
        
        results = {
            'detector': detector,
            'anomaly_labels': anomaly_labels,
            'anomaly_scores': anomaly_scores,
            'is_anomaly': is_anomaly,
            'n_anomalies': np.sum(is_anomaly),
            'anomaly_indices': np.where(is_anomaly)[0],
            'feature_names': feature_columns
        }
        
# Copyright (c) 2025 Mohamed Z. Hatim
        results['data_with_anomalies'] = data.copy()
        results['data_with_anomalies']['is_anomaly'] = is_anomaly
        results['data_with_anomalies']['anomaly_score'] = anomaly_scores
        
        return results
    
    def community_classification(self,
                               data: pd.DataFrame,
                               species_columns: List[str],
                               n_communities: int = None,
                               method: str = 'kmeans') -> Dict[str, Any]:
        """
        Classify vegetation communities using unsupervised learning.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with species abundance/presence
        species_columns : List[str]
            List of species columns
        n_communities : int, optional
            Number of communities to identify. If None, will be estimated
        method : str, optional
            Clustering method ('kmeans', 'dbscan'), by default 'kmeans'
            
        Returns
        -------
        Dict[str, Any]
            Community classification results
        """
# Copyright (c) 2025 Mohamed Z. Hatim
        X = data[species_columns].copy()
        X = X.fillna(0)  # Fill missing with 0 for species data
        
# Copyright (c) 2025 Mohamed Z. Hatim
        X_transformed = np.log1p(X)  # Log transformation for abundance data
        
# Copyright (c) 2025 Mohamed Z. Hatim
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_transformed)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if n_communities is None and method == 'kmeans':
            inertias = []
            K_range = range(2, min(11, len(X) // 2))
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=self.random_state)
                kmeans.fit(X_scaled)
                inertias.append(kmeans.inertia_)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            deltas = np.diff(inertias)
            delta_deltas = np.diff(deltas)
            n_communities = K_range[np.argmax(delta_deltas) + 2] if len(delta_deltas) > 0 else 3
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_communities, random_state=self.random_state)
            cluster_labels = clusterer.fit_predict(X_scaled)
            cluster_centers = clusterer.cluster_centers_
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = clusterer.fit_predict(X_scaled)
            unique_labels = np.unique(cluster_labels)
            n_communities = len(unique_labels[unique_labels != -1])  # Exclude noise
            cluster_centers = None
        else:
            raise ValueError(f"Unknown method: {method}")
        
# Copyright (c) 2025 Mohamed Z. Hatim
        communities = {}
        for i in range(n_communities):
            if method == 'dbscan':
                mask = cluster_labels == i
            else:
                mask = cluster_labels == i
            
            if np.any(mask):
                community_data = X[mask]
                dominant_species = community_data.mean().nlargest(5).index.tolist()
                
                communities[f'Community_{i}'] = {
                    'n_sites': np.sum(mask),
                    'dominant_species': dominant_species,
                    'mean_abundance': community_data.mean(),
                    'total_abundance': community_data.sum().sum(),
                    'species_richness': (community_data > 0).sum(axis=1).mean()
                }
        
        results = {
            'clusterer': clusterer,
            'cluster_labels': cluster_labels,
            'n_communities': n_communities,
            'communities': communities,
            'cluster_centers': cluster_centers,
            'species_names': species_columns
        }
        
# Copyright (c) 2025 Mohamed Z. Hatim
        results['data_with_clusters'] = data.copy()
        results['data_with_clusters']['community'] = cluster_labels
        
        return results
    
    def dimensionality_reduction(self,
                               data: pd.DataFrame,
                               feature_columns: List[str],
                               method: str = 'pca',
                               n_components: int = 2) -> Dict[str, Any]:
        """
        Perform dimensionality reduction for visualization and analysis.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        feature_columns : List[str]
            List of feature columns
        method : str, optional
            Reduction method ('pca', 'tsne'), by default 'pca'
        n_components : int, optional
            Number of components, by default 2
            
        Returns
        -------
        Dict[str, Any]
            Dimensionality reduction results
        """
# Copyright (c) 2025 Mohamed Z. Hatim
        X = data[feature_columns].copy()
        X = X.fillna(X.mean())
        
# Copyright (c) 2025 Mohamed Z. Hatim
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=self.random_state)
            X_reduced = reducer.fit_transform(X_scaled)
            explained_variance = reducer.explained_variance_ratio_
            loadings = pd.DataFrame(
                reducer.components_.T,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=feature_columns
            )
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=self.random_state)
            X_reduced = reducer.fit_transform(X_scaled)
            explained_variance = None
            loadings = None
        else:
            raise ValueError(f"Unknown method: {method}")
        
        results = {
            'reducer': reducer,
            'reduced_data': X_reduced,
            'explained_variance': explained_variance,
            'loadings': loadings,
            'feature_names': feature_columns,
            'method': method,
            'n_components': n_components
        }
        
# Copyright (c) 2025 Mohamed Z. Hatim
        component_names = [f'{method.upper()}{i+1}' for i in range(n_components)]
        results['reduced_df'] = pd.DataFrame(
            X_reduced,
            columns=component_names,
            index=data.index
        )
        
        return results
    
    def plot_model_performance(self, results: Dict[str, Any], model_type: str) -> plt.Figure:
        """
        Plot model performance metrics.
        
        Parameters
        ----------
        results : Dict[str, Any]
            Model results from analysis methods
        model_type : str
            Type of model ('classification', 'regression')
            
        Returns
        -------
        plt.Figure
            Performance plot figure
        """
        if model_type == 'classification':
            return self._plot_classification_performance(results)
        elif model_type == 'regression':
            return self._plot_regression_performance(results)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _plot_classification_performance(self, results: Dict[str, Any]) -> plt.Figure:
        """Plot classification model performance."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if 'feature_importance' in results:
            importance = results['feature_importance']
            if isinstance(importance, dict) and importance:
# Copyright (c) 2025 Mohamed Z. Hatim
                if isinstance(list(importance.values())[0], dict):
                    best_model = results.get('best_model', list(importance.keys())[0])
                    importance = importance[best_model]
                
                features = list(importance.keys())[:10]  # Top 10
                values = [importance[f] for f in features]
                
                axes[0, 0].barh(features, values)
                axes[0, 0].set_title('Feature Importance')
                axes[0, 0].set_xlabel('Importance')
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if 'predictions' in results:
            pred_data = results['predictions']
            if isinstance(pred_data, dict) and 'y_true' in pred_data:
                y_true = pred_data['y_true']
                y_pred = pred_data['y_pred']
                
                confusion_df = pd.crosstab(y_true, y_pred)
                sns.heatmap(confusion_df, annot=True, fmt='d', ax=axes[0, 1])
                axes[0, 1].set_title('Confusion Matrix')
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if 'performance' in results:
            perf = results['performance']
            if isinstance(perf, dict) and 'accuracy' in perf:
                metrics = ['accuracy']
                values = [perf['accuracy']]
                
                axes[1, 0].bar(metrics, values)
                axes[1, 0].set_title('Performance Metrics')
                axes[1, 0].set_ylim(0, 1)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        axes[1, 1].remove()
        
        plt.tight_layout()
        return fig
    
    def _plot_regression_performance(self, results: Dict[str, Any]) -> plt.Figure:
        """Plot regression model performance."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if 'feature_importance' in results:
            importance = results['feature_importance']
            if importance:
                features = list(importance.keys())[:10]  # Top 10
                values = [importance[f] for f in features]
                
                axes[0, 0].barh(features, values)
                axes[0, 0].set_title('Feature Importance')
                axes[0, 0].set_xlabel('Importance')
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if 'predictions' in results:
            y_true = results['predictions']['y_true']
            y_pred = results['predictions']['y_pred']
            
            axes[0, 1].scatter(y_true, y_pred, alpha=0.6)
            axes[0, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
            axes[0, 1].set_xlabel('Actual')
            axes[0, 1].set_ylabel('Predicted')
            axes[0, 1].set_title('Actual vs Predicted')
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if 'predictions' in results:
            residuals = y_true - y_pred
            axes[1, 0].scatter(y_pred, residuals, alpha=0.6)
            axes[1, 0].axhline(y=0, color='r', linestyle='--')
            axes[1, 0].set_xlabel('Predicted')
            axes[1, 0].set_ylabel('Residuals')
            axes[1, 0].set_title('Residual Plot')
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if 'performance' in results:
            perf = results['performance']
            metrics = list(perf.keys())
            values = list(perf.values())
            
            axes[1, 1].bar(metrics, values)
            axes[1, 1].set_title('Performance Metrics')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig


class PredictiveModeling:
    """
    Specialized class for predictive modeling in ecology.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize PredictiveModeling."""
        self.random_state = random_state
        self.models = {}
        self.predictions = {}
    
    def species_distribution_modeling(self,
                                    presence_data: pd.DataFrame,
                                    environmental_data: pd.DataFrame,
                                    species_column: str,
                                    coordinate_columns: List[str],
                                    environmental_columns: List[str]) -> Dict[str, Any]:
        """
        Model species distribution using environmental variables.
        
        Parameters
        ----------
        presence_data : pd.DataFrame
            Species presence/absence data
        environmental_data : pd.DataFrame
            Environmental variables
        species_column : str
            Column containing species presence/absence
        coordinate_columns : List[str]
            Coordinate columns ['longitude', 'latitude']
        environmental_columns : List[str]
            Environmental variable columns
            
        Returns
        -------
        Dict[str, Any]
            Species distribution modeling results
        """
# Copyright (c) 2025 Mohamed Z. Hatim
        merged_data = presence_data.merge(
            environmental_data,
            on=coordinate_columns,
            how='inner'
        )
        
# Copyright (c) 2025 Mohamed Z. Hatim
        X = merged_data[environmental_columns + coordinate_columns]
        y = merged_data[species_column]
        
# Copyright (c) 2025 Mohamed Z. Hatim
        X = X.fillna(X.mean())
        
# Copyright (c) 2025 Mohamed Z. Hatim
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=self.random_state, stratify=y
        )
        
# Copyright (c) 2025 Mohamed Z. Hatim
        models = {
            'logistic': LogisticRegression(random_state=self.random_state),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'svm': SVC(probability=True, random_state=self.random_state)
        }
        
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            results[name] = {
                'model': model,
                'accuracy': accuracy_score(y_test, y_pred),
                'predictions': y_pred,
                'probabilities': y_proba,
                'y_test': y_test
            }
        
# Copyright (c) 2025 Mohamed Z. Hatim
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        
        return {
            'models': results,
            'best_model': best_model_name,
            'scaler': scaler,
            'feature_names': environmental_columns + coordinate_columns,
            'merged_data': merged_data
        }
    
    def climate_change_projections(self,
                                 current_data: pd.DataFrame,
                                 future_climate: pd.DataFrame,
                                 species_column: str,
                                 climate_variables: List[str]) -> Dict[str, Any]:
        """
        Project species distributions under climate change scenarios.
        
        Parameters
        ----------
        current_data : pd.DataFrame
            Current species and climate data
        future_climate : pd.DataFrame
            Future climate projections
        species_column : str
            Species presence/absence column
        climate_variables : List[str]
            Climate variable columns
            
        Returns
        -------
        Dict[str, Any]
            Climate change projection results
        """
# Copyright (c) 2025 Mohamed Z. Hatim
        X_current = current_data[climate_variables]
        y_current = current_data[species_column]
        
# Copyright (c) 2025 Mohamed Z. Hatim
        X_current = X_current.fillna(X_current.mean())
        
# Copyright (c) 2025 Mohamed Z. Hatim
        scaler = StandardScaler()
        X_current_scaled = scaler.fit_transform(X_current)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        model = RandomForestClassifier(n_estimators=200, random_state=self.random_state)
        model.fit(X_current_scaled, y_current)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        X_future = future_climate[climate_variables]
        X_future = X_future.fillna(X_future.mean())
        X_future_scaled = scaler.transform(X_future)
        
        future_predictions = model.predict(X_future_scaled)
        future_probabilities = model.predict_proba(X_future_scaled)[:, 1]
        
# Copyright (c) 2025 Mohamed Z. Hatim
        current_suitability = model.predict_proba(X_current_scaled)[:, 1].mean()
        future_suitability = future_probabilities.mean()
        suitability_change = future_suitability - current_suitability
        
        return {
            'model': model,
            'scaler': scaler,
            'current_suitability': current_suitability,
            'future_suitability': future_suitability,
            'suitability_change': suitability_change,
            'future_predictions': future_predictions,
            'future_probabilities': future_probabilities,
            'feature_importance': dict(zip(climate_variables, model.feature_importances_))
        }


# Copyright (c) 2025 Mohamed Z. Hatim
def quick_ml_analysis(data: pd.DataFrame, 
                     target_column: str, 
                     feature_columns: List[str] = None,
                     analysis_type: str = 'classification') -> Dict[str, Any]:
    """
    Quick machine learning analysis for vegetation data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    target_column : str
        Target variable column
    feature_columns : List[str], optional
        Feature columns. If None, use all except target
    analysis_type : str, optional
        Type of analysis ('classification', 'regression'), by default 'classification'
        
    Returns
    -------
    Dict[str, Any]
        Analysis results
    """
    ml_analyzer = MachineLearningAnalyzer()
    
    if analysis_type == 'classification':
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        
        return ml_analyzer.species_identification(
            data, feature_columns, target_column
        )
    elif analysis_type == 'regression':
        return ml_analyzer.biomass_prediction(
            data, target_column, feature_columns
        )
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")