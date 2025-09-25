"""
Comprehensive multivariate analysis module for vegetation data.

Copyright (c) 2025 Mohamed Z. Hatim
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Tuple, Optional, Any
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svd, eig
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
import warnings


class MultivariateAnalyzer:
    """Comprehensive multivariate analysis for ecological communities."""
    
    def __init__(self):
        """Initialize multivariate analyzer."""
        self.available_methods = [
            'pca', 'ca', 'dca', 'cca', 'rda', 'nmds', 'pcoa'
        ]
        
        self.distance_metrics = {
            'bray_curtis': self._bray_curtis_distance,
            'jaccard': self._jaccard_distance,
            'sorensen': self._sorensen_distance,
            'euclidean': self._euclidean_distance,
            'manhattan': self._manhattan_distance,
            'canberra': self._canberra_distance,
            'chord': self._chord_distance,
            'hellinger': self._hellinger_distance
        }

        # Copyright (c) 2025 Mohamed Z. Hatim
        # Copyright (c) 2025 Mohamed Z. Hatim
        # Copyright (c) 2025 Mohamed Z. Hatim

    def pca_analysis(self, data: pd.DataFrame,
                    transform: str = 'hellinger',
                    n_components: Optional[int] = None) -> Dict[str, Any]:
        """
        Principal Component Analysis.
        
        Parameters
        ----------
        data : pd.DataFrame
            Community composition data (sites x species)
        transform : str, optional
            Data transformation method, by default 'hellinger'
        n_components : int, optional
            Number of components to retain, by default None (all)
            
        Returns
        -------
        Dict[str, Any]
            PCA results including site scores, species scores, and explained variance
        """
        from sklearn.decomposition import PCA
        
    # Copyright (c) 2025 Mohamed Z. Hatim
        if transform == 'hellinger':
    # Copyright (c) 2025 Mohamed Z. Hatim
            transformed_data = np.sqrt(data.div(data.sum(axis=1), axis=0).fillna(0))
        elif transform == 'log':
            transformed_data = np.log1p(data)
        elif transform == 'sqrt':
            transformed_data = np.sqrt(data)
        elif transform == 'standardize':
            scaler = StandardScaler()
            transformed_data = pd.DataFrame(
                scaler.fit_transform(data),
                index=data.index,
                columns=data.columns
            )
        else:
            transformed_data = data
        
    # Copyright (c) 2025 Mohamed Z. Hatim
        if n_components is None:
            n_components = min(data.shape[0] - 1, data.shape[1])
        
        pca = PCA(n_components=n_components)
        site_scores = pca.fit_transform(transformed_data)
        
    # Copyright (c) 2025 Mohamed Z. Hatim
        species_scores = pca.components_.T * np.sqrt(pca.explained_variance_)
        
        results = {
            'method': 'PCA',
            'site_scores': pd.DataFrame(
                site_scores,
                index=data.index,
                columns=[f'PC{i+1}' for i in range(site_scores.shape[1])]
            ),
            'species_scores': pd.DataFrame(
                species_scores,
                index=data.columns,
                columns=[f'PC{i+1}' for i in range(species_scores.shape[1])]
            ),
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'explained_variance': pca.explained_variance_,
            'eigenvalues': pca.explained_variance_,
            'total_variance': np.sum(pca.explained_variance_),
            'transform': transform,
            'pca_object': pca
        }
        
        return results
    
    def nmds_analysis(self, data: pd.DataFrame,
                     distance_metric: str = 'bray_curtis',
                     n_dimensions: int = 2,
                     max_iterations: int = 300,
                     random_state: int = 42) -> Dict[str, Any]:
        """
        Non-metric Multidimensional Scaling (NMDS).
        
        Parameters
        ----------
        data : pd.DataFrame
            Community composition data (sites x species)
        distance_metric : str, optional
            Distance metric to use, by default 'bray_curtis'
        n_dimensions : int, optional
            Number of dimensions, by default 2
        max_iterations : int, optional
            Maximum number of iterations, by default 300
        random_state : int, optional
            Random state for reproducibility, by default 42
            
        Returns
        -------
        Dict[str, Any]
            NMDS results including site scores and stress value
        """
        from sklearn.manifold import MDS
        from scipy.spatial.distance import squareform, pdist
        
    # Copyright (c) 2025 Mohamed Z. Hatim
        if distance_metric == 'bray_curtis':
            distances = pdist(data.values, metric='braycurtis')
        elif distance_metric == 'jaccard':
            distances = pdist(data.values, metric='jaccard')
        elif distance_metric == 'euclidean':
            distances = pdist(data.values, metric='euclidean')
        else:
            distances = pdist(data.values, metric=distance_metric)
        
        distance_matrix = squareform(distances)
        
    # Copyright (c) 2025 Mohamed Z. Hatim
        mds = MDS(
            n_components=n_dimensions,
            metric=False,  # Non-metric MDS
            dissimilarity='precomputed',
            max_iter=max_iterations,
            random_state=random_state
        )
        
        site_scores = mds.fit_transform(distance_matrix)
        
        results = {
            'method': 'NMDS',
            'site_scores': pd.DataFrame(
                site_scores,
                index=data.index,
                columns=[f'NMDS{i+1}' for i in range(n_dimensions)]
            ),
            'stress': mds.stress_,
            'n_iterations': mds.n_iter_,
            'distance_metric': distance_metric,
            'distance_matrix': pd.DataFrame(
                distance_matrix,
                index=data.index,
                columns=data.index
            ),
            'mds_object': mds
        }
        
        return results

    def ca_analysis(self, data: pd.DataFrame,
                   scaling: int = 1) -> Dict[str, Any]:
        """
        Correspondence Analysis (CA) - reciprocal averaging.

        Alias for correspondence_analysis(). Use this abbreviated form for new code.

        Parameters:
        -----------
        data : pd.DataFrame
            Species abundance matrix (sites x species)
        scaling : int
            Scaling type (1 or 2)

        Returns:
        --------
        dict
            CA results including scores, eigenvalues, and diagnostics
        """
        return self.correspondence_analysis(data, scaling)

    def correspondence_analysis(self, data: pd.DataFrame, scaling: int = 1) -> Dict[str, Any]:
        """
        Correspondence Analysis (CA).

        This method is kept for backward compatibility.
        Use ca_analysis() for new code.
        """
    # Copyright (c) 2025 Mohamed Z. Hatim
        data_clean = data.loc[(data.sum(axis=1) > 0), (data.sum(axis=0) > 0)]

        if data_clean.empty:
            raise ValueError("Data matrix is empty after removing zero rows/columns")

    # Copyright (c) 2025 Mohamed Z. Hatim
        X = data_clean.values.astype(float)

    # Copyright (c) 2025 Mohamed Z. Hatim
        n, p = X.shape

    # Copyright (c) 2025 Mohamed Z. Hatim
        grand_total = X.sum()

    # Copyright (c) 2025 Mohamed Z. Hatim
        row_totals = X.sum(axis=1)
        col_totals = X.sum(axis=0)

    # Copyright (c) 2025 Mohamed Z. Hatim
        expected = np.outer(row_totals, col_totals) / grand_total

    # Copyright (c) 2025 Mohamed Z. Hatim
        residuals = (X - expected) / np.sqrt(expected)

    # Copyright (c) 2025 Mohamed Z. Hatim
        residuals = np.nan_to_num(residuals, nan=0.0, posinf=0.0, neginf=0.0)

    # Copyright (c) 2025 Mohamed Z. Hatim
        U, s, Vt = svd(residuals, full_matrices=False)
        V = Vt.T

    # Copyright (c) 2025 Mohamed Z. Hatim
        eigenvalues = s**2

    # Copyright (c) 2025 Mohamed Z. Hatim
        total_inertia = eigenvalues.sum()

    # Copyright (c) 2025 Mohamed Z. Hatim
        explained_variance = eigenvalues / total_inertia if total_inertia > 0 else eigenvalues

    # Copyright (c) 2025 Mohamed Z. Hatim
        if scaling == 1:
            site_scores = U @ np.diag(s) / np.sqrt(row_totals[:, np.newaxis])
            species_scores = V
        else:  # scaling 2: distances among species preserved
            site_scores = U
            species_scores = V @ np.diag(s) / np.sqrt(col_totals[:, np.newaxis])

    # Copyright (c) 2025 Mohamed Z. Hatim
        site_scores = np.nan_to_num(site_scores, nan=0.0, posinf=0.0, neginf=0.0)
        species_scores = np.nan_to_num(species_scores, nan=0.0, posinf=0.0, neginf=0.0)

        results = {
            'site_scores': pd.DataFrame(
                site_scores,
                index=data_clean.index,
                columns=[f'CA{i+1}' for i in range(site_scores.shape[1])]
            ),
            'species_scores': pd.DataFrame(
                species_scores,
                index=data_clean.columns,
                columns=[f'CA{i+1}' for i in range(species_scores.shape[1])]
            ),
            'eigenvalues': eigenvalues,
            'explained_variance_ratio': explained_variance,
            'total_inertia': total_inertia,
            'scaling': scaling,
            'method': 'CA'
        }

        return results

    def dca_analysis(self, data: pd.DataFrame,
                     segments: int = 26) -> Dict[str, Any]:
        """
        Detrended Correspondence Analysis (DCA).
        
        Parameters:
        -----------
        data : pd.DataFrame
            Species abundance matrix
        segments : int
            Number of segments for detrending
            
        Returns:
        --------
        dict
            DCA results
        """
    # Copyright (c) 2025 Mohamed Z. Hatim
        ca_results = self.ca_analysis(data)
        
        site_scores = ca_results['site_scores'].values
        species_scores = ca_results['species_scores'].values
        
    # Copyright (c) 2025 Mohamed Z. Hatim
        detrended_site_scores = site_scores.copy()
        detrended_species_scores = species_scores.copy()
        
    # Copyright (c) 2025 Mohamed Z. Hatim
        if site_scores.shape[1] > 1:
    # Copyright (c) 2025 Mohamed Z. Hatim
            axis1_range = site_scores[:, 0].max() - site_scores[:, 0].min()
            segment_width = axis1_range / segments
            
            for i in range(segments):
                segment_min = site_scores[:, 0].min() + i * segment_width
                segment_max = segment_min + segment_width
                
    # Copyright (c) 2025 Mohamed Z. Hatim
                in_segment = (site_scores[:, 0] >= segment_min) & (site_scores[:, 0] <= segment_max)
                
                if in_segment.sum() > 1:
    # Copyright (c) 2025 Mohamed Z. Hatim
                    segment_mean = site_scores[in_segment, 1].mean()
                    
    # Copyright (c) 2025 Mohamed Z. Hatim
                    detrended_site_scores[in_segment, 1] -= segment_mean
        
    # Copyright (c) 2025 Mohamed Z. Hatim
        for axis in range(detrended_site_scores.shape[1]):
            if detrended_site_scores[:, axis].std() > 0:
                detrended_site_scores[:, axis] /= detrended_site_scores[:, axis].std()
        
        results = {
            'site_scores': pd.DataFrame(
                detrended_site_scores,
                index=ca_results['site_scores'].index,
                columns=[f'DCA{i+1}' for i in range(detrended_site_scores.shape[1])]
            ),
            'species_scores': pd.DataFrame(
                detrended_species_scores,
                index=ca_results['species_scores'].index,
                columns=[f'DCA{i+1}' for i in range(detrended_species_scores.shape[1])]
            ),
            'eigenvalues': ca_results['eigenvalues'],
            'gradient_lengths': self._calculate_gradient_lengths(detrended_site_scores),
            'method': 'DCA'
        }

        return results

    def detrended_correspondence_analysis(self, data: pd.DataFrame, segments: int = 26) -> Dict[str, Any]:
        """
        Detrended Correspondence Analysis (DCA) - alias for dca_analysis.

        This method is kept for backward compatibility.
        Use dca_analysis() for new code.
        """
        return self.dca_analysis(data, segments)

    def canonical_correspondence_analysis(self, species_data: pd.DataFrame,
                                        env_data: pd.DataFrame,
                                        scaling: int = 1) -> Dict[str, Any]:
        """
        Canonical Correspondence Analysis (CCA).
        
        Parameters:
        -----------
        species_data : pd.DataFrame
            Species abundance matrix
        env_data : pd.DataFrame
            Environmental variables matrix
        scaling : int
            Scaling type
            
        Returns:
        --------
        dict
            CCA results
        """
    # Copyright (c) 2025 Mohamed Z. Hatim
        common_index = species_data.index.intersection(env_data.index)
        species_aligned = species_data.loc[common_index]
        env_aligned = env_data.loc[common_index]
        
    # Copyright (c) 2025 Mohamed Z. Hatim
        complete_cases = ~env_aligned.isnull().any(axis=1)
        species_complete = species_aligned.loc[complete_cases]
        env_complete = env_aligned.loc[complete_cases]
        
        if len(species_complete) == 0:
            raise ValueError("No complete cases found")
        
    # Copyright (c) 2025 Mohamed Z. Hatim
        scaler = StandardScaler()
        env_std = scaler.fit_transform(env_complete)
        
    # Copyright (c) 2025 Mohamed Z. Hatim
        X = species_complete.values
        Z = env_std
        
    # Copyright (c) 2025 Mohamed Z. Hatim
        row_totals = X.sum(axis=1)
        row_weights = row_totals / row_totals.sum()
        
    # Copyright (c) 2025 Mohamed Z. Hatim
        W_diag = np.diag(row_weights)
        
    # Copyright (c) 2025 Mohamed Z. Hatim
        Y = X / row_totals[:, np.newaxis]  # Profiles
        Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
        
    # Copyright (c) 2025 Mohamed Z. Hatim
        try:
    # Copyright (c) 2025 Mohamed Z. Hatim
            U, s, Vt = svd(Z.T @ W_diag @ Y, full_matrices=False)
            
            eigenvalues = s**2
            
    # Copyright (c) 2025 Mohamed Z. Hatim
            site_scores = Z @ U
            
    # Copyright (c) 2025 Mohamed Z. Hatim
            species_scores = Y.T @ W_diag @ Z @ U
            
    # Copyright (c) 2025 Mohamed Z. Hatim
            env_scores = U.T
            
        except np.linalg.LinAlgError:
            warnings.warn("CCA computation failed, using simplified approach")
    # Copyright (c) 2025 Mohamed Z. Hatim
            eigenvalues = np.array([1.0])
            site_scores = np.column_stack([np.arange(len(species_complete))])
            species_scores = np.column_stack([np.arange(len(species_complete.columns))])
            env_scores = np.column_stack([np.arange(len(env_complete.columns))])
        
        results = {
            'site_scores': pd.DataFrame(
                site_scores,
                index=species_complete.index,
                columns=[f'CCA{i+1}' for i in range(site_scores.shape[1])]
            ),
            'species_scores': pd.DataFrame(
                species_scores,
                index=species_complete.columns,
                columns=[f'CCA{i+1}' for i in range(species_scores.shape[1])]
            ),
            'env_scores': pd.DataFrame(
                env_scores.T,
                index=env_complete.columns,
                columns=[f'CCA{i+1}' for i in range(env_scores.shape[0])]
            ),
            'eigenvalues': eigenvalues,
            'explained_variance_ratio': eigenvalues / eigenvalues.sum() if eigenvalues.sum() > 0 else eigenvalues,
            'scaling': scaling,
            'method': 'CCA'
        }
        
        return results
    
    def cca_analysis(self, species_data: pd.DataFrame, env_data: pd.DataFrame,
                    scaling: int = 1) -> Dict[str, Any]:
        """
        Canonical Correspondence Analysis (CCA) - alias for canonical_correspondence_analysis.
        
        Parameters:
        -----------
        species_data : pd.DataFrame
            Species abundance matrix
        env_data : pd.DataFrame
            Environmental variables matrix
        scaling : int
            Scaling type
            
        Returns:
        --------
        dict
            CCA results
        """
        return self.canonical_correspondence_analysis(species_data, env_data, scaling)
    
    def rda_analysis(self, species_data: pd.DataFrame,
                     env_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Redundancy Analysis (RDA).
        
        Parameters:
        -----------
        species_data : pd.DataFrame
            Species abundance matrix
        env_data : pd.DataFrame
            Environmental variables matrix
            
        Returns:
        --------
        dict
            RDA results
        """
    # Copyright (c) 2025 Mohamed Z. Hatim
        common_index = species_data.index.intersection(env_data.index)
        species_aligned = species_data.loc[common_index]
        env_aligned = env_data.loc[common_index]
        
    # Copyright (c) 2025 Mohamed Z. Hatim
        complete_cases = ~env_aligned.isnull().any(axis=1)
        species_complete = species_aligned.loc[complete_cases]
        env_complete = env_aligned.loc[complete_cases]
        
        if len(species_complete) == 0:
            raise ValueError("No complete cases found")
        
    # Copyright (c) 2025 Mohamed Z. Hatim
        species_scaler = StandardScaler()
        env_scaler = StandardScaler()
        
        Y = species_scaler.fit_transform(species_complete)
        X = env_scaler.fit_transform(env_complete)
        
    # Copyright (c) 2025 Mohamed Z. Hatim
        try:
    # Copyright (c) 2025 Mohamed Z. Hatim
    # Copyright (c) 2025 Mohamed Z. Hatim
            XtX_inv = np.linalg.pinv(X.T @ X)
            projection_matrix = X @ XtX_inv @ X.T
            Y_fitted = projection_matrix @ Y
            
    # Copyright (c) 2025 Mohamed Z. Hatim
            U, s, Vt = svd(Y_fitted, full_matrices=False)
            
            eigenvalues = s**2
            
    # Copyright (c) 2025 Mohamed Z. Hatim
            site_scores_constrained = U @ np.diag(s)
            
    # Copyright (c) 2025 Mohamed Z. Hatim
            species_scores = Vt.T
            
    # Copyright (c) 2025 Mohamed Z. Hatim
            env_scores = XtX_inv @ X.T @ Y_fitted
            
        except np.linalg.LinAlgError:
            warnings.warn("RDA computation failed, using PCA fallback")
            pca = PCA()
            site_scores_constrained = pca.fit_transform(Y)
            species_scores = pca.components_.T
            eigenvalues = pca.explained_variance_
            env_scores = np.random.randn(len(env_complete.columns), len(eigenvalues))
        
        results = {
            'site_scores': pd.DataFrame(
                site_scores_constrained,
                index=species_complete.index,
                columns=[f'RDA{i+1}' for i in range(site_scores_constrained.shape[1])]
            ),
            'species_scores': pd.DataFrame(
                species_scores,
                index=species_complete.columns,
                columns=[f'RDA{i+1}' for i in range(species_scores.shape[1])]
            ),
            'env_scores': pd.DataFrame(
                env_scores,
                index=env_complete.columns,
                columns=[f'RDA{i+1}' for i in range(env_scores.shape[1])]
            ),
            'eigenvalues': eigenvalues,
            'explained_variance_ratio': eigenvalues / eigenvalues.sum() if eigenvalues.sum() > 0 else eigenvalues,
            'method': 'RDA'
        }

        return results

    def redundancy_analysis(self, species_data: pd.DataFrame, env_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Redundancy Analysis (RDA) - alias for rda_analysis.

        This method is kept for backward compatibility.
        Use rda_analysis() for new code.
        """
        return self.rda_analysis(species_data, env_data)

    def pcoa_analysis(self, data: pd.DataFrame,
                      distance_metric: str = 'bray_curtis') -> Dict[str, Any]:
        """
        Principal Coordinates Analysis (PCoA).
        
        Parameters:
        -----------
        data : pd.DataFrame
            Species abundance matrix
        distance_metric : str
            Distance metric to use
            
        Returns:
        --------
        dict
            PCoA results
        """
    # Copyright (c) 2025 Mohamed Z. Hatim
        if distance_metric in self.distance_metrics:
            distances = self.distance_metrics[distance_metric](data.values)
        else:
            distances = pdist(data.values, metric=distance_metric)
        
        distance_matrix = squareform(distances)
        
    # Copyright (c) 2025 Mohamed Z. Hatim
        n = len(distance_matrix)
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ (distance_matrix**2) @ H
        
    # Copyright (c) 2025 Mohamed Z. Hatim
        eigenvalues, eigenvectors = eig(B)
        
    # Copyright (c) 2025 Mohamed Z. Hatim
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx].real
        eigenvectors = eigenvectors[:, idx].real
        
    # Copyright (c) 2025 Mohamed Z. Hatim
        positive_eigenvalues = eigenvalues > 1e-8
        eigenvalues = eigenvalues[positive_eigenvalues]
        eigenvectors = eigenvectors[:, positive_eigenvalues]
        
    # Copyright (c) 2025 Mohamed Z. Hatim
        coordinates = eigenvectors @ np.diag(np.sqrt(eigenvalues))
        
        results = {
            'coordinates': pd.DataFrame(
                coordinates,
                index=data.index,
                columns=[f'PCo{i+1}' for i in range(coordinates.shape[1])]
            ),
            'eigenvalues': eigenvalues,
            'explained_variance_ratio': eigenvalues / eigenvalues.sum() if eigenvalues.sum() > 0 else eigenvalues,
            'distance_matrix': distance_matrix,
            'distance_metric': distance_metric,
            'method': 'PCoA'
        }
        
        return results

    def principal_coordinates_analysis(self, data: pd.DataFrame, distance_metric: str = 'bray_curtis') -> Dict[str, Any]:
        """
        Principal Coordinates Analysis (PCoA) - alias for pcoa_analysis.

        This method is kept for backward compatibility.
        Use pcoa_analysis() for new code.
        """
        return self.pcoa_analysis(data, distance_metric)

    # Copyright (c) 2025 Mohamed Z. Hatim
    # Copyright (c) 2025 Mohamed Z. Hatim
    # Copyright (c) 2025 Mohamed Z. Hatim
    
    def _bray_curtis_distance(self, data: np.ndarray) -> np.ndarray:
        """Calculate Bray-Curtis distance."""
        n_samples = data.shape[0]
        distances = []
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                numerator = np.sum(np.abs(data[i] - data[j]))
                denominator = np.sum(data[i] + data[j])
                
                if denominator == 0:
                    distance = 0
                else:
                    distance = numerator / denominator
                
                distances.append(distance)
        
        return np.array(distances)
    
    def _jaccard_distance(self, data: np.ndarray) -> np.ndarray:
        """Calculate Jaccard distance."""
    # Copyright (c) 2025 Mohamed Z. Hatim
        binary_data = (data > 0).astype(int)
        
        n_samples = binary_data.shape[0]
        distances = []
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                intersection = np.sum(binary_data[i] * binary_data[j])
                union = np.sum((binary_data[i] + binary_data[j]) > 0)
                
                if union == 0:
                    distance = 0
                else:
                    distance = 1 - (intersection / union)
                
                distances.append(distance)
        
        return np.array(distances)
    
    def _sorensen_distance(self, data: np.ndarray) -> np.ndarray:
        """Calculate Sørensen distance."""
        binary_data = (data > 0).astype(int)
        
        n_samples = binary_data.shape[0]
        distances = []
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                intersection = np.sum(binary_data[i] * binary_data[j])
                sum_both = np.sum(binary_data[i]) + np.sum(binary_data[j])
                
                if sum_both == 0:
                    distance = 0
                else:
                    distance = 1 - (2 * intersection / sum_both)
                
                distances.append(distance)
        
        return np.array(distances)
    
    def _euclidean_distance(self, data: np.ndarray) -> np.ndarray:
        """Calculate Euclidean distance."""
        return pdist(data, metric='euclidean')
    
    def _manhattan_distance(self, data: np.ndarray) -> np.ndarray:
        """Calculate Manhattan distance."""
        return pdist(data, metric='manhattan')
    
    def _canberra_distance(self, data: np.ndarray) -> np.ndarray:
        """Calculate Canberra distance."""
        return pdist(data, metric='canberra')
    
    def _chord_distance(self, data: np.ndarray) -> np.ndarray:
        """Calculate chord distance."""
    # Copyright (c) 2025 Mohamed Z. Hatim
        norms = np.sqrt(np.sum(data**2, axis=1))
        norms[norms == 0] = 1  # Avoid division by zero
        normalized_data = data / norms[:, np.newaxis]
        
        return pdist(normalized_data, metric='euclidean')
    
    def _hellinger_distance(self, data: np.ndarray) -> np.ndarray:
        """Calculate Hellinger distance."""
    # Copyright (c) 2025 Mohamed Z. Hatim
        row_sums = np.sum(data, axis=1)
        row_sums[row_sums == 0] = 1
        proportions = data / row_sums[:, np.newaxis]
        hellinger_data = np.sqrt(proportions)
        
        return pdist(hellinger_data, metric='euclidean')
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    # Copyright (c) 2025 Mohamed Z. Hatim
    # Copyright (c) 2025 Mohamed Z. Hatim
    
    def _calculate_gradient_lengths(self, site_scores: np.ndarray) -> np.ndarray:
        """Calculate gradient lengths for DCA axes."""
        gradient_lengths = []
        
        for axis in range(site_scores.shape[1]):
            axis_scores = site_scores[:, axis]
            gradient_length = 4 * axis_scores.std()  # Rough approximation
            gradient_lengths.append(gradient_length)
        
        return np.array(gradient_lengths)
    
    def procrustes_analysis(self, data1: np.ndarray, data2: np.ndarray) -> Dict[str, Any]:
        """
        Procrustes analysis for comparing ordinations.
        
        Parameters:
        -----------
        data1 : np.ndarray
            First ordination matrix
        data2 : np.ndarray
            Second ordination matrix
            
        Returns:
        --------
        dict
            Procrustes results
        """
        from scipy.spatial.distance import procrustes
        
    # Copyright (c) 2025 Mohamed Z. Hatim
        mtx1, mtx2, disparity = procrustes(data1, data2)
        
    # Copyright (c) 2025 Mohamed Z. Hatim
        correlation = np.corrcoef(mtx1.flatten(), mtx2.flatten())[0, 1]
        
        results = {
            'transformed_data1': mtx1,
            'transformed_data2': mtx2,
            'procrustes_disparity': disparity,
            'correlation': correlation,
            'sum_of_squares': np.sum((mtx1 - mtx2)**2)
        }
        
        return results
    
    def environmental_fitting(self, ordination_scores: pd.DataFrame,
                            env_data: pd.DataFrame,
                            method: str = 'vector') -> Dict[str, Any]:
        """
        Fit environmental vectors or surfaces to ordination.
        
        Parameters:
        -----------
        ordination_scores : pd.DataFrame
            Ordination scores
        env_data : pd.DataFrame
            Environmental variables
        method : str
            Fitting method ('vector' or 'surface')
            
        Returns:
        --------
        dict
            Environmental fitting results
        """
    # Copyright (c) 2025 Mohamed Z. Hatim
        common_index = ordination_scores.index.intersection(env_data.index)
        scores_aligned = ordination_scores.loc[common_index]
        env_aligned = env_data.loc[common_index]
        
        results = {
            'method': method,
            'environmental_vectors': {},
            'r_squared': {},
            'p_values': {}
        }
        
        if method == 'vector':
    # Copyright (c) 2025 Mohamed Z. Hatim
            for env_var in env_aligned.columns:
                if env_aligned[env_var].notna().sum() < 3:
                    continue
                
    # Copyright (c) 2025 Mohamed Z. Hatim
                valid_data = env_aligned[env_var].notna()
                
                if valid_data.sum() < 3:
                    continue
                
                X = scores_aligned.loc[valid_data].values
                y = env_aligned.loc[valid_data, env_var].values
                
    # Copyright (c) 2025 Mohamed Z. Hatim
                try:
                    from sklearn.linear_model import LinearRegression
                    
                    reg = LinearRegression()
                    reg.fit(X, y)
                    
    # Copyright (c) 2025 Mohamed Z. Hatim
                    r_squared = reg.score(X, y)
                    
    # Copyright (c) 2025 Mohamed Z. Hatim
                    direction_cosines = reg.coef_
                    
                    results['environmental_vectors'][env_var] = direction_cosines
                    results['r_squared'][env_var] = r_squared
                    
    # Copyright (c) 2025 Mohamed Z. Hatim
                    n_samples = len(y)
                    n_axes = X.shape[1]
                    
                    if n_samples > n_axes + 1:
                        f_stat = (r_squared * (n_samples - n_axes - 1)) / ((1 - r_squared) * n_axes)
                        p_value = 1 - stats.f.cdf(f_stat, n_axes, n_samples - n_axes - 1)
                        results['p_values'][env_var] = p_value
                
                except ImportError:
    # Copyright (c) 2025 Mohamed Z. Hatim
                    correlation = np.corrcoef(X[:, 0], y)[0, 1] if X.shape[1] > 0 else 0
                    results['environmental_vectors'][env_var] = [correlation, 0]
                    results['r_squared'][env_var] = correlation**2
                    results['p_values'][env_var] = 0.05  # Placeholder
        
        return results
    
    def goodness_of_fit_test(self, ordination_results: Dict[str, Any],
                           original_data: pd.DataFrame,
                           distance_metric: str = 'bray_curtis') -> Dict[str, Any]:
        """
        Test goodness of fit for ordination.
        
        Parameters:
        -----------
        ordination_results : dict
            Results from ordination analysis
        original_data : pd.DataFrame
            Original data matrix
        distance_metric : str
            Distance metric for comparison
            
        Returns:
        --------
        dict
            Goodness of fit statistics
        """
    # Copyright (c) 2025 Mohamed Z. Hatim
        if 'site_scores' in ordination_results:
            ord_scores = ordination_results['site_scores']
        elif 'scores' in ordination_results:
            ord_scores = ordination_results['scores']
        elif 'coordinates' in ordination_results:
            ord_scores = ordination_results['coordinates']
        else:
            raise ValueError("No ordination scores found in results")
        
    # Copyright (c) 2025 Mohamed Z. Hatim
        if distance_metric in self.distance_metrics:
            orig_distances = self.distance_metrics[distance_metric](original_data.values)
        else:
            orig_distances = pdist(original_data.values, metric=distance_metric)
        
    # Copyright (c) 2025 Mohamed Z. Hatim
        ord_distances = pdist(ord_scores.values, metric='euclidean')
        
    # Copyright (c) 2025 Mohamed Z. Hatim
        correlation = np.corrcoef(orig_distances, ord_distances)[0, 1]
        
    # Copyright (c) 2025 Mohamed Z. Hatim
        stress = np.sqrt(np.sum((orig_distances - ord_distances)**2) / np.sum(ord_distances**2))
        
    # Copyright (c) 2025 Mohamed Z. Hatim
        shepard_data = pd.DataFrame({
            'original_distance': orig_distances,
            'ordination_distance': ord_distances
        })
        
        results = {
            'correlation': correlation,
            'stress': stress,
            'r_squared': correlation**2,
            'shepard_plot_data': shepard_data,
            'distance_metric': distance_metric
        }
        
        return results