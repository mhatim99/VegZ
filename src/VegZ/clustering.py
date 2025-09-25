"""
Comprehensive clustering analysis module for vegetation classification.

Copyright (c) 2025 Mohamed Z. Hatim
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Tuple, Optional, Any
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, cophenet
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import warnings


class VegetationClustering:
    """Comprehensive clustering analysis for vegetation classification."""
    
    def __init__(self):
        """Initialize clustering analyzer."""
        self.clustering_methods = {
            'hierarchical': self.hierarchical_clustering,
            'kmeans': self.kmeans_clustering,
            'twinspan': self.twinspan,
            'fuzzy_cmeans': self.fuzzy_cmeans_clustering,
            'dbscan': self.dbscan_clustering,
            'gaussian_mixture': self.gaussian_mixture_clustering
        }
        
        self.validation_metrics = {
            'silhouette': self._silhouette_analysis,
            'calinski_harabasz': self._calinski_harabasz_score,
            'gap_statistic': self._gap_statistic,
            'cophenetic': self._cophenetic_correlation
        }
    
    # Copyright (c) 2025 Mohamed Z. Hatim

    
    def twinspan(self, data: pd.DataFrame,
                 cut_levels: List[float] = [0, 2, 5, 10, 20],
                 max_divisions: int = 6,
                 min_group_size: int = 5) -> Dict[str, Any]:
        """
        Two-Way Indicator Species Analysis (TWINSPAN).
        
        Parameters:
        -----------
        data : pd.DataFrame
            Species abundance matrix (sites x species)
        cut_levels : list
            Pseudospecies cut levels
        max_divisions : int
            Maximum number of divisions
        min_group_size : int
            Minimum group size for division
            
        Returns:
        --------
        dict
            TWINSPAN results including classification and indicator species
        """
        # Copyright (c) 2025 Mohamed Z. Hatim
        pseudo_species_data = self._create_pseudospecies(data, cut_levels)
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        classification_tree = {
            'divisions': [],
            'groups': {},
            'indicator_species': {}
        }
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        initial_group = {
            'sites': data.index.tolist(),
            'level': 0,
            'parent': None,
            'eigenvalue': 0
        }
        
        groups_to_process = [initial_group]
        group_id = 1
        
        for division in range(max_divisions):
            if not groups_to_process:
                break
            
            # Copyright (c) 2025 Mohamed Z. Hatim
            current_group = max(groups_to_process, key=lambda x: x['eigenvalue'])
            groups_to_process.remove(current_group)
            
            if len(current_group['sites']) < min_group_size * 2:
# Copyright (c) 2025 Mohamed Z. Hatim
                classification_tree['groups'][group_id] = current_group
                group_id += 1
                continue
            
            # Copyright (c) 2025 Mohamed Z. Hatim
            division_result = self._twinspan_division(
                pseudo_species_data.loc[current_group['sites']]
            )
            
            if division_result['eigenvalue'] < 0.1:  # Minimum eigenvalue threshold
                classification_tree['groups'][group_id] = current_group
                group_id += 1
                continue
            
            # Copyright (c) 2025 Mohamed Z. Hatim
            group1_sites = [current_group['sites'][i] for i in division_result['group1_indices']]
            group2_sites = [current_group['sites'][i] for i in division_result['group2_indices']]
            
            group1 = {
                'sites': group1_sites,
                'level': current_group['level'] + 1,
                'parent': group_id,
                'eigenvalue': division_result['eigenvalue'] * 0.8  # Decay for next iteration
            }
            
            group2 = {
                'sites': group2_sites,
                'level': current_group['level'] + 1,
                'parent': group_id,
                'eigenvalue': division_result['eigenvalue'] * 0.8
            }
            
            # Copyright (c) 2025 Mohamed Z. Hatim
            if len(group1_sites) >= min_group_size:
                groups_to_process.append(group1)
            else:
                classification_tree['groups'][group_id + 1] = group1
            
            if len(group2_sites) >= min_group_size:
                groups_to_process.append(group2)
            else:
                classification_tree['groups'][group_id + 2] = group2
            
            # Copyright (c) 2025 Mohamed Z. Hatim
            classification_tree['divisions'].append({
                'division_id': division,
                'parent_group': group_id,
                'child_groups': [group_id + 1, group_id + 2],
                'eigenvalue': division_result['eigenvalue'],
                'indicator_species': division_result['indicator_species']
            })
            
            group_id += 3
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        for remaining_group in groups_to_process:
            classification_tree['groups'][group_id] = remaining_group
            group_id += 1
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        site_classification = self._assign_final_groups(classification_tree, data.index)
        
        results = {
            'site_classification': site_classification,
            'classification_tree': classification_tree,
            'pseudospecies_data': pseudo_species_data,
            'cut_levels': cut_levels,
            'method': 'TWINSPAN'
        }
        
        return results
    
    def _create_pseudospecies(self, data: pd.DataFrame, 
                            cut_levels: List[float]) -> pd.DataFrame:
        """Create pseudospecies from abundance data."""
        pseudo_data = pd.DataFrame(index=data.index)
        
        for species in data.columns:
            species_data = data[species]
            
            for i, cut_level in enumerate(cut_levels[1:], 1):
                pseudo_name = f"{species}_{i}"
                pseudo_data[pseudo_name] = (species_data >= cut_level).astype(int)
        
        return pseudo_data
    
    def _twinspan_division(self, pseudo_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform a single TWINSPAN division."""
        # Copyright (c) 2025 Mohamed Z. Hatim
        try:
            # Copyright (c) 2025 Mohamed Z. Hatim
            data_matrix = pseudo_data.values.astype(float)
            
            if data_matrix.sum() == 0:
                return {
                    'group1_indices': list(range(len(pseudo_data)//2)),
                    'group2_indices': list(range(len(pseudo_data)//2, len(pseudo_data))),
                    'eigenvalue': 0.0,
                    'indicator_species': []
                }
            
            # Copyright (c) 2025 Mohamed Z. Hatim
            row_totals = data_matrix.sum(axis=1)
            col_totals = data_matrix.sum(axis=0)
            
            # Copyright (c) 2025 Mohamed Z. Hatim
            row_totals[row_totals == 0] = 1
            col_totals[col_totals == 0] = 1
            
            # Copyright (c) 2025 Mohamed Z. Hatim
            row_profiles = data_matrix / row_totals[:, np.newaxis]
            col_profiles = data_matrix.T / col_totals[:, np.newaxis]
            
            # Copyright (c) 2025 Mohamed Z. Hatim
            scores = np.ones(len(pseudo_data))
            
            for iteration in range(20):  # Power iteration
                old_scores = scores.copy()
                
                # Copyright (c) 2025 Mohamed Z. Hatim
                species_scores = col_profiles @ scores
                species_scores = species_scores / np.linalg.norm(species_scores)
                
                # Copyright (c) 2025 Mohamed Z. Hatim
                scores = row_profiles @ species_scores
                
                # Copyright (c) 2025 Mohamed Z. Hatim
                if np.corrcoef(scores, old_scores)[0, 1] > 0.999:
                    break
            
            # Copyright (c) 2025 Mohamed Z. Hatim
            sorted_indices = np.argsort(scores)
            best_division = len(scores) // 2
            best_eigenvalue = 0
            
            # Copyright (c) 2025 Mohamed Z. Hatim
            for div_point in range(len(scores)//4, 3*len(scores)//4):
                group1_idx = sorted_indices[:div_point]
                group2_idx = sorted_indices[div_point:]
                
                if len(group1_idx) < 2 or len(group2_idx) < 2:
                    continue
                
                # Copyright (c) 2025 Mohamed Z. Hatim
                group1_mean = scores[group1_idx].mean()
                group2_mean = scores[group2_idx].mean()
                separation = abs(group1_mean - group2_mean)
                
                if separation > best_eigenvalue:
                    best_eigenvalue = separation
                    best_division = div_point
            
            group1_indices = sorted_indices[:best_division].tolist()
            group2_indices = sorted_indices[best_division:].tolist()
            
            # Copyright (c) 2025 Mohamed Z. Hatim
            indicator_species = self._identify_indicator_pseudospecies(
                pseudo_data, group1_indices, group2_indices
            )
            
        except Exception as e:
            warnings.warn(f"TWINSPAN division failed: {e}")
            # Copyright (c) 2025 Mohamed Z. Hatim
            mid_point = len(pseudo_data) // 2
            group1_indices = list(range(mid_point))
            group2_indices = list(range(mid_point, len(pseudo_data)))
            best_eigenvalue = 0.1
            indicator_species = []
        
        return {
            'group1_indices': group1_indices,
            'group2_indices': group2_indices,
            'eigenvalue': best_eigenvalue,
            'indicator_species': indicator_species
        }
    
    def _identify_indicator_pseudospecies(self, pseudo_data: pd.DataFrame,
                                        group1_indices: List[int],
                                        group2_indices: List[int]) -> List[str]:
        """Identify indicator pseudospecies for division."""
        indicators = []
        
        for species in pseudo_data.columns:
            group1_freq = pseudo_data.iloc[group1_indices][species].mean()
            group2_freq = pseudo_data.iloc[group2_indices][species].mean()
            
            # Copyright (c) 2025 Mohamed Z. Hatim
            freq_diff = abs(group1_freq - group2_freq)
            
            if freq_diff > 0.3:  # Threshold for indicator species
                indicators.append({
                    'species': species,
                    'frequency_difference': freq_diff,
                    'group1_frequency': group1_freq,
                    'group2_frequency': group2_freq
                })
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        indicators.sort(key=lambda x: x['frequency_difference'], reverse=True)
        
        return indicators[:5]  # Copyright (c) 2025 Mohamed Z. Hatim
    
    def _assign_final_groups(self, classification_tree: Dict[str, Any],
                           site_index: pd.Index) -> pd.Series:
        """Assign sites to final groups."""
        site_groups = pd.Series(index=site_index, dtype=int)
        
        group_counter = 1
        for group_id, group_info in classification_tree['groups'].items():
            for site in group_info['sites']:
                site_groups[site] = group_counter
            group_counter += 1
        
        return site_groups
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    
    def fuzzy_cmeans_clustering(self, data: pd.DataFrame,
                               n_clusters: int = 3,
                               fuzziness: float = 2.0,
                               max_iter: int = 100,
                               tol: float = 1e-4) -> Dict[str, Any]:
        """
        Fuzzy C-means clustering.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data matrix
        n_clusters : int
            Number of clusters
        fuzziness : float
            Fuzziness parameter (> 1)
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
            
        Returns:
        --------
        dict
            Fuzzy clustering results
        """
        X = data.values
        n_samples, n_features = X.shape
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        membership = np.random.rand(n_samples, n_clusters)
        membership = membership / membership.sum(axis=1)[:, np.newaxis]
        
        centers = np.zeros((n_clusters, n_features))
        
        for iteration in range(max_iter):
            # Copyright (c) 2025 Mohamed Z. Hatim
            for c in range(n_clusters):
                weights = membership[:, c] ** fuzziness
                centers[c] = (weights[:, np.newaxis] * X).sum(axis=0) / weights.sum()
            
            # Copyright (c) 2025 Mohamed Z. Hatim
            new_membership = np.zeros((n_samples, n_clusters))
            
            for i in range(n_samples):
                distances = np.linalg.norm(X[i] - centers, axis=1)
                distances[distances == 0] = 1e-10  # Avoid division by zero
                
                for c in range(n_clusters):
                    sum_term = np.sum((distances[c] / distances) ** (2 / (fuzziness - 1)))
                    new_membership[i, c] = 1 / sum_term
            
            # Copyright (c) 2025 Mohamed Z. Hatim
            if np.max(np.abs(membership - new_membership)) < tol:
                break
            
            membership = new_membership
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        hard_clusters = np.argmax(membership, axis=1)
        
        results = {
            'membership_matrix': pd.DataFrame(
                membership,
                index=data.index,
                columns=[f'Cluster_{i+1}' for i in range(n_clusters)]
            ),
            'cluster_centers': pd.DataFrame(
                centers,
                columns=data.columns,
                index=[f'Cluster_{i+1}' for i in range(n_clusters)]
            ),
            'hard_clusters': pd.Series(hard_clusters + 1, index=data.index, name='cluster'),
            'fuzziness_parameter': fuzziness,
            'n_iterations': iteration + 1,
            'method': 'Fuzzy_C_means'
        }
        
        return results
    
    def dbscan_clustering(self, data: pd.DataFrame,
                         eps: float = 0.5,
                         min_samples: int = 5,
                         distance_metric: str = 'euclidean') -> Dict[str, Any]:
        """
        DBSCAN clustering for identifying core communities.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data matrix
        eps : float
            Maximum distance between samples
        min_samples : int
            Minimum samples in neighborhood
        distance_metric : str
            Distance metric
            
        Returns:
        --------
        dict
            DBSCAN results
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=distance_metric)
        cluster_labels = dbscan.fit_predict(data.values)
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        core_samples = np.zeros_like(cluster_labels, dtype=bool)
        core_samples[dbscan.core_sample_indices_] = True
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        results = {
            'cluster_labels': pd.Series(cluster_labels, index=data.index, name='cluster'),
            'core_samples': pd.Series(core_samples, index=data.index, name='core_sample'),
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'eps': eps,
            'min_samples': min_samples,
            'method': 'DBSCAN'
        }
        
        return results
    
    def gaussian_mixture_clustering(self, data: pd.DataFrame,
                                  n_components: int = 3,
                                  covariance_type: str = 'full',
                                  max_iter: int = 100) -> Dict[str, Any]:
        """
        Gaussian Mixture Model clustering.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data matrix
        n_components : int
            Number of mixture components
        covariance_type : str
            Covariance type ('full', 'tied', 'diag', 'spherical')
        max_iter : int
            Maximum EM iterations
            
        Returns:
        --------
        dict
            GMM results
        """
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            random_state=42
        )
        
        cluster_labels = gmm.fit_predict(data.values)
        probabilities = gmm.predict_proba(data.values)
        
        results = {
            'cluster_labels': pd.Series(cluster_labels, index=data.index, name='cluster'),
            'probabilities': pd.DataFrame(
                probabilities,
                index=data.index,
                columns=[f'Component_{i+1}' for i in range(n_components)]
            ),
            'means': gmm.means_,
            'covariances': gmm.covariances_,
            'weights': gmm.weights_,
            'aic': gmm.aic(data.values),
            'bic': gmm.bic(data.values),
            'log_likelihood': gmm.score(data.values),
            'method': 'Gaussian_Mixture'
        }
        
        return results
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    
    def _silhouette_analysis(self, data: pd.DataFrame, 
                           labels: pd.Series) -> Dict[str, Any]:
        """Silhouette analysis for cluster validation."""
        if len(set(labels)) < 2:
            return {'mean_silhouette_score': 0, 'silhouette_scores': None}
        
        silhouette_avg = silhouette_score(data.values, labels)
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        from sklearn.metrics import silhouette_samples
        silhouette_scores = silhouette_samples(data.values, labels)
        
        results = {
            'mean_silhouette_score': silhouette_avg,
            'silhouette_scores': pd.Series(silhouette_scores, index=data.index)
        }
        
        return results
    
    def _calinski_harabasz_score(self, data: pd.DataFrame,
                               labels: pd.Series) -> float:
        """Calinski-Harabasz index."""
        if len(set(labels)) < 2:
            return 0
        
        return calinski_harabasz_score(data.values, labels)
    
    def _gap_statistic(self, data: pd.DataFrame,
                      k_range: range = range(1, 11),
                      n_refs: int = 10) -> Dict[str, Any]:
        """Gap statistic for optimal number of clusters."""
        gaps = []
        errors = []
        
        for k in k_range:
            # Copyright (c) 2025 Mohamed Z. Hatim
            if k == 1:
                wk_actual = np.sum((data.values - data.values.mean(axis=0))**2)
            else:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(data.values)
                wk_actual = kmeans.inertia_
            
            # Copyright (c) 2025 Mohamed Z. Hatim
            wk_refs = []
            
            for _ in range(n_refs):
                # Copyright (c) 2025 Mohamed Z. Hatim
                random_data = np.random.uniform(
                    data.values.min(axis=0),
                    data.values.max(axis=0),
                    size=data.values.shape
                )
                
                if k == 1:
                    wk_ref = np.sum((random_data - random_data.mean(axis=0))**2)
                else:
                    kmeans_ref = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans_ref.fit(random_data)
                    wk_ref = kmeans_ref.inertia_
                
                wk_refs.append(wk_ref)
            
            # Copyright (c) 2025 Mohamed Z. Hatim
            gap = np.log(np.mean(wk_refs)) - np.log(wk_actual)
            gaps.append(gap)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            se = np.std(np.log(wk_refs)) * np.sqrt(1 + 1/n_refs)
            errors.append(se)
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        optimal_k = 1
        for i in range(len(gaps) - 1):
            if gaps[i] >= gaps[i + 1] - errors[i + 1]:
                optimal_k = k_range[i]
                break
        
        results = {
            'gap_values': gaps,
            'standard_errors': errors,
            'optimal_k': optimal_k,
            'k_range': list(k_range)
        }
        
        return results
    
    def _cophenetic_correlation(self, data: pd.DataFrame,
                              linkage_matrix: np.ndarray) -> float:
        """Cophenetic correlation coefficient."""
        # Copyright (c) 2025 Mohamed Z. Hatim
        distances = pdist(data.values)
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        cophenetic_distances = cophenet(linkage_matrix)
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        correlation = np.corrcoef(distances, cophenetic_distances)[0, 1]
        
        return correlation
    
    def optimal_clusters_analysis(self, data: pd.DataFrame,
                                k_range: range = range(2, 11),
                                methods: List[str] = ['silhouette', 'gap_statistic']) -> Dict[str, Any]:
        """
        Comprehensive analysis for optimal number of clusters.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data matrix
        k_range : range
            Range of cluster numbers to test
        methods : list
            Validation methods to use
            
        Returns:
        --------
        dict
            Optimal cluster analysis results
        """
        results = {
            'k_range': list(k_range),
            'validation_scores': {},
            'recommendations': {}
        }
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        if 'silhouette' in methods:
            silhouette_scores = []
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(data.values)
                score = silhouette_score(data.values, labels)
                silhouette_scores.append(score)
            
            results['validation_scores']['silhouette'] = silhouette_scores
            optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
            results['recommendations']['silhouette'] = optimal_k_silhouette
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        if 'gap_statistic' in methods:
            gap_results = self._gap_statistic(data, k_range)
            results['validation_scores']['gap_statistic'] = gap_results['gap_values']
            results['recommendations']['gap_statistic'] = gap_results['optimal_k']
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        if 'elbow' in methods:
            wcss = []
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(data.values)
                wcss.append(kmeans.inertia_)
            
            results['validation_scores']['wcss'] = wcss
            
            # Copyright (c) 2025 Mohamed Z. Hatim
            if len(wcss) > 2:
                diffs = np.diff(wcss)
                diff2 = np.diff(diffs)
                elbow_idx = np.argmax(diff2) + 2  # +2 because of double diff
                if elbow_idx < len(k_range):
                    results['recommendations']['elbow'] = k_range[elbow_idx]
        
        return results
    
    def hierarchical_clustering(self, data: pd.DataFrame,
                               method: str = 'ward',
                               metric: str = 'euclidean',
                               n_clusters: Optional[int] = None) -> Dict[str, Any]:
        """
        Enhanced hierarchical clustering with validation.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data matrix
        method : str
            Linkage method
        metric : str
            Distance metric
        n_clusters : int, optional
            Number of clusters to extract
            
        Returns:
        --------
        dict
            Hierarchical clustering results with validation
        """
        # Copyright (c) 2025 Mohamed Z. Hatim
        if metric == 'precomputed':
            distances = data.values
        else:
            distances = pdist(data.values, metric=metric)
        
        linkage_matrix = linkage(distances, method=method)
        
        results = {
            'linkage_matrix': linkage_matrix,
            'method': method,
            'metric': metric
        }
        
        if n_clusters:
            # Copyright (c) 2025 Mohamed Z. Hatim
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            results['cluster_labels'] = pd.Series(cluster_labels, index=data.index, name='cluster')
            
            # Copyright (c) 2025 Mohamed Z. Hatim
            if len(set(cluster_labels)) > 1:
                silhouette_results = self._silhouette_analysis(data, cluster_labels)
                results['silhouette_score'] = silhouette_results['mean_silhouette_score']
                results['silhouette_scores'] = silhouette_results['silhouette_scores']
                
                results['calinski_harabasz_score'] = self._calinski_harabasz_score(data, cluster_labels)
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        results['cophenetic_correlation'] = self._cophenetic_correlation(data, linkage_matrix)
        
        return results
    
    def kmeans_clustering(self, data: pd.DataFrame,
                         n_clusters: int = 3,
                         n_init: int = 10,
                         max_iter: int = 300) -> Dict[str, Any]:
        """
        Enhanced K-means clustering with validation.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data matrix
        n_clusters : int
            Number of clusters
        n_init : int
            Number of initializations
        max_iter : int
            Maximum iterations
            
        Returns:
        --------
        dict
            K-means results with validation
        """
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            random_state=42
        )
        
        cluster_labels = kmeans.fit_predict(data.values)
        
        results = {
            'cluster_labels': pd.Series(cluster_labels, index=data.index, name='cluster'),
            'cluster_centers': pd.DataFrame(
                kmeans.cluster_centers_,
                columns=data.columns,
                index=[f'Cluster_{i}' for i in range(n_clusters)]
            ),
            'inertia': kmeans.inertia_,
            'n_iter': kmeans.n_iter_,
            'method': 'K_means'
        }
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        if len(set(cluster_labels)) > 1:
            silhouette_results = self._silhouette_analysis(data, cluster_labels)
            results['silhouette_score'] = silhouette_results['mean_silhouette_score']
            results['silhouette_scores'] = silhouette_results['silhouette_scores']
            
            results['calinski_harabasz_score'] = self._calinski_harabasz_score(data, cluster_labels)
        
        return results
    
    def optimal_k_analysis(self, data: pd.DataFrame, 
                          k_range: range = range(2, 11),
                          methods: List[str] = ['elbow', 'silhouette', 'gap']) -> Dict[str, Any]:
        """
        Find optimal number of clusters using multiple methods.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data matrix
        k_range : range
            Range of k values to test
        methods : list
            Methods to use ('elbow', 'silhouette', 'gap')
            
        Returns:
        --------
        dict
            Optimal k analysis results
        """
        results = {
            'k_range': list(k_range),
            'metrics': {},
            'optimal_k': {},
            'recommendations': {}
        }
        
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        
        for k in k_range:
            # Copyright (c) 2025 Mohamed Z. Hatim
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data.values)
            
            # Copyright (c) 2025 Mohamed Z. Hatim
            inertias.append(kmeans.inertia_)
            
            if len(set(labels)) > 1:  # Need more than 1 cluster for these metrics
                silhouette_scores.append(silhouette_score(data.values, labels))
                calinski_scores.append(calinski_harabasz_score(data.values, labels))
            else:
                silhouette_scores.append(0)
                calinski_scores.append(0)
        
        results['metrics']['inertia'] = inertias
        results['metrics']['silhouette_scores'] = silhouette_scores
        results['metrics']['calinski_harabasz_scores'] = calinski_scores
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        if 'elbow' in methods:
            optimal_k_elbow = self._find_elbow_point(list(k_range), inertias)
            results['optimal_k']['elbow'] = optimal_k_elbow
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        if 'silhouette' in methods:
            optimal_k_silhouette = list(k_range)[np.argmax(silhouette_scores)]
            results['optimal_k']['silhouette'] = optimal_k_silhouette
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        if 'gap' in methods:
            gap_stats = self._calculate_gap_statistic(data, k_range)
            optimal_k_gap = list(k_range)[np.argmax(gap_stats)]
            results['optimal_k']['gap'] = optimal_k_gap
            results['metrics']['gap_statistic'] = gap_stats
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        optimal_ks = list(results['optimal_k'].values())
        if optimal_ks:
            # Copyright (c) 2025 Mohamed Z. Hatim
            from collections import Counter
            counter = Counter(optimal_ks)
            most_common = counter.most_common(1)[0]
            if most_common[1] > 1:  # Copyright (c) 2025 Mohamed Z. Hatim
                results['recommendations']['consensus'] = most_common[0]
            else:
                results['recommendations']['consensus'] = int(np.median(optimal_ks))
        
        return results
    
    def comprehensive_elbow_analysis(self, data: pd.DataFrame,
                                    k_range: range = range(1, 16),
                                    methods: List[str] = ['knee_locator', 'derivative', 'variance_explained', 'distortion_jump'],
                                    transform: str = 'hellinger',
                                    plot_results: bool = True) -> Dict[str, Any]:
        """
        Comprehensive elbow analysis with multiple detection algorithms.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Species abundance matrix (sites x species)
        k_range : range
            Range of k values to test
        methods : list
            Elbow detection methods to use
        transform : str
            Data transformation method
        plot_results : bool
            Whether to create visualization plots
            
        Returns:
        --------
        dict
            Comprehensive elbow analysis results
        """
        # Copyright (c) 2025 Mohamed Z. Hatim
        if transform == 'hellinger':
            # Copyright (c) 2025 Mohamed Z. Hatim
            row_sums = data.sum(axis=1)
            row_sums[row_sums == 0] = 1
            transformed_data = np.sqrt(data.div(row_sums, axis=0).fillna(0))
        elif transform == 'log':
            transformed_data = np.log1p(data)
        elif transform == 'sqrt':
            transformed_data = np.sqrt(data)
        else:
            transformed_data = data
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        davies_bouldin_scores = []
        
        for k in k_range:
            if k == 1:
                inertias.append(self._calculate_total_variance(transformed_data.values))
                silhouette_scores.append(0)
                calinski_scores.append(0)
                davies_bouldin_scores.append(float('inf'))
            else:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(transformed_data.values)
                
                inertias.append(kmeans.inertia_)
                
                try:
                    silhouette_scores.append(silhouette_score(transformed_data.values, labels))
                    calinski_scores.append(calinski_harabasz_score(transformed_data.values, labels))
                    from sklearn.metrics import davies_bouldin_score
                    davies_bouldin_scores.append(davies_bouldin_score(transformed_data.values, labels))
                except:
                    silhouette_scores.append(0)
                    calinski_scores.append(0)
                    davies_bouldin_scores.append(float('inf'))
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        results = {
            'k_values': list(k_range),
            'metrics': {
                'inertia': inertias,
                'silhouette_scores': silhouette_scores,
                'calinski_harabasz_scores': calinski_scores,
                'davies_bouldin_scores': davies_bouldin_scores
            },
            'elbow_points': {},
            'method_details': {},
            'recommendations': {}
        }
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        if 'knee_locator' in methods:
            elbow_k = self._knee_locator_method(list(k_range), inertias)
            results['elbow_points']['knee_locator'] = elbow_k
            results['method_details']['knee_locator'] = {
                'description': 'Kneedle algorithm for automatic knee/elbow detection',
                'reference': 'Satopaa et al. (2011)'
            }
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        if 'derivative' in methods:
            elbow_k = self._derivative_elbow_method(list(k_range), inertias)
            results['elbow_points']['derivative'] = elbow_k
            results['method_details']['derivative'] = {
                'description': 'Second derivative maximum for curvature detection'
            }
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        if 'variance_explained' in methods:
            elbow_k = self._variance_explained_elbow(list(k_range), inertias)
            results['elbow_points']['variance_explained'] = elbow_k
            results['method_details']['variance_explained'] = {
                'description': 'Point where additional clusters explain <10% more variance'
            }
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        if 'distortion_jump' in methods:
            elbow_k = self._distortion_jump_method(list(k_range), inertias)
            results['elbow_points']['distortion_jump'] = elbow_k
            results['method_details']['distortion_jump'] = {
                'description': 'Jump method based on distortion changes',
                'reference': 'Sugar & James (2003)'
            }
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        if 'l_method' in methods:
            elbow_k = self._l_method_elbow(list(k_range), inertias)
            results['elbow_points']['l_method'] = elbow_k
            results['method_details']['l_method'] = {
                'description': 'L-method for determining number of clusters',
                'reference': 'Salvador & Chan (2004)'
            }
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        elbow_points = [v for v in results['elbow_points'].values() if v is not None]
        if elbow_points:
            from collections import Counter
            counter = Counter(elbow_points)
            most_common = counter.most_common(1)[0]
            
            if most_common[1] > 1:  # Copyright (c) 2025 Mohamed Z. Hatim
                results['recommendations']['consensus'] = most_common[0]
            else:
                results['recommendations']['consensus'] = int(np.median(elbow_points))
            
            results['recommendations']['confidence'] = most_common[1] / len(elbow_points)
        else:
            results['recommendations']['consensus'] = None
            results['recommendations']['confidence'] = 0
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        if silhouette_scores:
            best_silhouette_idx = np.argmax(silhouette_scores[1:]) + 1  # Copyright (c) 2025 Mohamed Z. Hatim
            results['recommendations']['silhouette_optimal'] = list(k_range)[best_silhouette_idx]
        
        if calinski_scores:
            best_calinski_idx = np.argmax(calinski_scores[1:]) + 1  # Copyright (c) 2025 Mohamed Z. Hatim
            results['recommendations']['calinski_optimal'] = list(k_range)[best_calinski_idx]
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        if plot_results:
            results['plots'] = self._create_elbow_plots(results)
        
        return results
    
    def _calculate_total_variance(self, data: np.ndarray) -> float:
        """Calculate total variance of the dataset."""
        return np.sum(np.var(data, axis=0))
    
    def _knee_locator_method(self, k_values: List[int], inertias: List[float]) -> Optional[int]:
        """
        Knee locator method (Kneedle algorithm) for elbow detection.
        
        Based on: Satopaa, V., et al. (2011). "Finding a kneedle in a haystack: 
        Detecting knee points in system behavior."
        """
        if len(inertias) < 3:
            return None
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        x_norm = np.array(k_values, dtype=float)
        y_norm = np.array(inertias, dtype=float)
        
        x_norm = (x_norm - x_norm.min()) / (x_norm.max() - x_norm.min())
        y_norm = (y_norm - y_norm.min()) / (y_norm.max() - y_norm.min())
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        y_norm = 1 - y_norm
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        diagonal = x_norm
        differences = y_norm - diagonal
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        if len(differences) > 0:
            knee_idx = np.argmax(differences)
            return k_values[knee_idx]
        
        return None
    
    def _derivative_elbow_method(self, k_values: List[int], inertias: List[float]) -> Optional[int]:
        """Find elbow using second derivative method."""
        if len(inertias) < 3:
            return None
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        first_deriv = np.diff(inertias)
        second_deriv = np.diff(first_deriv)
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        if len(second_deriv) > 0:
            elbow_idx = np.argmax(np.abs(second_deriv)) + 1
            if elbow_idx < len(k_values):
                return k_values[elbow_idx]
        
        return None
    
    def _variance_explained_elbow(self, k_values: List[int], inertias: List[float]) -> Optional[int]:
        """Find elbow where additional clusters explain less than threshold variance."""
        if len(inertias) < 3:
            return None
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        total_variance = inertias[0]  
        threshold = 0.1  
        
        for i in range(1, len(inertias)):
            if i == len(inertias) - 1:  
                return k_values[i-1]
            
            current_explained = (total_variance - inertias[i]) / total_variance
            next_explained = (total_variance - inertias[i+1]) / total_variance
            
            improvement = next_explained - current_explained
            
            if improvement < threshold:
                return k_values[i]
        
        return None
    
    def _distortion_jump_method(self, k_values: List[int], inertias: List[float]) -> Optional[int]:
        """
        Jump method for elbow detection.
        
        Based on: Sugar, C. A., & James, G. M. (2003). "Finding the number of 
        clusters in a dataset: An information-theoretic approach."
        """
        if len(inertias) < 4:
            return None
        
# Copyright (c) 2025 Mohamed Z. Hatim
        distortions = np.array(inertias)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        jumps = []
        for i in range(1, len(distortions) - 1):
            jump = (distortions[i-1] - distortions[i]) - (distortions[i] - distortions[i+1])
            jumps.append(jump)
        
        if jumps:
# Copyright (c) 2025 Mohamed Z. Hatim
            max_jump_idx = np.argmax(jumps)
            return k_values[max_jump_idx + 1]  # +1 because jumps is offset
        
        return None
    
    def _l_method_elbow(self, k_values: List[int], inertias: List[float]) -> Optional[int]:
        """
        L-method for determining the number of clusters.
        
        Based on: Salvador, S., & Chan, P. (2004). "Determining the number of 
        clusters/segments in hierarchical clustering/segmentation algorithms."
        """
        if len(inertias) < 4:
            return None
        
        best_k = None
        best_score = float('inf')
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        for split_idx in range(2, len(inertias) - 1):
            # Copyright (c) 2025 Mohamed Z. Hatim
            left_x = np.array(k_values[:split_idx])
            left_y = np.array(inertias[:split_idx])
            right_x = np.array(k_values[split_idx:])
            right_y = np.array(inertias[split_idx:])
            
            # Copyright (c) 2025 Mohamed Z. Hatim
            try:
                left_coef = np.polyfit(left_x, left_y, 1)
                right_coef = np.polyfit(right_x, right_y, 1)
                
                # Copyright (c) 2025 Mohamed Z. Hatim
                left_pred = np.polyval(left_coef, left_x)
                right_pred = np.polyval(right_coef, right_x)
                
                left_r2 = 1 - np.sum((left_y - left_pred)**2) / np.sum((left_y - np.mean(left_y))**2)
                right_r2 = 1 - np.sum((right_y - right_pred)**2) / np.sum((right_y - np.mean(right_y))**2)
                
                # Copyright (c) 2025 Mohamed Z. Hatim
                left_weight = len(left_x) / len(k_values)
                right_weight = len(right_x) / len(k_values)
                
                combined_r2 = left_weight * left_r2 + right_weight * right_r2
                
                # Copyright (c) 2025 Mohamed Z. Hatim
                score = -combined_r2
                
                if score < best_score:
                    best_score = score
                    best_k = k_values[split_idx]
            
            except:
                continue
        
        return best_k
    
    def _create_elbow_plots(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive elbow analysis plots."""
        k_values = results['k_values']
        metrics = results['metrics']
        elbow_points = results['elbow_points']
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comprehensive Elbow Analysis for Optimal K Selection', fontsize=16)
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        ax1 = axes[0, 0]
        ax1.plot(k_values, metrics['inertia'], 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia (Within-cluster sum of squares)')
        ax1.set_title('Elbow Method - Inertia Curve')
        ax1.grid(True, alpha=0.3)
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        colors = ['red', 'orange', 'green', 'purple', 'brown']
        for i, (method, elbow_k) in enumerate(elbow_points.items()):
            if elbow_k and elbow_k in k_values:
                elbow_idx = k_values.index(elbow_k)
                ax1.axvline(x=elbow_k, color=colors[i % len(colors)], 
                           linestyle='--', alpha=0.7, label=f'{method}: k={elbow_k}')
        ax1.legend()
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        ax2 = axes[0, 1]
        silhouette_k = k_values[1:]  # Skip k=1
        silhouette_scores = metrics['silhouette_scores'][1:]
        ax2.plot(silhouette_k, silhouette_scores, 'go-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Average Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True, alpha=0.3)
        
        if silhouette_scores:
            best_silhouette_k = silhouette_k[np.argmax(silhouette_scores)]
            ax2.axvline(x=best_silhouette_k, color='red', linestyle='--', 
                       label=f'Best: k={best_silhouette_k}')
            ax2.legend()
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        ax3 = axes[1, 0]
        calinski_k = k_values[1:]  # Skip k=1
        calinski_scores = metrics['calinski_harabasz_scores'][1:]
        ax3.plot(calinski_k, calinski_scores, 'mo-', linewidth=2, markersize=8)
        ax3.set_xlabel('Number of Clusters (k)')
        ax3.set_ylabel('Calinski-Harabasz Index')
        ax3.set_title('Calinski-Harabasz Index')
        ax3.grid(True, alpha=0.3)
        
        if calinski_scores:
            best_calinski_k = calinski_k[np.argmax(calinski_scores)]
            ax3.axvline(x=best_calinski_k, color='red', linestyle='--',
                       label=f'Best: k={best_calinski_k}')
            ax3.legend()
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        ax4 = axes[1, 1]
        db_k = k_values[1:]  # Skip k=1
        db_scores = metrics['davies_bouldin_scores'][1:]
        # Copyright (c) 2025 Mohamed Z. Hatim
        finite_db = [(k, score) for k, score in zip(db_k, db_scores) if np.isfinite(score)]
        if finite_db:
            finite_k, finite_scores = zip(*finite_db)
            ax4.plot(finite_k, finite_scores, 'co-', linewidth=2, markersize=8)
            best_db_k = finite_k[np.argmin(finite_scores)]
            ax4.axvline(x=best_db_k, color='red', linestyle='--',
                       label=f'Best: k={best_db_k}')
            ax4.legend()
        
        ax4.set_xlabel('Number of Clusters (k)')
        ax4.set_ylabel('Davies-Bouldin Index')
        ax4.set_title('Davies-Bouldin Index (Lower is Better)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return {
            'figure': fig,
            'axes': axes,
            'description': 'Comprehensive elbow analysis with multiple metrics'
        }
    
    def _find_elbow_point(self, k_values: List[int], inertias: List[float]) -> int:
        """Find elbow point in inertia curve using knee locator method."""
        # Copyright (c) 2025 Mohamed Z. Hatim
        elbow_k = self._knee_locator_method(k_values, inertias)
        if elbow_k is not None:
            return elbow_k
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        if len(inertias) < 3:
            return k_values[0]
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        first_diff = np.diff(inertias)
        second_diff = np.diff(first_diff)
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        elbow_idx = np.argmax(np.abs(second_diff)) + 1  # +1 because of double diff
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        elbow_idx = min(elbow_idx, len(k_values) - 1)
        
        return k_values[elbow_idx]
    
    def _calculate_gap_statistic(self, data: pd.DataFrame, k_range: range) -> List[float]:
        """Calculate gap statistic for each k."""
        gap_stats = []
        
        for k in k_range:
            # Copyright (c) 2025 Mohamed Z. Hatim
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data.values)
            
            # Copyright (c) 2025 Mohamed Z. Hatim
            wcss_obs = 0
            for i in range(k):
                cluster_points = data.values[labels == i]
                if len(cluster_points) > 0:
                    centroid = np.mean(cluster_points, axis=0)
                    wcss_obs += np.sum((cluster_points - centroid) ** 2)
            
            # Copyright (c) 2025 Mohamed Z. Hatim
            n_refs = 10  # Number of reference datasets
            ref_wcss = []
            
            for _ in range(n_refs):
                # Copyright (c) 2025 Mohamed Z. Hatim
                ref_data = np.random.uniform(
                    low=data.values.min(axis=0),
                    high=data.values.max(axis=0),
                    size=data.shape
                )
                
                kmeans_ref = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels_ref = kmeans_ref.fit_predict(ref_data)
                
                wcss_ref = 0
                for i in range(k):
                    cluster_points = ref_data[labels_ref == i]
                    if len(cluster_points) > 0:
                        centroid = np.mean(cluster_points, axis=0)
                        wcss_ref += np.sum((cluster_points - centroid) ** 2)
                
                ref_wcss.append(wcss_ref)
            
            # Copyright (c) 2025 Mohamed Z. Hatim
            expected_wcss = np.mean(ref_wcss)
            gap = np.log(expected_wcss) - np.log(wcss_obs) if wcss_obs > 0 else 0
            gap_stats.append(gap)
        
        return gap_stats