"""
Statistical analysis module for multivariate ecological statistics.

Copyright (c) 2025 Mohamed Z. Hatim
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Tuple, Optional, Any
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csgraph
import warnings
from itertools import combinations


class EcologicalStatistics:
    """Comprehensive statistical analysis for ecological data."""
    
    def __init__(self):
        """Initialize statistical analyzer."""
        self.available_tests = [
            'permanova', 'anosim', 'mrpp', 'mantel',
            'partial_mantel', 'bioenv', 'simper'
        ]
        
        self.distance_metrics = {
            'bray_curtis': self._bray_curtis_distance,
            'jaccard': self._jaccard_distance,
            'euclidean': self._euclidean_distance,
            'manhattan': self._manhattan_distance
        }
    
    def calculate_distance_matrix(self, data: pd.DataFrame, 
                                 metric: str = 'bray_curtis') -> pd.DataFrame:
        """
        Calculate distance matrix using specified metric.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data matrix (samples x features)
        metric : str
            Distance metric to use
            
        Returns:
        --------
        pd.DataFrame
            Distance matrix
        """
        if metric in self.distance_metrics:
            distances = self.distance_metrics[metric](data.values)
        else:
# Copyright (c) 2025 Mohamed Z. Hatim
            try:
# Copyright (c) 2025 Mohamed Z. Hatim
                scipy_metric = metric.replace('_', '')
                if scipy_metric == 'braycurtis':
                    scipy_metric = 'braycurtis'
                distances = pdist(data.values, metric=scipy_metric)
            except ValueError:
# Copyright (c) 2025 Mohamed Z. Hatim
                distances = self._bray_curtis_distance(data.values)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        distance_matrix = squareform(distances)
        
        return pd.DataFrame(
            distance_matrix,
            index=data.index,
            columns=data.index
        )
    
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
    
    def permanova(self, distance_matrix: Union[pd.DataFrame, np.ndarray],
                  groups: Union[pd.Series, List],
                  permutations: int = 999) -> Dict[str, Any]:
        """
        Permutational Multivariate Analysis of Variance (PERMANOVA).
        
        Parameters:
        -----------
        distance_matrix : pd.DataFrame or np.ndarray
            Distance matrix or data matrix
        groups : pd.Series or list
            Group assignments
        permutations : int
            Number of permutations
            
        Returns:
        --------
        dict
            PERMANOVA results including F-statistic and p-value
        """
# Copyright (c) 2025 Mohamed Z. Hatim
        if isinstance(distance_matrix, pd.DataFrame):
            if distance_matrix.shape[0] == distance_matrix.shape[1]:
# Copyright (c) 2025 Mohamed Z. Hatim
                dist_matrix = distance_matrix.values
            else:
# Copyright (c) 2025 Mohamed Z. Hatim
                dist_matrix = squareform(pdist(distance_matrix.values))
        else:
            if distance_matrix.shape[0] == distance_matrix.shape[1]:
                dist_matrix = distance_matrix
            else:
                dist_matrix = squareform(pdist(distance_matrix))
        
        if isinstance(groups, pd.Series):
            group_labels = groups.values
        else:
            group_labels = np.array(groups)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        observed_f = self._calculate_permanova_f(dist_matrix, group_labels)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        permuted_f_stats = []
        n_samples = len(group_labels)
        
        for _ in range(permutations):
# Copyright (c) 2025 Mohamed Z. Hatim
            permuted_groups = np.random.permutation(group_labels)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            f_stat = self._calculate_permanova_f(dist_matrix, permuted_groups)
            permuted_f_stats.append(f_stat)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        p_value = (np.sum(np.array(permuted_f_stats) >= observed_f) + 1) / (permutations + 1)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        unique_groups = np.unique(group_labels)
        df_between = len(unique_groups) - 1
        df_within = n_samples - len(unique_groups)
        df_total = n_samples - 1
        
# Copyright (c) 2025 Mohamed Z. Hatim
        ss_total = self._calculate_total_sum_squares(dist_matrix)
        ss_between = self._calculate_between_sum_squares(dist_matrix, group_labels)
        ss_within = ss_total - ss_between
        
        r_squared = ss_between / ss_total if ss_total > 0 else 0
        
        results = {
            'f_statistic': observed_f,
            'p_value': p_value,
            'r_squared': r_squared,
            'df_between': df_between,
            'df_within': df_within,
            'df_total': df_total,
            'ss_between': ss_between,
            'ss_within': ss_within,
            'ss_total': ss_total,
            'permutations': permutations,
            'method': 'PERMANOVA'
        }
        
        return results
    
    def _calculate_permanova_f(self, dist_matrix: np.ndarray, 
                             group_labels: np.ndarray) -> float:
        """Calculate PERMANOVA F-statistic."""
        n_samples = len(group_labels)
        unique_groups = np.unique(group_labels)
        n_groups = len(unique_groups)
        
        if n_groups < 2:
            return 0.0
        
# Copyright (c) 2025 Mohamed Z. Hatim
        ss_total = self._calculate_total_sum_squares(dist_matrix)
        ss_between = self._calculate_between_sum_squares(dist_matrix, group_labels)
        ss_within = ss_total - ss_between
        
# Copyright (c) 2025 Mohamed Z. Hatim
        df_between = n_groups - 1
        df_within = n_samples - n_groups
        
        if df_within <= 0 or ss_within <= 0:
            return 0.0
        
# Copyright (c) 2025 Mohamed Z. Hatim
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        
        f_stat = ms_between / ms_within
        
        return f_stat
    
    def _calculate_total_sum_squares(self, dist_matrix: np.ndarray) -> float:
        """Calculate total sum of squares."""
        n = len(dist_matrix)
        return np.sum(dist_matrix**2) / (2 * n)
    
    def _calculate_between_sum_squares(self, dist_matrix: np.ndarray,
                                     group_labels: np.ndarray) -> float:
        """Calculate between-group sum of squares."""
        unique_groups = np.unique(group_labels)
        n_total = len(group_labels)
        
        ss_between = 0
        
        for group in unique_groups:
            group_mask = group_labels == group
            n_group = np.sum(group_mask)
            
            if n_group <= 1:
                continue
            
# Copyright (c) 2025 Mohamed Z. Hatim
            group_distances = dist_matrix[np.ix_(group_mask, group_mask)]
            ss_group = np.sum(group_distances**2) / (2 * n_group)
            
            ss_between += n_group * ss_group / n_total
        
        ss_total = self._calculate_total_sum_squares(dist_matrix)
        
        return ss_total - ss_between
    
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
    
    def anosim(self, distance_matrix: Union[pd.DataFrame, np.ndarray],
               groups: Union[pd.Series, List],
               permutations: int = 999) -> Dict[str, Any]:
        """
        Analysis of Similarities (ANOSIM).
        
        Parameters:
        -----------
        distance_matrix : pd.DataFrame or np.ndarray
            Distance matrix
        groups : pd.Series or list
            Group assignments
        permutations : int
            Number of permutations
            
        Returns:
        --------
        dict
            ANOSIM results including R-statistic and p-value
        """
# Copyright (c) 2025 Mohamed Z. Hatim
        if isinstance(distance_matrix, pd.DataFrame):
            dist_matrix = distance_matrix.values
        else:
            dist_matrix = distance_matrix
        
        if isinstance(groups, pd.Series):
            group_labels = groups.values
        else:
            group_labels = np.array(groups)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        observed_r = self._calculate_anosim_r(dist_matrix, group_labels)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        permuted_r_stats = []
        
        for _ in range(permutations):
            permuted_groups = np.random.permutation(group_labels)
            r_stat = self._calculate_anosim_r(dist_matrix, permuted_groups)
            permuted_r_stats.append(r_stat)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        p_value = (np.sum(np.array(permuted_r_stats) >= observed_r) + 1) / (permutations + 1)
        
        results = {
            'r_statistic': observed_r,
            'p_value': p_value,
            'permutations': permutations,
            'method': 'ANOSIM'
        }
        
        return results
    
    def _calculate_anosim_r(self, dist_matrix: np.ndarray,
                          group_labels: np.ndarray) -> float:
        """Calculate ANOSIM R-statistic."""
        unique_groups = np.unique(group_labels)
        
        if len(unique_groups) < 2:
            return 0.0
        
# Copyright (c) 2025 Mohamed Z. Hatim
        n = len(dist_matrix)
        ranks = np.zeros_like(dist_matrix)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        triu_indices = np.triu_indices(n, k=1)
        distances = dist_matrix[triu_indices]
        distance_ranks = stats.rankdata(distances)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        rank_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                ranks[i, j] = ranks[j, i] = distance_ranks[rank_idx]
                rank_idx += 1
        
# Copyright (c) 2025 Mohamed Z. Hatim
        within_ranks = []
        between_ranks = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if group_labels[i] == group_labels[j]:
                    within_ranks.append(ranks[i, j])
                else:
                    between_ranks.append(ranks[i, j])
        
        if len(within_ranks) == 0 or len(between_ranks) == 0:
            return 0.0
        
        mean_within = np.mean(within_ranks)
        mean_between = np.mean(between_ranks)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        n_comparisons = len(distances)
        mean_all_ranks = (n_comparisons + 1) / 2
        
        r_stat = (mean_between - mean_within) / (2 * mean_all_ranks - mean_within - mean_between)
        
        return r_stat
    
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
    
    def mrpp(self, distance_matrix: Union[pd.DataFrame, np.ndarray],
             groups: Union[pd.Series, List],
             permutations: int = 999) -> Dict[str, Any]:
        """
        Multi-Response Permutation Procedures (MRPP).
        
        Parameters:
        -----------
        distance_matrix : pd.DataFrame or np.ndarray
            Distance matrix
        groups : pd.Series or list
            Group assignments
        permutations : int
            Number of permutations
            
        Returns:
        --------
        dict
            MRPP results including delta and A statistics
        """
# Copyright (c) 2025 Mohamed Z. Hatim
        if isinstance(distance_matrix, pd.DataFrame):
            dist_matrix = distance_matrix.values
        else:
            dist_matrix = distance_matrix
        
        if isinstance(groups, pd.Series):
            group_labels = groups.values
        else:
            group_labels = np.array(groups)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        observed_delta = self._calculate_mrpp_delta(dist_matrix, group_labels)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        n = len(group_labels)
        all_distances = []
        for i in range(n):
            for j in range(i + 1, n):
                all_distances.append(dist_matrix[i, j])
        
        expected_delta = np.mean(all_distances)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        permuted_deltas = []
        
        for _ in range(permutations):
            permuted_groups = np.random.permutation(group_labels)
            delta = self._calculate_mrpp_delta(dist_matrix, permuted_groups)
            permuted_deltas.append(delta)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        p_value = (np.sum(np.array(permuted_deltas) <= observed_delta) + 1) / (permutations + 1)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        a_statistic = (expected_delta - observed_delta) / expected_delta
        
        results = {
            'delta': observed_delta,
            'expected_delta': expected_delta,
            'a_statistic': a_statistic,
            'p_value': p_value,
            'permutations': permutations,
            'method': 'MRPP'
        }
        
        return results
    
    def _calculate_mrpp_delta(self, dist_matrix: np.ndarray,
                            group_labels: np.ndarray) -> float:
        """Calculate MRPP delta statistic."""
        unique_groups, group_counts = np.unique(group_labels, return_counts=True)
        n_total = len(group_labels)
        
        weighted_within_sum = 0
        
        for group, count in zip(unique_groups, group_counts):
            if count <= 1:
                continue
            
            group_mask = group_labels == group
            group_indices = np.where(group_mask)[0]
            
# Copyright (c) 2025 Mohamed Z. Hatim
            within_distances = []
            for i in range(len(group_indices)):
                for j in range(i + 1, len(group_indices)):
                    within_distances.append(dist_matrix[group_indices[i], group_indices[j]])
            
            if within_distances:
                mean_within_distance = np.mean(within_distances)
                weight = count / n_total
                weighted_within_sum += weight * mean_within_distance
        
        return weighted_within_sum
    
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
    
    def mantel_test(self, matrix1: Union[pd.DataFrame, np.ndarray],
                    matrix2: Union[pd.DataFrame, np.ndarray],
                    permutations: int = 999,
                    method: str = 'pearson') -> Dict[str, Any]:
        """
        Mantel test for matrix correlation.
        
        Parameters:
        -----------
        matrix1 : pd.DataFrame or np.ndarray
            First matrix
        matrix2 : pd.DataFrame or np.ndarray
            Second matrix
        permutations : int
            Number of permutations
        method : str
            Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
        --------
        dict
            Mantel test results
        """
# Copyright (c) 2025 Mohamed Z. Hatim
        if isinstance(matrix1, pd.DataFrame):
            mat1 = matrix1.values
        else:
            mat1 = matrix1
        
        if isinstance(matrix2, pd.DataFrame):
            mat2 = matrix2.values
        else:
            mat2 = matrix2
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if mat1.shape != mat2.shape:
            raise ValueError("Matrices must have the same dimensions")
        
# Copyright (c) 2025 Mohamed Z. Hatim
        n = mat1.shape[0]
        triu_indices = np.triu_indices(n, k=1)
        
        vec1 = mat1[triu_indices]
        vec2 = mat2[triu_indices]
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if method == 'pearson':
            observed_r = np.corrcoef(vec1, vec2)[0, 1]
        elif method == 'spearman':
            observed_r = stats.spearmanr(vec1, vec2)[0]
        elif method == 'kendall':
            observed_r = stats.kendalltau(vec1, vec2)[0]
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if np.isnan(observed_r):
            observed_r = 0.0
        
# Copyright (c) 2025 Mohamed Z. Hatim
        permuted_correlations = []
        
        for _ in range(permutations):
# Copyright (c) 2025 Mohamed Z. Hatim
            perm_indices = np.random.permutation(n)
            mat2_permuted = mat2[np.ix_(perm_indices, perm_indices)]
            
# Copyright (c) 2025 Mohamed Z. Hatim
            vec2_permuted = mat2_permuted[triu_indices]
            
# Copyright (c) 2025 Mohamed Z. Hatim
            if method == 'pearson':
                r = np.corrcoef(vec1, vec2_permuted)[0, 1]
            elif method == 'spearman':
                r = stats.spearmanr(vec1, vec2_permuted)[0]
            elif method == 'kendall':
                r = stats.kendalltau(vec1, vec2_permuted)[0]
            
            if np.isnan(r):
                r = 0.0
            
            permuted_correlations.append(r)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        permuted_correlations = np.array(permuted_correlations)
        p_value = (np.sum(np.abs(permuted_correlations) >= np.abs(observed_r)) + 1) / (permutations + 1)
        
        results = {
            'correlation': observed_r,
            'p_value': p_value,
            'permutations': permutations,
            'method': f'Mantel_{method}',
            'permuted_correlations': permuted_correlations
        }
        
        return results
    
    def partial_mantel_test(self, matrix1: Union[pd.DataFrame, np.ndarray],
                           matrix2: Union[pd.DataFrame, np.ndarray],
                           matrix3: Union[pd.DataFrame, np.ndarray],
                           permutations: int = 999,
                           method: str = 'pearson') -> Dict[str, Any]:
        """
        Partial Mantel test controlling for a third matrix.
        
        Parameters:
        -----------
        matrix1 : pd.DataFrame or np.ndarray
            First matrix
        matrix2 : pd.DataFrame or np.ndarray
            Second matrix
        matrix3 : pd.DataFrame or np.ndarray
            Control matrix
        permutations : int
            Number of permutations
        method : str
            Correlation method
            
        Returns:
        --------
        dict
            Partial Mantel test results
        """
# Copyright (c) 2025 Mohamed Z. Hatim
        def extract_upper_tri(matrix):
            if isinstance(matrix, pd.DataFrame):
                mat = matrix.values
            else:
                mat = matrix
            
            n = mat.shape[0]
            triu_indices = np.triu_indices(n, k=1)
            return mat[triu_indices]
        
        vec1 = extract_upper_tri(matrix1)
        vec2 = extract_upper_tri(matrix2)
        vec3 = extract_upper_tri(matrix3)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        def partial_correlation(x, y, z):
            """Calculate partial correlation between x and y controlling for z."""
# Copyright (c) 2025 Mohamed Z. Hatim
            rxy = np.corrcoef(x, y)[0, 1]
            rxz = np.corrcoef(x, z)[0, 1]
            ryz = np.corrcoef(y, z)[0, 1]
            
# Copyright (c) 2025 Mohamed Z. Hatim
            rxy = 0.0 if np.isnan(rxy) else rxy
            rxz = 0.0 if np.isnan(rxz) else rxz
            ryz = 0.0 if np.isnan(ryz) else ryz
            
# Copyright (c) 2025 Mohamed Z. Hatim
            denominator = np.sqrt((1 - rxz**2) * (1 - ryz**2))
            
            if denominator == 0:
                return 0.0
            
            partial_r = (rxy - rxz * ryz) / denominator
            
            return partial_r
        
# Copyright (c) 2025 Mohamed Z. Hatim
        observed_partial_r = partial_correlation(vec1, vec2, vec3)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        permuted_partial_correlations = []
        n = matrix1.shape[0] if isinstance(matrix1, np.ndarray) else matrix1.shape[0]
        
        for _ in range(permutations):
# Copyright (c) 2025 Mohamed Z. Hatim
            perm_indices = np.random.permutation(n)
            
            if isinstance(matrix2, pd.DataFrame):
                mat2_permuted = matrix2.iloc[perm_indices, perm_indices].values
            else:
                mat2_permuted = matrix2[np.ix_(perm_indices, perm_indices)]
            
            vec2_permuted = extract_upper_tri(mat2_permuted)
            
            partial_r = partial_correlation(vec1, vec2_permuted, vec3)
            permuted_partial_correlations.append(partial_r)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        permuted_partial_correlations = np.array(permuted_partial_correlations)
        p_value = (np.sum(np.abs(permuted_partial_correlations) >= np.abs(observed_partial_r)) + 1) / (permutations + 1)
        
        results = {
            'partial_correlation': observed_partial_r,
            'p_value': p_value,
            'permutations': permutations,
            'method': f'Partial_Mantel_{method}',
            'permuted_correlations': permuted_partial_correlations
        }
        
        return results
    
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
    
    def indicator_species_analysis(self, species_data: pd.DataFrame,
                                 groups: Union[pd.Series, List],
                                 permutations: int = 999) -> Dict[str, Any]:
        """
        Indicator Species Analysis (IndVal).
        
        Parameters:
        -----------
        species_data : pd.DataFrame
            Species abundance matrix
        groups : pd.Series or list
            Group assignments
        permutations : int
            Number of permutations
            
        Returns:
        --------
        dict
            IndVal results for each species
        """
        if isinstance(groups, pd.Series):
            group_labels = groups.values
        else:
            group_labels = np.array(groups)
        
        unique_groups = np.unique(group_labels)
        results = {}
        
        for species in species_data.columns:
            species_abundances = species_data[species].values
            
# Copyright (c) 2025 Mohamed Z. Hatim
            group_indvals = {}
            
            for group in unique_groups:
                group_mask = group_labels == group
                
# Copyright (c) 2025 Mohamed Z. Hatim
                group_abundance = species_abundances[group_mask]
                total_abundance = species_abundances.sum()
                
                if total_abundance == 0:
                    relative_abundance = 0
                else:
                    relative_abundance = group_abundance.sum() / total_abundance
                
# Copyright (c) 2025 Mohamed Z. Hatim
                group_presence = (group_abundance > 0).sum()
                group_size = group_mask.sum()
                
                if group_size == 0:
                    relative_frequency = 0
                else:
                    relative_frequency = group_presence / group_size
                
# Copyright (c) 2025 Mohamed Z. Hatim
                indval = relative_abundance * relative_frequency * 100
                
                group_indvals[group] = {
                    'indval': indval,
                    'relative_abundance': relative_abundance,
                    'relative_frequency': relative_frequency
                }
            
# Copyright (c) 2025 Mohamed Z. Hatim
            max_group = max(group_indvals.keys(), key=lambda g: group_indvals[g]['indval'])
            max_indval = group_indvals[max_group]['indval']
            
# Copyright (c) 2025 Mohamed Z. Hatim
            permuted_indvals = []
            
            for _ in range(permutations):
                permuted_groups = np.random.permutation(group_labels)
                
# Copyright (c) 2025 Mohamed Z. Hatim
                perm_group_mask = permuted_groups == max_group
                perm_group_abundance = species_abundances[perm_group_mask]
                
                if total_abundance == 0:
                    perm_rel_abundance = 0
                else:
                    perm_rel_abundance = perm_group_abundance.sum() / total_abundance
                
                perm_group_presence = (perm_group_abundance > 0).sum()
                perm_group_size = perm_group_mask.sum()
                
                if perm_group_size == 0:
                    perm_rel_frequency = 0
                else:
                    perm_rel_frequency = perm_group_presence / perm_group_size
                
                perm_indval = perm_rel_abundance * perm_rel_frequency * 100
                permuted_indvals.append(perm_indval)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            p_value = (np.sum(np.array(permuted_indvals) >= max_indval) + 1) / (permutations + 1)
            
            results[species] = {
                'max_group': max_group,
                'indval': max_indval,
                'p_value': p_value,
                'group_details': group_indvals
            }
        
        return results
    
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
    
    def simper_analysis(self, species_data: pd.DataFrame,
                       groups: Union[pd.Series, List],
                       distance_metric: str = 'bray_curtis') -> Dict[str, Any]:
        """
        Similarity Percentages (SIMPER) analysis.
        
        Parameters:
        -----------
        species_data : pd.DataFrame
            Species abundance matrix
        groups : pd.Series or list
            Group assignments
        distance_metric : str
            Distance metric to use
            
        Returns:
        --------
        dict
            SIMPER results
        """
        if isinstance(groups, pd.Series):
            group_labels = groups.values
        else:
            group_labels = np.array(groups)
        
        unique_groups = np.unique(group_labels)
        results = {}
        
# Copyright (c) 2025 Mohamed Z. Hatim
        for group in unique_groups:
            group_mask = group_labels == group
            group_data = species_data[group_mask]
            
            if len(group_data) < 2:
                continue
            
# Copyright (c) 2025 Mohamed Z. Hatim
            similarities = []
            species_contributions = {species: [] for species in species_data.columns}
            
            for i in range(len(group_data)):
                for j in range(i + 1, len(group_data)):
# Copyright (c) 2025 Mohamed Z. Hatim
                    sample1 = group_data.iloc[i].values
                    sample2 = group_data.iloc[j].values
                    
                    if distance_metric == 'bray_curtis':
                        distance = self._bray_curtis_single(sample1, sample2)
                    else:
                        distance = np.linalg.norm(sample1 - sample2)
                    
                    similarity = 1 - distance
                    similarities.append(similarity)
                    
# Copyright (c) 2025 Mohamed Z. Hatim
                    for k, species in enumerate(species_data.columns):
# Copyright (c) 2025 Mohamed Z. Hatim
                        avg_abundance = (sample1[k] + sample2[k]) / 2
                        species_contribution = avg_abundance / (sample1.sum() + sample2.sum()) * similarity
                        species_contributions[species].append(species_contribution)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            avg_similarity = np.mean(similarities) if similarities else 0
            
            species_avg_contrib = {}
            for species in species_data.columns:
                if species_contributions[species]:
                    species_avg_contrib[species] = np.mean(species_contributions[species])
                else:
                    species_avg_contrib[species] = 0
            
            results[f'within_{group}'] = {
                'average_similarity': avg_similarity,
                'species_contributions': species_avg_contrib
            }
        
# Copyright (c) 2025 Mohamed Z. Hatim
        for i, group1 in enumerate(unique_groups):
            for group2 in unique_groups[i + 1:]:
                group1_mask = group_labels == group1
                group2_mask = group_labels == group2
                
                group1_data = species_data[group1_mask]
                group2_data = species_data[group2_mask]
                
# Copyright (c) 2025 Mohamed Z. Hatim
                dissimilarities = []
                species_contributions = {species: [] for species in species_data.columns}
                
                for idx1, (_, sample1) in enumerate(group1_data.iterrows()):
                    for idx2, (_, sample2) in enumerate(group2_data.iterrows()):
                        s1_values = sample1.values
                        s2_values = sample2.values
                        
                        if distance_metric == 'bray_curtis':
                            dissimilarity = self._bray_curtis_single(s1_values, s2_values)
                        else:
                            dissimilarity = np.linalg.norm(s1_values - s2_values)
                        
                        dissimilarities.append(dissimilarity)
                        
# Copyright (c) 2025 Mohamed Z. Hatim
                        for k, species in enumerate(species_data.columns):
                            contrib = abs(s1_values[k] - s2_values[k]) / (s1_values.sum() + s2_values.sum()) * dissimilarity
                            species_contributions[species].append(contrib)
                
# Copyright (c) 2025 Mohamed Z. Hatim
                avg_dissimilarity = np.mean(dissimilarities) if dissimilarities else 0
                
                species_avg_contrib = {}
                for species in species_data.columns:
                    if species_contributions[species]:
                        species_avg_contrib[species] = np.mean(species_contributions[species])
                    else:
                        species_avg_contrib[species] = 0
                
                results[f'between_{group1}_{group2}'] = {
                    'average_dissimilarity': avg_dissimilarity,
                    'species_contributions': species_avg_contrib
                }
        
        return results
    
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
    
    def _bray_curtis_single(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate Bray-Curtis distance between two samples."""
        numerator = np.sum(np.abs(x - y))
        denominator = np.sum(x + y)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _bray_curtis_distance(self, data: np.ndarray) -> np.ndarray:
        """Calculate Bray-Curtis distance matrix."""
        n_samples = data.shape[0]
        distances = []
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distance = self._bray_curtis_single(data[i], data[j])
                distances.append(distance)
        
        return np.array(distances)
    
    def _jaccard_distance(self, data: np.ndarray) -> np.ndarray:
        """Calculate Jaccard distance matrix."""
        binary_data = (data > 0).astype(int)
        return pdist(binary_data, metric='jaccard')
    
    def _euclidean_distance(self, data: np.ndarray) -> np.ndarray:
        """Calculate Euclidean distance matrix."""
        return pdist(data, metric='euclidean')
    
    def _manhattan_distance(self, data: np.ndarray) -> np.ndarray:
        """Calculate Manhattan distance matrix."""
        return pdist(data, metric='manhattan')