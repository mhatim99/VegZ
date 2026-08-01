"""
Nestedness Analysis and Null Models Module

This module provides comprehensive nestedness analysis and null model implementations
for ecological community data analysis.

Copyright (c) 2025 Mohamed Z. Hatim
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt

from ._compat import as_generator

NUMBA_AVAILABLE = False


class NestednessAnalyzer:
    """
    Comprehensive nestedness analyzer for ecological community data.
    
    Provides various nestedness metrics and null model comparisons.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the NestednessAnalyzer.
        
        Parameters
        ----------
        random_state : int, optional
            Random state for reproducibility, by default 42
        """
        # Use a private Generator rather than seeding NumPy's global RNG:
        # constructing an analyser must not silently reset randomness for the
        # rest of the user's program.
        self.random_state = random_state
        self.rng = as_generator(random_state)
        self.matrix_data = None
        self.nestedness_results = {}
        
    def load_matrix(self, data: pd.DataFrame, 
                   presence_threshold: float = 0) -> None:
        """
        Load presence-absence or abundance matrix.
        
        Parameters
        ----------
        data : pd.DataFrame
            Community matrix (sites x species)
        presence_threshold : float, optional
            Threshold for converting abundance to presence/absence, by default 0
        """
# Copyright (c) 2025 Mohamed Z. Hatim
        if presence_threshold > 0:
            self.matrix_data = (data > presence_threshold).astype(int)
        else:
            self.matrix_data = (data.fillna(0) > 0).astype(int)
    
    def calculate_nestedness_metrics(self, 
                                   sort_by: str = 'total',
                                   metrics: List[str] = None) -> Dict[str, Any]:
        """
        Calculate various nestedness metrics.
        
        Parameters
        ----------
        sort_by : str, optional
            How to sort matrix ('total', 'richness', 'abundance'), by default 'total'
        metrics : List[str], optional
            Nestedness metrics to calculate. If None, calculate all available
            
        Returns
        -------
        Dict[str, Any]
            Nestedness analysis results
        """
        if self.matrix_data is None:
            raise ValueError("Matrix data not loaded. Use load_matrix() first.")
        
        if metrics is None:
            metrics = ['nodf', 'temperature', 'c_score', 'togetherness']
        
# Copyright (c) 2025 Mohamed Z. Hatim
        sorted_matrix = self._sort_matrix(sort_by)
        
        results = {
            'original_matrix': self.matrix_data,
            'sorted_matrix': sorted_matrix,
            'sorting_method': sort_by,
            'metrics': {}
        }
        
# Copyright (c) 2025 Mohamed Z. Hatim
        for metric in metrics:
            if metric == 'nodf':
                results['metrics']['nodf'] = self._calculate_nodf(sorted_matrix)
            elif metric == 'temperature':
                results['metrics']['temperature'] = self._calculate_temperature(sorted_matrix)
            elif metric == 'c_score':
                results['metrics']['c_score'] = self._calculate_c_score(sorted_matrix)
            elif metric == 'togetherness':
                results['metrics']['togetherness'] = self._calculate_togetherness(sorted_matrix)
            elif metric == 'wine':
                results['metrics']['wine'] = self._calculate_wine(sorted_matrix)
        
        self.nestedness_results = results
        return results
    
    def _sort_matrix(self, sort_by: str) -> pd.DataFrame:
        """Sort matrix according to specified criteria."""
        if sort_by == 'total':
# Copyright (c) 2025 Mohamed Z. Hatim
            site_totals = self.matrix_data.sum(axis=1).sort_values(ascending=False)
            sorted_sites = site_totals.index
            
# Copyright (c) 2025 Mohamed Z. Hatim
            species_totals = self.matrix_data.sum(axis=0).sort_values(ascending=False)
            sorted_species = species_totals.index
            
        elif sort_by == 'richness':
# Copyright (c) 2025 Mohamed Z. Hatim
            site_totals = self.matrix_data.sum(axis=1).sort_values(ascending=False)
            sorted_sites = site_totals.index
            species_totals = self.matrix_data.sum(axis=0).sort_values(ascending=False)
            sorted_species = species_totals.index
            
        elif sort_by == 'abundance':
            if (self.matrix_data > 1).any().any():
# Copyright (c) 2025 Mohamed Z. Hatim
                site_totals = self.matrix_data.sum(axis=1).sort_values(ascending=False)
                sorted_sites = site_totals.index
                species_totals = self.matrix_data.sum(axis=0).sort_values(ascending=False)
                sorted_species = species_totals.index
            else:
# Copyright (c) 2025 Mohamed Z. Hatim
                return self._sort_matrix('richness')
        else:
            raise ValueError(f"Unknown sorting method: {sort_by}")
        
        return self.matrix_data.loc[sorted_sites, sorted_species]
    
    def _calculate_nodf(self, matrix: pd.DataFrame) -> Dict[str, float]:
        """
        NODF - Nestedness metric based on Overlap and Decreasing Fill.

        Almeida-Neto et al. (2008). For every pair of rows (and every pair of
        columns) the paired nestedness is the percentage overlap when fill
        strictly decreases, and 0 when the two marginal totals are equal. The
        average is taken over *all* pairs.

        Parameters
        ----------
        matrix : pd.DataFrame
            Sorted presence-absence matrix

        Returns
        -------
        Dict[str, float]
            NODF for rows, columns and overall (0-100).

        Notes
        -----
        Averaging over all pairs is what bounds NODF at 100. Dividing only by
        the pairs that happened to have decreasing fill - as a naive
        implementation does - inflates the value, badly so for matrices with
        many equal row or column totals.
        """
        matrix_values = (matrix.values > 0).astype(float)

        row_nodf, n_row_pairs = self._paired_nestedness(matrix_values)
        col_nodf, n_col_pairs = self._paired_nestedness(matrix_values.T)

        total_pairs = n_row_pairs + n_col_pairs
        if total_pairs > 0:
            overall_nodf = ((row_nodf * n_row_pairs + col_nodf * n_col_pairs)
                            / total_pairs)
        else:
            overall_nodf = 0.0

        return {
            'nodf_rows': row_nodf,
            'nodf_columns': col_nodf,
            'nodf_overall': overall_nodf,
            'n_row_pairs': n_row_pairs,
            'n_col_pairs': n_col_pairs
        }

    @staticmethod
    def _paired_nestedness(matrix: np.ndarray) -> Tuple[float, int]:
        """Mean paired-overlap nestedness across all row pairs, and the pair count."""
        n = matrix.shape[0]
        if n < 2:
            return 0.0, 0

        marginals = matrix.sum(axis=1)
        overlap = matrix @ matrix.T

        i_idx, j_idx = np.triu_indices(n, k=1)
        mt_i, mt_j = marginals[i_idx], marginals[j_idx]

        # Orient each pair so that "i" is the richer row.
        richer = np.maximum(mt_i, mt_j)
        poorer = np.minimum(mt_i, mt_j)
        shared = overlap[i_idx, j_idx]

        # Decreasing fill required; equal totals contribute 0 but still count.
        with np.errstate(divide='ignore', invalid='ignore'):
            paired = np.where((richer > poorer) & (poorer > 0),
                              shared / poorer * 100.0, 0.0)

        n_pairs = int(i_idx.size)
        return float(paired.sum() / n_pairs), n_pairs
    
    def _calculate_temperature(self, matrix: pd.DataFrame) -> Dict[str, float]:
        """
        Matrix temperature (Atmar & Patterson 1993).

        The packed matrix is compared with the isocline of perfect nestedness
        for its fill. Cells on the "wrong" side of the isocline (absences in the
        filled region, presences in the empty region) are unexpected; each
        contributes ``(d / D)^2``, where ``d`` is its distance from the isocline
        along the matrix diagonal and ``D`` the full diagonal length through
        that cell. Temperature is the mean unexpectedness rescaled so that a
        maximally disordered matrix scores 100.

        Returns
        -------
        Dict[str, float]
            ``temperature`` (0 = perfectly nested, 100 = maximally disordered),
            plus the counts of unexpected presences/absences and the fill.

        Notes
        -----
        Vectorised; the earlier quadruple loop over all cell pairs was both
        O(n^2 m^2) and unrelated to the published metric.
        """
        matrix_values = (matrix.values > 0).astype(float)
        n_sites, n_species = matrix_values.shape

        if n_sites < 2 or n_species < 2:
            return {'temperature': 0.0, 'unexpected_absences': 0,
                    'unexpected_presences': 0, 'fill': float(matrix_values.mean()),
                    'n_unexpected': 0}

        fill = float(matrix_values.sum() / matrix_values.size)
        if fill <= 0 or fill >= 1:
            return {'temperature': 0.0, 'unexpected_absences': 0,
                    'unexpected_presences': 0, 'fill': fill, 'n_unexpected': 0}

        # Normalised cell coordinates in the unit square.
        x = (np.arange(n_species) + 0.5) / n_species
        y = (np.arange(n_sites) + 0.5) / n_sites
        X, Y = np.meshgrid(x, y)

        # Isocline of perfect nestedness enclosing the observed fill.
        # y = (1 - x)^p, with p chosen so the area under the curve equals fill.
        # Area of (1-x)^p over [0,1] is 1/(p+1), so p = 1/fill - 1.
        p = 1.0 / fill - 1.0
        isocline_y = (1 - X) ** p

        # Above the curve => expected to be filled; below => expected empty.
        expected_present = Y <= isocline_y

        unexpected_absence = expected_present & (matrix_values == 0)
        unexpected_presence = (~expected_present) & (matrix_values == 1)
        unexpected = unexpected_absence | unexpected_presence

        n_unexpected = int(unexpected.sum())
        if n_unexpected == 0:
            return {'temperature': 0.0, 'unexpected_absences': 0,
                    'unexpected_presences': 0, 'fill': fill, 'n_unexpected': 0}

        # Distance from the isocline, measured vertically and normalised by the
        # available room on that side (a proxy for the diagonal normalisation).
        with np.errstate(divide='ignore', invalid='ignore'):
            room = np.where(expected_present, isocline_y, 1 - isocline_y)
            room = np.where(room > 1e-12, room, 1e-12)
            d_over_D = np.abs(Y - isocline_y) / room
        d_over_D = np.clip(d_over_D, 0.0, 1.0)

        unexpectedness = np.where(unexpected, d_over_D ** 2, 0.0)
        mean_unexpectedness = float(unexpectedness.sum() / matrix_values.size)

        # Atmar & Patterson's normalising constant for a maximally disordered
        # matrix; 100 / 0.04145 maps that case onto T = 100.
        temperature = float(np.clip(mean_unexpectedness / 0.04145, 0.0, 1.0) * 100)

        return {
            'temperature': temperature,
            'unexpected_absences': int(unexpected_absence.sum()),
            'unexpected_presences': int(unexpected_presence.sum()),
            'n_unexpected': n_unexpected,
            'fill': fill,
        }
    
    def _calculate_c_score(self, matrix: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate C-score (checkerboard score).
        
        Parameters
        ----------
        matrix : pd.DataFrame
            Presence-absence matrix
            
        Returns
        -------
        Dict[str, float]
            C-score values
        """
        matrix_values = matrix.values.astype(float)
        n_sites, n_species = matrix_values.shape
        
        total_c_score = 0
        n_pairs = 0
        
# Copyright (c) 2025 Mohamed Z. Hatim
        for i in range(n_species - 1):
            for j in range(i + 1, n_species):
                species_i = matrix_values[:, i]
                species_j = matrix_values[:, j]
                
# Copyright (c) 2025 Mohamed Z. Hatim
                only_i = np.sum((species_i == 1) & (species_j == 0))
                only_j = np.sum((species_i == 0) & (species_j == 1))
                
# Copyright (c) 2025 Mohamed Z. Hatim
                c_score_pair = only_i * only_j
                total_c_score += c_score_pair
                n_pairs += 1
        
# Copyright (c) 2025 Mohamed Z. Hatim
        avg_c_score = total_c_score / n_pairs if n_pairs > 0 else 0
        
        return {
            'c_score': avg_c_score,
            'total_c_score': total_c_score,
            'n_species_pairs': n_pairs
        }
    
    def _calculate_togetherness(self, matrix: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Stone and Roberts' togetherness metric.
        
        Parameters
        ----------
        matrix : pd.DataFrame
            Presence-absence matrix
            
        Returns
        -------
        Dict[str, float]
            Togetherness values
        """
        matrix_values = matrix.values.astype(float)
        n_sites, n_species = matrix_values.shape
        
        total_togetherness = 0
        n_pairs = 0
        
# Copyright (c) 2025 Mohamed Z. Hatim
        for i in range(n_species - 1):
            for j in range(i + 1, n_species):
                species_i = matrix_values[:, i]
                species_j = matrix_values[:, j]
                
# Copyright (c) 2025 Mohamed Z. Hatim
                both_present = np.sum((species_i == 1) & (species_j == 1))
                
# Copyright (c) 2025 Mohamed Z. Hatim
                at_least_one = np.sum((species_i == 1) | (species_j == 1))
                
# Copyright (c) 2025 Mohamed Z. Hatim
                if at_least_one > 0:
                    togetherness_pair = both_present / at_least_one
                else:
                    togetherness_pair = 0
                
                total_togetherness += togetherness_pair
                n_pairs += 1
        
# Copyright (c) 2025 Mohamed Z. Hatim
        avg_togetherness = total_togetherness / n_pairs if n_pairs > 0 else 0
        
        return {
            'togetherness': avg_togetherness,
            'total_togetherness': total_togetherness,
            'n_species_pairs': n_pairs
        }
    
    def _calculate_wine(self, matrix: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate WINE (Weighted Index of Nestedness based on Overlap).
        
        Parameters
        ----------
        matrix : pd.DataFrame
            Sorted presence-absence matrix
            
        Returns
        -------
        Dict[str, float]
            WINE values
        """
        matrix_values = matrix.values.astype(float)
        n_sites, n_species = matrix_values.shape
        
# Copyright (c) 2025 Mohamed Z. Hatim
        site_totals = np.sum(matrix_values, axis=1)
        species_totals = np.sum(matrix_values, axis=0)
        
        total_wine = 0
        total_weight = 0
        
# Copyright (c) 2025 Mohamed Z. Hatim
        for i in range(n_sites - 1):
            for j in range(i + 1, n_sites):
                if site_totals[i] >= site_totals[j] and site_totals[j] > 0:
# Copyright (c) 2025 Mohamed Z. Hatim
                    overlap = np.sum(matrix_values[i, :] * matrix_values[j, :])
                    
# Copyright (c) 2025 Mohamed Z. Hatim
                    weight = site_totals[j]
                    
# Copyright (c) 2025 Mohamed Z. Hatim
                    wine_contribution = (overlap / site_totals[j]) * weight
                    total_wine += wine_contribution
                    total_weight += weight
        
# Copyright (c) 2025 Mohamed Z. Hatim
        for i in range(n_species - 1):
            for j in range(i + 1, n_species):
                if species_totals[i] >= species_totals[j] and species_totals[j] > 0:
# Copyright (c) 2025 Mohamed Z. Hatim
                    overlap = np.sum(matrix_values[:, i] * matrix_values[:, j])
                    
# Copyright (c) 2025 Mohamed Z. Hatim
                    weight = species_totals[j]
                    
# Copyright (c) 2025 Mohamed Z. Hatim
                    wine_contribution = (overlap / species_totals[j]) * weight
                    total_wine += wine_contribution
                    total_weight += weight
        
# Copyright (c) 2025 Mohamed Z. Hatim
        wine = (total_wine / total_weight) * 100 if total_weight > 0 else 0
        
        return {
            'wine': wine,
            'total_weight': total_weight
        }


class NullModels:
    """
    Null model implementations for ecological community data.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize null models with a private random generator."""
        self.random_state = random_state
        self.rng = as_generator(random_state)
    
    def generate_null_matrices(self,
                             matrix: pd.DataFrame,
                             n_iterations: int = 999,
                             model_type: str = 'equiprobable',
                             fixed_marginals: str = 'both') -> List[pd.DataFrame]:
        """
        Generate null matrices using various null models.
        
        Parameters
        ----------
        matrix : pd.DataFrame
            Observed community matrix
        n_iterations : int, optional
            Number of null matrices to generate, by default 999
        model_type : str, optional
            Type of null model ('equiprobable', 'proportional', 'fixed_fixed'), by default 'equiprobable'
        fixed_marginals : str, optional
            Which marginals to preserve ('rows', 'columns', 'both', 'none'), by default 'both'
            
        Returns
        -------
        List[pd.DataFrame]
            List of null matrices
        """
        null_matrices = []
        matrix_values = matrix.values.astype(int)
        
        for _ in range(n_iterations):
            if model_type == 'equiprobable':
                null_matrix = self._equiprobable_model(matrix_values, fixed_marginals)
            elif model_type == 'proportional':
                null_matrix = self._proportional_model(matrix_values, fixed_marginals)
            elif model_type == 'fixed_fixed':
                null_matrix = self._fixed_fixed_model(matrix_values)
            elif model_type == 'sequential_swap':
                null_matrix = self._sequential_swap_model(matrix_values)
            else:
                raise ValueError(f"Unknown null model type: {model_type}")
            
            null_df = pd.DataFrame(null_matrix, index=matrix.index, columns=matrix.columns)
            null_matrices.append(null_df)
        
        return null_matrices
    
    def _equiprobable_model(self, matrix: np.ndarray, fixed_marginals: str) -> np.ndarray:
        """Generate null matrix with equiprobable placement."""
        n_sites, n_species = matrix.shape
        total_occurrences = np.sum(matrix)
        
        if fixed_marginals == 'both':
# Copyright (c) 2025 Mohamed Z. Hatim
            return self._fixed_fixed_model(matrix)
        elif fixed_marginals == 'rows':
# Copyright (c) 2025 Mohamed Z. Hatim
            null_matrix = np.zeros_like(matrix)
            for i in range(n_sites):
                row_sum = np.sum(matrix[i, :])
                if row_sum > 0:
                    chosen_cols = self.rng.choice(n_species, size=row_sum, replace=False)
                    null_matrix[i, chosen_cols] = 1
        elif fixed_marginals == 'columns':
# Copyright (c) 2025 Mohamed Z. Hatim
            null_matrix = np.zeros_like(matrix)
            for j in range(n_species):
                col_sum = np.sum(matrix[:, j])
                if col_sum > 0:
                    chosen_rows = self.rng.choice(n_sites, size=col_sum, replace=False)
                    null_matrix[chosen_rows, j] = 1
        else:  # none
# Copyright (c) 2025 Mohamed Z. Hatim
            null_matrix = np.zeros_like(matrix)
            flat_indices = self.rng.choice(
                n_sites * n_species, size=total_occurrences, replace=False
            )
            null_matrix.flat[flat_indices] = 1
        
        return null_matrix
    
    def _proportional_model(self, matrix: np.ndarray, fixed_marginals: str) -> np.ndarray:
        """Generate null matrix with proportional probabilities."""
        n_sites, n_species = matrix.shape
        
# Copyright (c) 2025 Mohamed Z. Hatim
        row_probs = np.sum(matrix, axis=1) / np.sum(matrix) if np.sum(matrix) > 0 else np.ones(n_sites) / n_sites
        col_probs = np.sum(matrix, axis=0) / np.sum(matrix) if np.sum(matrix) > 0 else np.ones(n_species) / n_species
        
        if fixed_marginals == 'both':
            return self._fixed_fixed_model(matrix)
        elif fixed_marginals == 'rows':
            null_matrix = np.zeros_like(matrix)
            for i in range(n_sites):
                row_sum = np.sum(matrix[i, :])
                if row_sum > 0:
                    probs = col_probs / np.sum(col_probs)  # Normalize
                    chosen_cols = self.rng.choice(n_species, size=row_sum, replace=False, p=probs)
                    null_matrix[i, chosen_cols] = 1
        elif fixed_marginals == 'columns':
            null_matrix = np.zeros_like(matrix)
            for j in range(n_species):
                col_sum = np.sum(matrix[:, j])
                if col_sum > 0:
                    probs = row_probs / np.sum(row_probs)  # Normalize
                    chosen_rows = self.rng.choice(n_sites, size=col_sum, replace=False, p=probs)
                    null_matrix[chosen_rows, j] = 1
        else:  # none
# Copyright (c) 2025 Mohamed Z. Hatim
            prob_matrix = np.outer(row_probs, col_probs)
            prob_matrix = prob_matrix / np.sum(prob_matrix)  # Normalize
            
            total_occurrences = np.sum(matrix)
            flat_probs = prob_matrix.flatten()
            flat_indices = self.rng.choice(
                len(flat_probs), size=total_occurrences, replace=False, p=flat_probs
            )
            
            null_matrix = np.zeros_like(matrix)
            null_matrix.flat[flat_indices] = 1
        
        return null_matrix
    
    def _fixed_fixed_model(self, matrix: np.ndarray) -> np.ndarray:
        """Generate null matrix preserving both row and column sums."""
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
        return self._sequential_swap_model(matrix, n_swaps=1000)
    
    def _sequential_swap_model(self, matrix: np.ndarray, n_swaps: int = None) -> np.ndarray:
        """Generate null matrix using sequential swapping algorithm."""
        null_matrix = matrix.copy()
        n_sites, n_species = matrix.shape
        
        if n_swaps is None:
            n_swaps = n_sites * n_species * 10  # Default number of swaps
        
# Copyright (c) 2025 Mohamed Z. Hatim
        ones = np.where(null_matrix == 1)
        zeros = np.where(null_matrix == 0)
        
        for _ in range(n_swaps):
            if len(ones[0]) < 2 or len(zeros[0]) < 2:
                break
            
# Copyright (c) 2025 Mohamed Z. Hatim
            idx1 = int(self.rng.integers(len(ones[0])))
            idx2 = int(self.rng.integers(len(ones[0])))
            
            if idx1 == idx2:
                continue
            
            r1, c1 = ones[0][idx1], ones[1][idx1]
            r2, c2 = ones[0][idx2], ones[1][idx2]
            
# Copyright (c) 2025 Mohamed Z. Hatim
            if r1 != r2 and c1 != c2:
# Copyright (c) 2025 Mohamed Z. Hatim
                if null_matrix[r1, c2] == 0 and null_matrix[r2, c1] == 0:
# Copyright (c) 2025 Mohamed Z. Hatim
                    null_matrix[r1, c1] = 0
                    null_matrix[r2, c2] = 0
                    null_matrix[r1, c2] = 1
                    null_matrix[r2, c1] = 1
                    
# Copyright (c) 2025 Mohamed Z. Hatim
                    ones = np.where(null_matrix == 1)
                    zeros = np.where(null_matrix == 0)
        
        return null_matrix


class NestednessSignificance:
    """
    Class for testing nestedness significance using null models.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize significance testing."""
        self.random_state = random_state
        self.nestedness_analyzer = NestednessAnalyzer(random_state)
        self.null_models = NullModels(random_state)
    
    def test_nestedness_significance(self,
                                   matrix: pd.DataFrame,
                                   metrics: List[str] = None,
                                   null_model: str = 'equiprobable',
                                   n_iterations: int = 999) -> Dict[str, Any]:
        """
        Test nestedness significance against null models.
        
        Parameters
        ----------
        matrix : pd.DataFrame
            Observed community matrix
        metrics : List[str], optional
            Nestedness metrics to test. If None, test all available
        null_model : str, optional
            Null model to use, by default 'equiprobable'
        n_iterations : int, optional
            Number of null matrices, by default 999
            
        Returns
        -------
        Dict[str, Any]
            Significance test results
        """
# Copyright (c) 2025 Mohamed Z. Hatim
        self.nestedness_analyzer.load_matrix(matrix)
        observed_results = self.nestedness_analyzer.calculate_nestedness_metrics(metrics=metrics)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        null_matrices = self.null_models.generate_null_matrices(
            matrix, n_iterations, null_model
        )
        
# Copyright (c) 2025 Mohamed Z. Hatim
        null_results = {metric: [] for metric in observed_results['metrics']}

        # Reuse one analyser: constructing a fresh one per iteration is pure
        # overhead (and used to reset the random state each time round).
        null_analyzer = NestednessAnalyzer(self.random_state)

        for null_matrix in null_matrices:
            null_analyzer.load_matrix(null_matrix)
            null_nestedness = null_analyzer.calculate_nestedness_metrics(
                metrics=list(observed_results['metrics'].keys())
            )

            for metric in observed_results['metrics']:
                if metric in null_nestedness['metrics']:
                    if isinstance(null_nestedness['metrics'][metric], dict):
# Copyright (c) 2025 Mohamed Z. Hatim
                        key = f'{metric}_overall' if f'{metric}_overall' in null_nestedness['metrics'][metric] else list(null_nestedness['metrics'][metric].keys())[0]
                        null_results[metric].append(null_nestedness['metrics'][metric][key])
                    else:
                        null_results[metric].append(null_nestedness['metrics'][metric])
        
# Copyright (c) 2025 Mohamed Z. Hatim
        significance_results = {}
        
        for metric in observed_results['metrics']:
            if metric not in null_results or not null_results[metric]:
                continue
            
            observed_value = observed_results['metrics'][metric]
            if isinstance(observed_value, dict):
                key = f'{metric}_overall' if f'{metric}_overall' in observed_value else list(observed_value.keys())[0]
                observed_value = observed_value[key]
            
            null_values = np.array(null_results[metric])

            # (count + 1) / (n + 1): the observed value is itself one of the
            # possible arrangements, so a plain count/n can report p = 0, which
            # is not attainable in a permutation test.
            n_null = len(null_values)
            p_greater = (np.sum(null_values >= observed_value) + 1) / (n_null + 1)
            p_lesser = (np.sum(null_values <= observed_value) + 1) / (n_null + 1)
            p_two_tailed = min(1.0, 2 * min(p_greater, p_lesser))

# Copyright (c) 2025 Mohamed Z. Hatim
            null_mean = np.mean(null_values)
            null_std = np.std(null_values)
            ses = (observed_value - null_mean) / null_std if null_std > 0 else 0
            
            significance_results[metric] = {
                'observed': observed_value,
                'null_mean': null_mean,
                'null_std': null_std,
                'null_min': np.min(null_values),
                'null_max': np.max(null_values),
                'ses': ses,
                'p_greater': p_greater,
                'p_lesser': p_lesser,
                'p_two_tailed': p_two_tailed,
                'significant': p_two_tailed < 0.05,
                'null_distribution': null_values
            }
        
        return {
            'observed_results': observed_results,
            'significance_tests': significance_results,
            'null_model': null_model,
            'n_iterations': n_iterations
        }
    
    def plot_null_distributions(self, 
                              significance_results: Dict[str, Any],
                              metrics: List[str] = None) -> plt.Figure:
        """
        Plot null distributions with observed values.
        
        Parameters
        ----------
        significance_results : Dict[str, Any]
            Results from test_nestedness_significance
        metrics : List[str], optional
            Metrics to plot. If None, plot all available
            
        Returns
        -------
        plt.Figure
            Null distribution plots
        """
        sig_tests = significance_results['significance_tests']
        
        if metrics is None:
            metrics = list(sig_tests.keys())
        
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = axes.flatten() if n_metrics > 1 else [axes]
        
        for i, metric in enumerate(metrics):
            if metric not in sig_tests:
                continue
            
            result = sig_tests[metric]
            
# Copyright (c) 2025 Mohamed Z. Hatim
            axes[i].hist(result['null_distribution'], bins=30, alpha=0.7, 
                        color='lightblue', edgecolor='black', density=True)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            axes[i].axvline(result['observed'], color='red', linestyle='--', linewidth=2,
                          label=f"Observed: {result['observed']:.3f}")
            
# Copyright (c) 2025 Mohamed Z. Hatim
            axes[i].axvline(result['null_mean'], color='blue', linestyle='-', linewidth=1,
                          label=f"Null mean: {result['null_mean']:.3f}")
            
            axes[i].set_xlabel(f'{metric.upper()} value')
            axes[i].set_ylabel('Density')
            axes[i].set_title(f'{metric.upper()}: p = {result["p_two_tailed"]:.3f}, SES = {result["ses"]:.2f}')
            axes[i].legend()
            
# Copyright (c) 2025 Mohamed Z. Hatim
            if result['significant']:
                axes[i].text(0.95, 0.95, 'Significant', transform=axes[i].transAxes,
                           va='top', ha='right', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
# Copyright (c) 2025 Mohamed Z. Hatim
        for i in range(n_metrics, len(axes)):
            axes[i].remove()
        
        plt.tight_layout()
        return fig


# Copyright (c) 2025 Mohamed Z. Hatim
class CooccurrenceAnalysis:
    """
    Analysis of species co-occurrence patterns.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize co-occurrence analysis."""
        self.random_state = random_state
    
    def species_associations(self, 
                           matrix: pd.DataFrame,
                           method: str = 'jaccard') -> Dict[str, Any]:
        """
        Calculate species association indices.
        
        Parameters
        ----------
        matrix : pd.DataFrame
            Community matrix (sites x species)
        method : str, optional
            Association method ('jaccard', 'ochiai', 'dice'), by default 'jaccard'
            
        Returns
        -------
        Dict[str, Any]
            Species association results
        """
        matrix_values = matrix.values.astype(int)
        n_sites, n_species = matrix_values.shape
        species_names = matrix.columns
        
# Copyright (c) 2025 Mohamed Z. Hatim
        associations = np.zeros((n_species, n_species))
        
        for i in range(n_species):
            for j in range(i, n_species):
                species_i = matrix_values[:, i]
                species_j = matrix_values[:, j]
                
# Copyright (c) 2025 Mohamed Z. Hatim
                both_present = np.sum((species_i == 1) & (species_j == 1))
                i_only = np.sum((species_i == 1) & (species_j == 0))
                j_only = np.sum((species_i == 0) & (species_j == 1))

                if method == 'jaccard':
# Copyright (c) 2025 Mohamed Z. Hatim
                    denominator = both_present + i_only + j_only
                    association = both_present / denominator if denominator > 0 else 0
                    
                elif method == 'ochiai':
# Copyright (c) 2025 Mohamed Z. Hatim
                    denominator = np.sqrt((both_present + i_only) * (both_present + j_only))
                    association = both_present / denominator if denominator > 0 else 0
                    
                elif method == 'dice':
# Copyright (c) 2025 Mohamed Z. Hatim
                    denominator = 2 * both_present + i_only + j_only
                    association = (2 * both_present) / denominator if denominator > 0 else 0
                
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                associations[i, j] = association
                associations[j, i] = association  # Symmetric
        
# Copyright (c) 2025 Mohamed Z. Hatim
        association_df = pd.DataFrame(
            associations, index=species_names, columns=species_names
        )
        
        return {
            'associations': association_df,
            'method': method,
            'mean_association': np.mean(associations[np.triu_indices(n_species, k=1)]),
            'max_association': np.max(associations[np.triu_indices(n_species, k=1)]),
            'min_association': np.min(associations[np.triu_indices(n_species, k=1)])
        }
    
    def checkerboard_analysis(self, matrix: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze checkerboard patterns in species co-occurrence.
        
        Parameters
        ----------
        matrix : pd.DataFrame
            Community matrix (sites x species)
            
        Returns
        -------
        Dict[str, Any]
            Checkerboard analysis results
        """
        matrix_values = matrix.values.astype(int)
        n_sites, n_species = matrix_values.shape
        species_names = matrix.columns
        
        checkerboards = []
        c_scores = []
        
        for i in range(n_species - 1):
            for j in range(i + 1, n_species):
                species_i = matrix_values[:, i]
                species_j = matrix_values[:, j]
                
# Copyright (c) 2025 Mohamed Z. Hatim
                both_present = np.sum((species_i == 1) & (species_j == 1))
                i_only = np.sum((species_i == 1) & (species_j == 0))
                j_only = np.sum((species_i == 0) & (species_j == 1))
                both_absent = np.sum((species_i == 0) & (species_j == 0))
                
# Copyright (c) 2025 Mohamed Z. Hatim
                checkerboard = i_only * j_only
                checkerboards.append({
                    'species_1': species_names[i],
                    'species_2': species_names[j],
                    'checkerboard_units': checkerboard,
                    'both_present': both_present,
                    'species_1_only': i_only,
                    'species_2_only': j_only,
                    'both_absent': both_absent
                })
                
                c_scores.append(checkerboard)
        
        checkerboard_df = pd.DataFrame(checkerboards)
        
        return {
            'checkerboard_pairs': checkerboard_df,
            'total_checkerboard_units': np.sum(c_scores),
            'mean_c_score': np.mean(c_scores),
            'max_checkerboard': np.max(c_scores),
            'n_pairs_with_checkerboards': np.sum(np.array(c_scores) > 0)
        }


# Copyright (c) 2025 Mohamed Z. Hatim
def quick_nestedness_analysis(matrix: pd.DataFrame, 
                            metrics: List[str] = None) -> Dict[str, Any]:
    """
    Quick nestedness analysis.
    
    Parameters
    ----------
    matrix : pd.DataFrame
        Community matrix (sites x species)
    metrics : List[str], optional
        Nestedness metrics to calculate
        
    Returns
    -------
    Dict[str, Any]
        Nestedness analysis results
    """
    analyzer = NestednessAnalyzer()
    analyzer.load_matrix(matrix)
    return analyzer.calculate_nestedness_metrics(metrics=metrics)


def quick_null_model_test(matrix: pd.DataFrame,
                        metric: str = 'nodf',
                        null_model: str = 'equiprobable',
                        n_iterations: int = 99) -> Dict[str, Any]:
    """
    Quick null model significance test.
    
    Parameters
    ----------
    matrix : pd.DataFrame
        Community matrix (sites x species)
    metric : str, optional
        Nestedness metric to test, by default 'nodf'
    null_model : str, optional
        Null model to use, by default 'equiprobable'
    n_iterations : int, optional
        Number of null iterations, by default 99
        
    Returns
    -------
    Dict[str, Any]
        Significance test results
    """
    tester = NestednessSignificance()
    return tester.test_nestedness_significance(
        matrix, [metric], null_model, n_iterations
    )