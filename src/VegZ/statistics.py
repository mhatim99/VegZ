"""
Statistical analysis module for multivariate ecological statistics.

Copyright (c) 2025 Mohamed Z. Hatim
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Tuple, Optional, Any
from scipy import stats
from scipy.spatial.distance import pdist, squareform

from ._compat import as_generator

#: Mapping of VegZ distance names onto SciPy metric names.
_METRIC_ALIASES = {
    'bray_curtis': 'braycurtis',
    'braycurtis': 'braycurtis',
    'jaccard': 'jaccard',
    'sorensen': 'dice',
    'dice': 'dice',
    'euclidean': 'euclidean',
    'manhattan': 'cityblock',
    'cityblock': 'cityblock',
    'canberra': 'canberra',
    'chebyshev': 'chebyshev',
    'correlation': 'correlation',
    'cosine': 'cosine',
}

#: Metrics that operate on presence/absence and need binary input.
_BINARY_METRICS = {'jaccard', 'dice'}


def _as_square_distance(matrix: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """Return a square distance matrix from a square matrix or raw data."""
    values = matrix.values if isinstance(matrix, pd.DataFrame) else np.asarray(matrix)
    if values.ndim != 2:
        raise ValueError("Expected a 2-D distance or data matrix")
    if values.shape[0] == values.shape[1] and np.allclose(np.diag(values), 0):
        return np.asarray(values, dtype=float)
    return squareform(pdist(np.asarray(values, dtype=float)))


def _as_labels(groups: Union[pd.Series, List, np.ndarray]) -> np.ndarray:
    if isinstance(groups, pd.Series):
        return groups.values
    return np.asarray(groups)


class EcologicalStatistics:
    """Comprehensive statistical analysis for ecological data."""

    def __init__(self):
        """Initialize statistical analyzer."""
        self.available_tests = [
            'permanova', 'adonis', 'permdisp', 'anosim', 'mrpp', 'mantel',
            'partial_mantel', 'indval', 'simper'
        ]

        self.distance_metrics = {
            'bray_curtis': self._bray_curtis_distance,
            'jaccard': self._jaccard_distance,
            'sorensen': self._sorensen_distance,
            'euclidean': self._euclidean_distance,
            'manhattan': self._manhattan_distance
        }

    def calculate_distance_matrix(self, data: pd.DataFrame,
                                  metric: str = 'bray_curtis') -> pd.DataFrame:
        """
        Calculate distance matrix using the specified metric.

        Parameters:
        -----------
        data : pd.DataFrame
            Data matrix (samples x features)
        metric : str
            Distance metric. Accepts VegZ names ('bray_curtis', 'sorensen',
            'manhattan', ...) as well as any metric supported by
            :func:`scipy.spatial.distance.pdist`.

        Returns:
        --------
        pd.DataFrame
            Square distance matrix indexed by sample.
        """
        distances = self._condensed_distances(data.values, metric)

        return pd.DataFrame(
            squareform(distances),
            index=data.index,
            columns=data.index
        )

    @staticmethod
    def _condensed_distances(values: np.ndarray, metric: str) -> np.ndarray:
        """Condensed distance vector, raising clearly on unknown metrics."""
        key = str(metric).lower()
        scipy_metric = _METRIC_ALIASES.get(key, key)
        values = np.asarray(values, dtype=float)

        if scipy_metric in _BINARY_METRICS:
            values = (values > 0).astype(bool)

        try:
            return pdist(values, metric=scipy_metric)
        except ValueError as exc:
            raise ValueError(
                f"Unknown distance metric '{metric}'. Supported names: "
                f"{', '.join(sorted(_METRIC_ALIASES))} or any SciPy pdist metric."
            ) from exc

    # ------------------------------------------------------------------
    # PERMANOVA
    # ------------------------------------------------------------------

    def permanova(self, distance_matrix: Union[pd.DataFrame, np.ndarray],
                  groups: Union[pd.Series, List],
                  permutations: int = 999,
                  random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        Permutational Multivariate Analysis of Variance (PERMANOVA).

        Implements Anderson (2001): the total sum of squares is
        ``SS_T = sum_{i<j} d_ij^2 / N`` and the within-group sum of squares is
        ``SS_W = sum_g sum_{i<j in g} d_ij^2 / n_g``; ``SS_A = SS_T - SS_W`` and
        ``F = (SS_A / (a - 1)) / (SS_W / (N - a))``.

        Parameters:
        -----------
        distance_matrix : pd.DataFrame or np.ndarray
            Square distance matrix, or a raw data matrix (Euclidean distances
            are then computed internally).
        groups : pd.Series or list
            Group assignments
        permutations : int
            Number of permutations
        random_state : int, optional
            Seed for reproducible permutations.

        Returns:
        --------
        dict
            PERMANOVA results including F-statistic, R-squared and p-value.
        """
        dist_matrix = _as_square_distance(distance_matrix)
        group_labels = _as_labels(groups)

        if len(group_labels) != dist_matrix.shape[0]:
            raise ValueError(
                f"groups has {len(group_labels)} entries but the distance "
                f"matrix has {dist_matrix.shape[0]} samples"
            )

        squared = dist_matrix ** 2
        n_samples = len(group_labels)
        unique_groups = np.unique(group_labels)
        n_groups = len(unique_groups)

        if n_groups < 2:
            raise ValueError("PERMANOVA requires at least two groups")

        ss_total = self._total_sum_squares(squared)
        ss_within = self._within_sum_squares(squared, group_labels)
        ss_between = ss_total - ss_within

        df_between = n_groups - 1
        df_within = n_samples - n_groups

        observed_f = self._permanova_f(squared, group_labels)

        rng = as_generator(random_state)
        permuted_f_stats = np.empty(permutations, dtype=float)
        for i in range(permutations):
            permuted_f_stats[i] = self._permanova_f(
                squared, rng.permutation(group_labels)
            )

        p_value = (np.sum(permuted_f_stats >= observed_f) + 1) / (permutations + 1)
        r_squared = ss_between / ss_total if ss_total > 0 else 0.0

        return {
            'f_statistic': observed_f,
            'p_value': p_value,
            'r_squared': r_squared,
            'df_between': df_between,
            'df_within': df_within,
            'df_total': n_samples - 1,
            'ss_between': ss_between,
            'ss_within': ss_within,
            'ss_total': ss_total,
            'ms_between': ss_between / df_between if df_between > 0 else np.nan,
            'ms_within': ss_within / df_within if df_within > 0 else np.nan,
            'permutations': permutations,
            'permuted_f_statistics': permuted_f_stats,
            'method': 'PERMANOVA'
        }

    @staticmethod
    def _total_sum_squares(squared: np.ndarray) -> float:
        """SS_T = sum_{i<j} d_ij^2 / N (Anderson 2001)."""
        n = squared.shape[0]
        if n == 0:
            return 0.0
        return float(np.sum(np.triu(squared, k=1)) / n)

    @staticmethod
    def _within_sum_squares(squared: np.ndarray, group_labels: np.ndarray) -> float:
        """SS_W = sum_g sum_{i<j in g} d_ij^2 / n_g."""
        ss_within = 0.0
        for group in np.unique(group_labels):
            mask = group_labels == group
            n_group = int(np.sum(mask))
            if n_group < 2:
                continue
            sub = squared[np.ix_(mask, mask)]
            ss_within += float(np.sum(np.triu(sub, k=1)) / n_group)
        return ss_within

    def _permanova_f(self, squared: np.ndarray, group_labels: np.ndarray) -> float:
        """Pseudo-F statistic for one grouping."""
        n_samples = len(group_labels)
        n_groups = len(np.unique(group_labels))

        if n_groups < 2:
            return 0.0

        ss_total = self._total_sum_squares(squared)
        ss_within = self._within_sum_squares(squared, group_labels)
        ss_between = ss_total - ss_within

        df_between = n_groups - 1
        df_within = n_samples - n_groups

        if df_within <= 0 or ss_within <= 0:
            return 0.0

        return float((ss_between / df_between) / (ss_within / df_within))

    # Retained for backwards compatibility with code that called the helpers.
    def _calculate_permanova_f(self, dist_matrix: np.ndarray,
                               group_labels: np.ndarray) -> float:
        return self._permanova_f(np.asarray(dist_matrix, dtype=float) ** 2, group_labels)

    def _calculate_total_sum_squares(self, dist_matrix: np.ndarray) -> float:
        return self._total_sum_squares(np.asarray(dist_matrix, dtype=float) ** 2)

    def _calculate_between_sum_squares(self, dist_matrix: np.ndarray,
                                       group_labels: np.ndarray) -> float:
        squared = np.asarray(dist_matrix, dtype=float) ** 2
        return self._total_sum_squares(squared) - self._within_sum_squares(squared, group_labels)

    # ------------------------------------------------------------------
    # ANOSIM
    # ------------------------------------------------------------------

    def anosim(self, distance_matrix: Union[pd.DataFrame, np.ndarray],
               groups: Union[pd.Series, List],
               permutations: int = 999,
               random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        Analysis of Similarities (ANOSIM).

        ``R = (mean_between_rank - mean_within_rank) / (M / 2)`` where
        ``M = n(n-1)/2`` is the number of pairs (Clarke 1993). R lies in
        [-1, 1]; values near 1 indicate strong group separation.

        Parameters:
        -----------
        distance_matrix : pd.DataFrame or np.ndarray
            Square distance matrix
        groups : pd.Series or list
            Group assignments
        permutations : int
            Number of permutations
        random_state : int, optional
            Seed for reproducible permutations.

        Returns:
        --------
        dict
            ANOSIM results including the R statistic and p-value.
        """
        dist_matrix = _as_square_distance(distance_matrix)
        group_labels = _as_labels(groups)

        if len(group_labels) != dist_matrix.shape[0]:
            raise ValueError("groups length does not match the distance matrix")

        n = dist_matrix.shape[0]
        triu = np.triu_indices(n, k=1)
        ranks = stats.rankdata(dist_matrix[triu])

        observed_r = self._anosim_r(ranks, group_labels, triu)

        rng = as_generator(random_state)
        permuted_r_stats = np.empty(permutations, dtype=float)
        for i in range(permutations):
            permuted_r_stats[i] = self._anosim_r(
                ranks, rng.permutation(group_labels), triu
            )

        p_value = (np.sum(permuted_r_stats >= observed_r) + 1) / (permutations + 1)

        return {
            'r_statistic': observed_r,
            'p_value': p_value,
            'permutations': permutations,
            'permuted_r_statistics': permuted_r_stats,
            'method': 'ANOSIM'
        }

    @staticmethod
    def _anosim_r(ranks: np.ndarray, group_labels: np.ndarray,
                  triu: Tuple[np.ndarray, np.ndarray]) -> float:
        """ANOSIM R for a set of pre-computed pairwise ranks."""
        same_group = group_labels[triu[0]] == group_labels[triu[1]]

        if not same_group.any() or same_group.all():
            return 0.0

        mean_within = float(ranks[same_group].mean())
        mean_between = float(ranks[~same_group].mean())

        n_pairs = ranks.size
        return (mean_between - mean_within) / (n_pairs / 2.0)

    def _calculate_anosim_r(self, dist_matrix: np.ndarray,
                            group_labels: np.ndarray) -> float:
        """Backwards-compatible helper taking a full distance matrix."""
        dist_matrix = np.asarray(dist_matrix, dtype=float)
        triu = np.triu_indices(dist_matrix.shape[0], k=1)
        ranks = stats.rankdata(dist_matrix[triu])
        return self._anosim_r(ranks, np.asarray(group_labels), triu)

    # ------------------------------------------------------------------
    # MRPP
    # ------------------------------------------------------------------

    def mrpp(self, distance_matrix: Union[pd.DataFrame, np.ndarray],
             groups: Union[pd.Series, List],
             permutations: int = 999,
             random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        Multi-Response Permutation Procedures (MRPP).

        Parameters:
        -----------
        distance_matrix : pd.DataFrame or np.ndarray
            Square distance matrix
        groups : pd.Series or list
            Group assignments
        permutations : int
            Number of permutations
        random_state : int, optional
            Seed for reproducible permutations.

        Returns:
        --------
        dict
            MRPP results including delta and the chance-corrected A statistic.
        """
        dist_matrix = _as_square_distance(distance_matrix)
        group_labels = _as_labels(groups)

        if len(group_labels) != dist_matrix.shape[0]:
            raise ValueError("groups length does not match the distance matrix")

        observed_delta = self._mrpp_delta(dist_matrix, group_labels)

        triu = np.triu_indices(dist_matrix.shape[0], k=1)
        expected_delta = float(np.mean(dist_matrix[triu]))

        rng = as_generator(random_state)
        permuted_deltas = np.empty(permutations, dtype=float)
        for i in range(permutations):
            permuted_deltas[i] = self._mrpp_delta(
                dist_matrix, rng.permutation(group_labels)
            )

        p_value = (np.sum(permuted_deltas <= observed_delta) + 1) / (permutations + 1)
        a_statistic = (1 - observed_delta / expected_delta) if expected_delta > 0 else 0.0

        return {
            'delta': observed_delta,
            'expected_delta': expected_delta,
            'a_statistic': a_statistic,
            'p_value': p_value,
            'permutations': permutations,
            'permuted_deltas': permuted_deltas,
            'method': 'MRPP'
        }

    @staticmethod
    def _mrpp_delta(dist_matrix: np.ndarray, group_labels: np.ndarray) -> float:
        """Weighted mean within-group distance."""
        unique_groups, group_counts = np.unique(group_labels, return_counts=True)
        n_total = len(group_labels)

        weighted_within_sum = 0.0
        weight_total = 0.0

        for group, count in zip(unique_groups, group_counts):
            if count <= 1:
                continue
            mask = group_labels == group
            sub = dist_matrix[np.ix_(mask, mask)]
            triu = np.triu_indices(int(count), k=1)
            mean_within = float(sub[triu].mean())
            weight = count / n_total
            weighted_within_sum += weight * mean_within
            weight_total += weight

        # Renormalise if singleton groups were skipped so that delta stays
        # comparable with the expected value (a weighted mean, not a sum).
        if 0 < weight_total < 1:
            weighted_within_sum /= weight_total

        return weighted_within_sum

    def _calculate_mrpp_delta(self, dist_matrix: np.ndarray,
                              group_labels: np.ndarray) -> float:
        return self._mrpp_delta(np.asarray(dist_matrix, dtype=float),
                                np.asarray(group_labels))


    # ------------------------------------------------------------------
    # PERMDISP / betadisper
    # ------------------------------------------------------------------

    def permdisp(self, distance_matrix: Union[pd.DataFrame, np.ndarray],
                 groups: Union[pd.Series, List],
                 permutations: int = 999,
                 centroid_type: str = 'centroid',
                 pairwise: bool = False,
                 random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        PERMDISP: test of multivariate homogeneity of group dispersions.

        Anderson (2006). Each sample is placed in principal-coordinate space,
        its distance to its own group's centroid (or spatial median) is
        computed, and those distances are compared between groups with a
        one-way ANOVA plus a permutation test.

        This is the necessary companion to :meth:`permanova`. PERMANOVA is
        sensitive to differences in *dispersion* as well as differences in
        *location*, so a significant PERMANOVA with a significant PERMDISP may
        reflect groups of unequal spread rather than groups centred in
        different places. Report both.

        Parameters:
        -----------
        distance_matrix : pd.DataFrame or np.ndarray
            Square distance matrix, or a raw data matrix (Euclidean distances
            are then computed internally).
        groups : pd.Series or list
            Group assignments.
        permutations : int
            Number of permutations of the distance-to-centroid values.
        centroid_type : {'centroid', 'median'}
            Use the group centroid (mean) or the spatial median. The spatial
            median is more robust to outliers.
        pairwise : bool
            Also run permutation tests for every pair of groups.
        random_state : int, optional
            Seed for reproducible permutations.

        Returns:
        --------
        dict
            ``f_statistic``, ``p_value``, the per-sample
            ``distances_to_centroid``, per-group ``group_dispersions`` and
            (optionally) a ``pairwise`` table.

        Notes
        -----
        Non-Euclidean distances give the Gower-centred matrix negative
        eigenvalues. Following Anderson (2006), axes with negative eigenvalues
        contribute negatively to the squared distance-to-centroid, and the
        absolute value is taken before the square root.
        """
        if centroid_type not in ('centroid', 'median'):
            raise ValueError("centroid_type must be 'centroid' or 'median'")

        dist_matrix = _as_square_distance(distance_matrix)
        group_labels = _as_labels(groups)

        if len(group_labels) != dist_matrix.shape[0]:
            raise ValueError("groups length does not match the distance matrix")

        unique_groups = np.unique(group_labels)
        if len(unique_groups) < 2:
            raise ValueError("PERMDISP requires at least two groups")

        distances_to_centroid = self._distances_to_centroid(
            dist_matrix, group_labels, unique_groups, centroid_type)

        observed_f = self._levene_f(distances_to_centroid, group_labels, unique_groups)

        rng = as_generator(random_state)
        permuted_f = np.empty(permutations, dtype=float)
        for i in range(permutations):
            permuted_f[i] = self._levene_f(
                rng.permutation(distances_to_centroid), group_labels, unique_groups)

        p_value = float((np.sum(permuted_f >= observed_f) + 1) / (permutations + 1))

        n_samples = len(group_labels)
        n_groups = len(unique_groups)
        df_between, df_within = n_groups - 1, n_samples - n_groups

        index = (distance_matrix.index if isinstance(distance_matrix, pd.DataFrame)
                 and distance_matrix.shape[0] == distance_matrix.shape[1]
                 else pd.RangeIndex(n_samples))

        group_dispersions = {
            group: {
                'mean_distance_to_centroid': float(
                    distances_to_centroid[group_labels == group].mean()),
                'sd': float(distances_to_centroid[group_labels == group].std(ddof=1))
                if int(np.sum(group_labels == group)) > 1 else 0.0,
                'n': int(np.sum(group_labels == group)),
            }
            for group in unique_groups
        }

        results: Dict[str, Any] = {
            'f_statistic': observed_f,
            'p_value': p_value,
            'df_between': df_between,
            'df_within': df_within,
            'distances_to_centroid': pd.Series(distances_to_centroid, index=index,
                                               name='distance_to_centroid'),
            'group_dispersions': group_dispersions,
            'centroid_type': centroid_type,
            'permutations': permutations,
            'permuted_f_statistics': permuted_f,
            'method': 'PERMDISP',
        }

        if pairwise:
            results['pairwise'] = self._permdisp_pairwise(
                distances_to_centroid, group_labels, unique_groups,
                permutations, rng)

        return results

    # betadisper is vegan's name for the same procedure.
    betadisper = permdisp

    def _distances_to_centroid(self, dist_matrix: np.ndarray,
                               group_labels: np.ndarray,
                               unique_groups: np.ndarray,
                               centroid_type: str) -> np.ndarray:
        """Distance from each sample to its group centroid in PCoA space."""
        n = dist_matrix.shape[0]
        centering = np.eye(n) - np.ones((n, n)) / n
        gower = -0.5 * centering @ (dist_matrix ** 2) @ centering
        gower = (gower + gower.T) / 2

        eigenvalues, eigenvectors = np.linalg.eigh(gower)

        # Keep every non-trivial axis; negative eigenvalues carry real
        # information about how non-Euclidean the distance is.
        keep = np.abs(eigenvalues) > 1e-10
        eigenvalues = eigenvalues[keep]
        coordinates = eigenvectors[:, keep] * np.sqrt(np.abs(eigenvalues))
        positive = eigenvalues > 0

        distances = np.zeros(n, dtype=float)
        for group in unique_groups:
            mask = group_labels == group
            block = coordinates[mask]
            if block.shape[0] == 0:
                continue

            if centroid_type == 'median':
                centre = self._spatial_median(block)
            else:
                centre = block.mean(axis=0)

            deviations = block - centre
            # Real axes add to the squared distance, imaginary axes subtract.
            squared = (np.sum(deviations[:, positive] ** 2, axis=1)
                       - np.sum(deviations[:, ~positive] ** 2, axis=1))
            distances[mask] = np.sqrt(np.abs(squared))

        return distances

    @staticmethod
    def _spatial_median(points: np.ndarray, max_iter: int = 200,
                        tol: float = 1e-8) -> np.ndarray:
        """Geometric (spatial) median via Weiszfeld's algorithm."""
        centre = points.mean(axis=0)
        for _ in range(max_iter):
            offsets = points - centre
            norms = np.sqrt((offsets ** 2).sum(axis=1))
            near = norms < tol
            if near.all():
                return centre
            weights = np.where(near, 0.0, 1.0 / np.where(near, 1.0, norms))
            if weights.sum() == 0:
                return centre
            new_centre = (weights[:, None] * points).sum(axis=0) / weights.sum()
            if np.linalg.norm(new_centre - centre) < tol:
                return new_centre
            centre = new_centre
        return centre

    @staticmethod
    def _levene_f(values: np.ndarray, group_labels: np.ndarray,
                  unique_groups: np.ndarray) -> float:
        """One-way ANOVA F statistic on the distance-to-centroid values."""
        n = values.size
        k = len(unique_groups)
        if k < 2 or n <= k:
            return 0.0

        grand_mean = values.mean()
        ss_between = 0.0
        ss_within = 0.0
        for group in unique_groups:
            block = values[group_labels == group]
            if block.size == 0:
                continue
            ss_between += block.size * (block.mean() - grand_mean) ** 2
            ss_within += float(np.sum((block - block.mean()) ** 2))

        if ss_within <= 0:
            return 0.0

        return float((ss_between / (k - 1)) / (ss_within / (n - k)))

    def _permdisp_pairwise(self, distances: np.ndarray, group_labels: np.ndarray,
                           unique_groups: np.ndarray, permutations: int,
                           rng) -> pd.DataFrame:
        """Permutation test of dispersion difference for every pair of groups."""
        records = []
        for i, group_a in enumerate(unique_groups):
            for group_b in unique_groups[i + 1:]:
                mask = np.isin(group_labels, [group_a, group_b])
                sub_values = distances[mask]
                sub_labels = group_labels[mask]
                pair = np.array([group_a, group_b], dtype=unique_groups.dtype)

                observed = self._levene_f(sub_values, sub_labels, pair)
                permuted = np.array([
                    self._levene_f(rng.permutation(sub_values), sub_labels, pair)
                    for _ in range(permutations)
                ])
                p_value = float((np.sum(permuted >= observed) + 1) / (permutations + 1))

                records.append({
                    'group_1': group_a,
                    'group_2': group_b,
                    'f_statistic': observed,
                    'p_value': p_value,
                    'mean_dispersion_1': float(sub_values[sub_labels == group_a].mean()),
                    'mean_dispersion_2': float(sub_values[sub_labels == group_b].mean()),
                })

        table = pd.DataFrame(records)
        if not table.empty:
            table['p_value_holm'] = self._holm_adjust(table['p_value'].values)
        return table

    @staticmethod
    def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
        """Holm-Bonferroni step-down adjustment for multiple comparisons."""
        p_values = np.asarray(p_values, dtype=float)
        n = p_values.size
        order = np.argsort(p_values)
        adjusted = np.empty(n, dtype=float)

        running_max = 0.0
        for rank, idx in enumerate(order):
            value = (n - rank) * p_values[idx]
            running_max = max(running_max, value)
            adjusted[idx] = min(running_max, 1.0)

        return adjusted

    # ------------------------------------------------------------------
    # Multi-factor PERMANOVA (adonis2-style)
    # ------------------------------------------------------------------

    def adonis(self, distance_matrix: Union[pd.DataFrame, np.ndarray],
               data: pd.DataFrame,
               terms: Optional[List[str]] = None,
               by: str = 'terms',
               permutations: int = 999,
               strata: Optional[Union[pd.Series, np.ndarray]] = None,
               random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        Multi-factor PERMANOVA (``adonis2``-style).

        Partitions a distance matrix among several explanatory variables, which
        may be categorical or continuous, with optional interactions.

        Parameters:
        -----------
        distance_matrix : pd.DataFrame or np.ndarray
            Square distance matrix, or a raw data matrix.
        data : pd.DataFrame
            Explanatory variables, one row per sample, in the same order as the
            distance matrix.
        terms : list of str, optional
            Model terms. Use ``'a:b'`` for an interaction and ``'a*b'`` as
            shorthand for ``a + b + a:b``. Defaults to every column of ``data``
            as a main effect.
        by : {'terms', 'margin'}
            ``'terms'`` gives sequential (Type I) sums of squares, so the order
            of ``terms`` matters. ``'margin'`` gives marginal (Type III) sums of
            squares, where each term is assessed with all others present.
        permutations : int
            Number of permutations.
        strata : array-like, optional
            Restrict permutations to within these blocks.
        random_state : int, optional
            Seed for reproducible permutations.

        Returns:
        --------
        dict
            ``table`` (an ANOVA-style DataFrame with Df, SumOfSqs, R2, F and
            Pr(>F) per term), plus the total and residual sums of squares.

        Notes
        -----
        Sums of squares are computed from the Gower-centred matrix
        ``G = -0.5 * H D^2 H``: the total is ``trace(G)`` and the sum of squares
        explained by a model with hat matrix ``Hx`` is ``trace(Hx G Hx)``. For a
        single categorical factor this reduces exactly to :meth:`permanova`.
        """
        if by not in ('terms', 'margin'):
            raise ValueError("by must be 'terms' or 'margin'")

        dist_matrix = _as_square_distance(distance_matrix)
        n = dist_matrix.shape[0]

        if len(data) != n:
            raise ValueError(
                f"data has {len(data)} rows but the distance matrix has {n} samples")

        if terms is None:
            terms = list(data.columns)
        terms = self._expand_terms(terms)

        gower = self._gower_center(dist_matrix)
        ss_total = float(np.trace(gower))

        # Design matrix columns for each term, all centred so the intercept is
        # handled separately.
        term_columns = {term: self._term_design(data, term) for term in terms}

        rng = as_generator(random_state)
        strata_array = None if strata is None else np.asarray(strata)

        observed = self._adonis_statistics(gower, term_columns, terms, by, ss_total)

        permuted_f = {term: np.empty(permutations, dtype=float) for term in terms}
        for p in range(permutations):
            order = self._permutation_order(n, rng, strata_array)
            permuted_gower = gower[np.ix_(order, order)]
            stats_p = self._adonis_statistics(
                permuted_gower, term_columns, terms, by, float(np.trace(permuted_gower)))
            for term in terms:
                permuted_f[term][p] = stats_p['f'][term]

        rows = []
        for term in terms:
            f_obs = observed['f'][term]
            p_value = float((np.sum(permuted_f[term] >= f_obs) + 1) / (permutations + 1))
            rows.append({
                'term': term,
                'Df': observed['df'][term],
                'SumOfSqs': observed['ss'][term],
                'R2': observed['ss'][term] / ss_total if ss_total > 0 else np.nan,
                'F': f_obs,
                'Pr(>F)': p_value,
            })

        rows.append({'term': 'Residual', 'Df': observed['df_residual'],
                     'SumOfSqs': observed['ss_residual'],
                     'R2': observed['ss_residual'] / ss_total if ss_total > 0 else np.nan,
                     'F': np.nan, 'Pr(>F)': np.nan})
        rows.append({'term': 'Total', 'Df': n - 1, 'SumOfSqs': ss_total,
                     'R2': 1.0, 'F': np.nan, 'Pr(>F)': np.nan})

        return {
            'table': pd.DataFrame(rows).set_index('term'),
            'ss_total': ss_total,
            'ss_residual': observed['ss_residual'],
            'terms': terms,
            'by': by,
            'permutations': permutations,
            'method': 'PERMANOVA (adonis)',
        }

    @staticmethod
    def _expand_terms(terms: List[str]) -> List[str]:
        """Expand ``a*b`` shorthand into main effects plus their interaction."""
        expanded: List[str] = []
        for term in terms:
            if '*' in term:
                parts = [p.strip() for p in term.split('*')]
                for part in parts:
                    if part not in expanded:
                        expanded.append(part)
                interaction = ':'.join(parts)
                if interaction not in expanded:
                    expanded.append(interaction)
            elif term not in expanded:
                expanded.append(term)
        return expanded

    @staticmethod
    def _gower_center(dist_matrix: np.ndarray) -> np.ndarray:
        """G = -0.5 * H D^2 H, whose trace is the PERMANOVA total sum of squares."""
        n = dist_matrix.shape[0]
        centering = np.eye(n) - np.ones((n, n)) / n
        gower = -0.5 * centering @ (dist_matrix ** 2) @ centering
        return (gower + gower.T) / 2

    @staticmethod
    def _factor_columns(values: np.ndarray) -> np.ndarray:
        """Dummy-code a variable: contrast columns for factors, itself if numeric."""
        series = pd.Series(values)
        if pd.api.types.is_numeric_dtype(series) and series.nunique() > 2:
            return series.to_numpy(dtype=float).reshape(-1, 1)

        levels = pd.unique(series)
        if len(levels) < 2:
            return np.zeros((len(series), 0))
        # Drop the first level; the intercept is removed by centring later.
        return np.column_stack([(series == level).to_numpy(dtype=float)
                                for level in levels[1:]])

    def _term_design(self, data: pd.DataFrame, term: str) -> np.ndarray:
        """Centred design columns for one model term (main effect or interaction)."""
        if ':' in term:
            blocks = [self._factor_columns(data[part.strip()].to_numpy())
                      for part in term.split(':')]
            design = blocks[0]
            for block in blocks[1:]:
                if design.shape[1] == 0 or block.shape[1] == 0:
                    design = np.zeros((len(data), 0))
                    break
                design = np.column_stack([
                    design[:, i] * block[:, j]
                    for i in range(design.shape[1])
                    for j in range(block.shape[1])
                ])
        else:
            if term not in data.columns:
                raise ValueError(f"Term '{term}' is not a column of data")
            design = self._factor_columns(data[term].to_numpy())

        if design.shape[1] == 0:
            return design
        return design - design.mean(axis=0)

    @staticmethod
    def _explained_ss(gower: np.ndarray, design: np.ndarray) -> float:
        """trace(Hx G Hx) for the hat matrix of ``design``."""
        if design.size == 0 or design.shape[1] == 0:
            return 0.0
        hat = design @ np.linalg.pinv(design.T @ design) @ design.T
        return float(np.trace(hat @ gower @ hat))

    @staticmethod
    def _design_rank(design: np.ndarray) -> int:
        if design.size == 0 or design.shape[1] == 0:
            return 0
        return int(np.linalg.matrix_rank(design))

    def _adonis_statistics(self, gower: np.ndarray,
                           term_columns: Dict[str, np.ndarray],
                           terms: List[str], by: str,
                           ss_total: float) -> Dict[str, Any]:
        """Sums of squares, degrees of freedom and F for every term."""
        n = gower.shape[0]

        def stack(selected: List[str]) -> np.ndarray:
            blocks = [term_columns[t] for t in selected
                      if term_columns[t].shape[1] > 0]
            if not blocks:
                return np.zeros((n, 0))
            return np.column_stack(blocks)

        full_design = stack(terms)
        ss_model = self._explained_ss(gower, full_design)
        df_model = self._design_rank(full_design)
        ss_residual = ss_total - ss_model
        df_residual = n - 1 - df_model

        ss: Dict[str, float] = {}
        df: Dict[str, int] = {}

        if by == 'terms':
            # Sequential: each term's SS is what it adds to the terms before it.
            previous_ss, previous_df = 0.0, 0
            for i, term in enumerate(terms):
                design = stack(terms[:i + 1])
                current_ss = self._explained_ss(gower, design)
                current_df = self._design_rank(design)
                ss[term] = current_ss - previous_ss
                df[term] = current_df - previous_df
                previous_ss, previous_df = current_ss, current_df
        else:
            # Marginal: each term's SS is what it adds when entered last.
            for term in terms:
                others = [t for t in terms if t != term]
                reduced_design = stack(others)
                ss[term] = ss_model - self._explained_ss(gower, reduced_design)
                df[term] = df_model - self._design_rank(reduced_design)

        ms_residual = ss_residual / df_residual if df_residual > 0 else np.nan
        f_values = {
            term: (ss[term] / df[term]) / ms_residual
            if df[term] > 0 and ms_residual and ms_residual > 0 else 0.0
            for term in terms
        }

        return {'ss': ss, 'df': df, 'f': f_values,
                'ss_residual': ss_residual, 'df_residual': df_residual}

    @staticmethod
    def _permutation_order(n: int, rng, strata: Optional[np.ndarray]) -> np.ndarray:
        """A permutation of 0..n-1, restricted within strata when given."""
        if strata is None:
            return rng.permutation(n)

        order = np.arange(n)
        for level in np.unique(strata):
            idx = np.where(strata == level)[0]
            order[idx] = rng.permutation(idx)
        return order

    # ------------------------------------------------------------------
    # Mantel tests
    # ------------------------------------------------------------------

    def mantel_test(self, matrix1: Union[pd.DataFrame, np.ndarray],
                    matrix2: Union[pd.DataFrame, np.ndarray],
                    permutations: int = 999,
                    method: str = 'pearson',
                    alternative: str = 'greater',
                    random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        Mantel test for correlation between two distance matrices.

        Parameters:
        -----------
        matrix1, matrix2 : pd.DataFrame or np.ndarray
            Square distance matrices with matching dimensions.
        permutations : int
            Number of row/column permutations of ``matrix2``.
        method : {'pearson', 'spearman', 'kendall'}
            Correlation coefficient.
        alternative : {'greater', 'less', 'two-sided'}
            Direction of the test. ``'greater'`` (the ecological convention,
            matching vegan's ``mantel``) is the default.
        random_state : int, optional
            Seed for reproducible permutations.

        Returns:
        --------
        dict
            Mantel test results.
        """
        mat1 = matrix1.values if isinstance(matrix1, pd.DataFrame) else np.asarray(matrix1)
        mat2 = matrix2.values if isinstance(matrix2, pd.DataFrame) else np.asarray(matrix2)

        if mat1.shape != mat2.shape:
            raise ValueError("Matrices must have the same dimensions")
        if alternative not in ('greater', 'less', 'two-sided'):
            raise ValueError("alternative must be 'greater', 'less' or 'two-sided'")

        n = mat1.shape[0]
        triu = np.triu_indices(n, k=1)

        vec1 = mat1[triu]
        vec2 = mat2[triu]

        corr = self._correlation(method)
        observed_r = corr(vec1, vec2)

        rng = as_generator(random_state)
        permuted_correlations = np.empty(permutations, dtype=float)
        for i in range(permutations):
            perm = rng.permutation(n)
            permuted_correlations[i] = corr(vec1, mat2[np.ix_(perm, perm)][triu])

        p_value = self._permutation_p(observed_r, permuted_correlations, alternative)

        return {
            'correlation': observed_r,
            'p_value': p_value,
            'alternative': alternative,
            'permutations': permutations,
            'method': f'Mantel_{method}',
            'permuted_correlations': permuted_correlations
        }

    def partial_mantel_test(self, matrix1: Union[pd.DataFrame, np.ndarray],
                            matrix2: Union[pd.DataFrame, np.ndarray],
                            matrix3: Union[pd.DataFrame, np.ndarray],
                            permutations: int = 999,
                            method: str = 'pearson',
                            alternative: str = 'greater',
                            random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        Partial Mantel test of matrix1 vs matrix2 controlling for matrix3.

        Parameters:
        -----------
        matrix1, matrix2 : pd.DataFrame or np.ndarray
            Distance matrices to correlate.
        matrix3 : pd.DataFrame or np.ndarray
            Control (conditioning) distance matrix.
        permutations : int
            Number of permutations.
        method : {'pearson', 'spearman', 'kendall'}
            Correlation coefficient.
        alternative : {'greater', 'less', 'two-sided'}
            Direction of the test.
        random_state : int, optional
            Seed for reproducible permutations.

        Returns:
        --------
        dict
            Partial Mantel test results.
        """
        def as_array(matrix):
            return matrix.values if isinstance(matrix, pd.DataFrame) else np.asarray(matrix)

        mat1, mat2, mat3 = as_array(matrix1), as_array(matrix2), as_array(matrix3)
        if not (mat1.shape == mat2.shape == mat3.shape):
            raise ValueError("All three matrices must have the same dimensions")
        if alternative not in ('greater', 'less', 'two-sided'):
            raise ValueError("alternative must be 'greater', 'less' or 'two-sided'")

        n = mat1.shape[0]
        triu = np.triu_indices(n, k=1)
        corr = self._correlation(method)

        vec1 = mat1[triu]
        vec3 = mat3[triu]

        def partial_correlation(x, y, z):
            rxy, rxz, ryz = corr(x, y), corr(x, z), corr(y, z)
            denominator = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
            if denominator == 0:
                return 0.0
            return float((rxy - rxz * ryz) / denominator)

        observed_partial_r = partial_correlation(vec1, mat2[triu], vec3)

        rng = as_generator(random_state)
        permuted = np.empty(permutations, dtype=float)
        for i in range(permutations):
            perm = rng.permutation(n)
            permuted[i] = partial_correlation(vec1, mat2[np.ix_(perm, perm)][triu], vec3)

        p_value = self._permutation_p(observed_partial_r, permuted, alternative)

        return {
            'partial_correlation': observed_partial_r,
            'p_value': p_value,
            'alternative': alternative,
            'permutations': permutations,
            'method': f'Partial_Mantel_{method}',
            'permuted_correlations': permuted
        }

    @staticmethod
    def _correlation(method: str):
        """Return a NaN-safe correlation function for the requested method."""
        method = method.lower()

        def safe(value) -> float:
            value = float(value)
            return 0.0 if np.isnan(value) else value

        if method == 'pearson':
            return lambda x, y: safe(np.corrcoef(x, y)[0, 1])
        if method == 'spearman':
            return lambda x, y: safe(stats.spearmanr(x, y)[0])
        if method == 'kendall':
            return lambda x, y: safe(stats.kendalltau(x, y)[0])
        raise ValueError(f"Unknown correlation method: {method}")

    @staticmethod
    def _permutation_p(observed: float, permuted: np.ndarray,
                       alternative: str) -> float:
        """Permutation p-value with the standard (count + 1) / (n + 1) form."""
        n = permuted.size
        if alternative == 'greater':
            count = np.sum(permuted >= observed)
        elif alternative == 'less':
            count = np.sum(permuted <= observed)
        else:
            count = np.sum(np.abs(permuted) >= abs(observed))
        return float((count + 1) / (n + 1))

    # ------------------------------------------------------------------
    # Indicator species analysis
    # ------------------------------------------------------------------

    def indicator_species_analysis(self, species_data: pd.DataFrame,
                                   groups: Union[pd.Series, List],
                                   permutations: int = 999,
                                   random_state: Optional[int] = None,
                                   as_frame: bool = False) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Dufrene-Legendre Indicator Species Analysis (IndVal).

        For species *j* in group *k*:

        * ``A_kj`` (specificity) = mean abundance of *j* in *k* divided by the
          sum of its mean abundances over all groups;
        * ``B_kj`` (fidelity) = proportion of sites in *k* containing *j*;
        * ``IndVal_kj = A_kj * B_kj * 100``.

        Using *mean* rather than *total* abundance is what makes the statistic
        insensitive to unequal group sizes.

        Parameters:
        -----------
        species_data : pd.DataFrame
            Species abundance matrix (sites x species)
        groups : pd.Series or list
            Group assignments
        permutations : int
            Number of permutations for the significance test (0 to skip).
        random_state : int, optional
            Seed for reproducible permutations.
        as_frame : bool
            Return a tidy DataFrame (one row per species x group) instead of
            the nested dictionary.

        Returns:
        --------
        dict or pd.DataFrame
            IndVal results per species.
        """
        group_labels = _as_labels(groups)
        if len(group_labels) != len(species_data):
            raise ValueError("groups length does not match the number of sites")

        unique_groups = np.unique(group_labels)
        abundances = species_data.values.astype(float)
        presence = abundances > 0

        indval, specificity, fidelity = self._indval_matrix(
            abundances, presence, group_labels, unique_groups
        )

        best_idx = np.argmax(indval, axis=0)
        max_indval = indval[best_idx, np.arange(indval.shape[1])]

        p_values = None
        if permutations and permutations > 0:
            rng = as_generator(random_state)
            exceed = np.zeros(indval.shape[1], dtype=int)
            for _ in range(permutations):
                permuted = rng.permutation(group_labels)
                perm_indval, _, _ = self._indval_matrix(
                    abundances, presence, permuted, unique_groups
                )
                exceed += (perm_indval.max(axis=0) >= max_indval)
            p_values = (exceed + 1) / (permutations + 1)

        if as_frame:
            records = []
            for j, species in enumerate(species_data.columns):
                for k, group in enumerate(unique_groups):
                    records.append({
                        'species': species,
                        'cluster': group,
                        'indicator_value': indval[k, j],
                        'specificity': specificity[k, j],
                        'fidelity': fidelity[k, j],
                        'is_max_group': bool(k == best_idx[j]),
                        'p_value': float(p_values[j]) if p_values is not None else np.nan,
                    })
            return pd.DataFrame(records)

        results: Dict[str, Any] = {}
        for j, species in enumerate(species_data.columns):
            group_details = {
                group: {
                    'indval': float(indval[k, j]),
                    'relative_abundance': float(specificity[k, j]),
                    'relative_frequency': float(fidelity[k, j]),
                }
                for k, group in enumerate(unique_groups)
            }
            results[species] = {
                'max_group': unique_groups[best_idx[j]],
                'indval': float(max_indval[j]),
                'p_value': float(p_values[j]) if p_values is not None else np.nan,
                'group_details': group_details,
            }
        return results

    @staticmethod
    def _indval_matrix(abundances: np.ndarray, presence: np.ndarray,
                       group_labels: np.ndarray, unique_groups: np.ndarray):
        """Return (indval, specificity, fidelity) arrays of shape (groups, species)."""
        n_groups = len(unique_groups)
        n_species = abundances.shape[1]

        mean_abundance = np.zeros((n_groups, n_species), dtype=float)
        fidelity = np.zeros((n_groups, n_species), dtype=float)

        for k, group in enumerate(unique_groups):
            mask = group_labels == group
            n_in_group = int(np.sum(mask))
            if n_in_group == 0:
                continue
            mean_abundance[k] = abundances[mask].mean(axis=0)
            fidelity[k] = presence[mask].mean(axis=0)

        totals = mean_abundance.sum(axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            specificity = np.where(totals > 0, mean_abundance / totals, 0.0)

        indval = specificity * fidelity * 100
        return indval, specificity, fidelity

    # ------------------------------------------------------------------
    # SIMPER
    # ------------------------------------------------------------------

    def simper_analysis(self, species_data: pd.DataFrame,
                        groups: Union[pd.Series, List],
                        distance_metric: str = 'bray_curtis') -> Dict[str, Any]:
        """
        Similarity Percentages (SIMPER) analysis.

        Decomposes the average between-group Bray-Curtis dissimilarity (and the
        average within-group similarity) into per-species contributions.

        Parameters:
        -----------
        species_data : pd.DataFrame
            Species abundance matrix
        groups : pd.Series or list
            Group assignments
        distance_metric : {'bray_curtis', 'euclidean'}
            Dissimilarity used for the decomposition.

        Returns:
        --------
        dict
            Per-group and per-group-pair results. Each entry contains a
            ``species_table`` DataFrame ordered by mean contribution, with
            ``contribution``, ``contribution_pct``, ``cumulative_pct``, ``sd``
            and ``ratio`` (contribution / sd, a consistency measure).
        """
        group_labels = _as_labels(groups)
        if len(group_labels) != len(species_data):
            raise ValueError("groups length does not match the number of sites")

        unique_groups = np.unique(group_labels)
        values = species_data.values.astype(float)
        columns = list(species_data.columns)
        results: Dict[str, Any] = {}

        for group in unique_groups:
            mask = group_labels == group
            block = values[mask]
            if block.shape[0] < 2:
                continue

            contributions, totals = self._pairwise_contributions(
                block, block, distance_metric, within=True
            )
            table = self._simper_table(contributions, columns)
            results[f'within_{group}'] = {
                'average_similarity': float(1 - np.mean(totals)),
                'average_dissimilarity': float(np.mean(totals)),
                'species_contributions': dict(zip(columns, contributions.mean(axis=0))),
                'species_table': table,
                'n_pairs': int(contributions.shape[0]),
            }

        for i, group1 in enumerate(unique_groups):
            for group2 in unique_groups[i + 1:]:
                block1 = values[group_labels == group1]
                block2 = values[group_labels == group2]
                if block1.size == 0 or block2.size == 0:
                    continue

                contributions, totals = self._pairwise_contributions(
                    block1, block2, distance_metric, within=False
                )
                table = self._simper_table(contributions, columns)
                results[f'between_{group1}_{group2}'] = {
                    'average_dissimilarity': float(np.mean(totals)),
                    'species_contributions': dict(zip(columns, contributions.mean(axis=0))),
                    'species_table': table,
                    'n_pairs': int(contributions.shape[0]),
                }

        return results

    @staticmethod
    def _pairwise_contributions(block1: np.ndarray, block2: np.ndarray,
                                distance_metric: str, within: bool):
        """Per-species contributions for every cross (or within) sample pair."""
        if within:
            i_idx, j_idx = np.triu_indices(block1.shape[0], k=1)
            left, right = block1[i_idx], block1[j_idx]
        else:
            left = np.repeat(block1, block2.shape[0], axis=0)
            right = np.tile(block2, (block1.shape[0], 1))

        if distance_metric == 'bray_curtis':
            # Bray-Curtis decomposes exactly: d = sum_i |x_i - y_i| / sum(x + y)
            denom = left.sum(axis=1) + right.sum(axis=1)
            denom = np.where(denom == 0, 1.0, denom)
            contributions = np.abs(left - right) / denom[:, None]
        elif distance_metric == 'euclidean':
            # Squared Euclidean distance decomposes exactly: d^2 = sum_i (x_i - y_i)^2
            contributions = (left - right) ** 2
        else:
            raise ValueError(
                "SIMPER supports distance_metric='bray_curtis' or 'euclidean'; "
                f"got '{distance_metric}'."
            )

        totals = contributions.sum(axis=1)
        return contributions, totals

    @staticmethod
    def _simper_table(contributions: np.ndarray, columns: List[str]) -> pd.DataFrame:
        """Summarise per-species contributions into an ordered table."""
        mean_contrib = contributions.mean(axis=0)
        sd_contrib = contributions.std(axis=0, ddof=1) if contributions.shape[0] > 1 \
            else np.zeros_like(mean_contrib)

        total = mean_contrib.sum()
        pct = (mean_contrib / total * 100) if total > 0 else np.zeros_like(mean_contrib)

        with np.errstate(invalid='ignore', divide='ignore'):
            ratio = np.where(sd_contrib > 0, mean_contrib / sd_contrib, np.nan)

        table = pd.DataFrame({
            'species': columns,
            'contribution': mean_contrib,
            'sd': sd_contrib,
            'ratio': ratio,
            'contribution_pct': pct,
        }).sort_values('contribution', ascending=False).reset_index(drop=True)

        table['cumulative_pct'] = table['contribution_pct'].cumsum()
        return table

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------

    def _bray_curtis_single(self, x: np.ndarray, y: np.ndarray) -> float:
        """Bray-Curtis distance between two samples."""
        numerator = float(np.sum(np.abs(x - y)))
        denominator = float(np.sum(x + y))
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def _bray_curtis_distance(self, data: np.ndarray) -> np.ndarray:
        """Condensed Bray-Curtis distance vector."""
        return self._condensed_distances(data, 'bray_curtis')

    def _jaccard_distance(self, data: np.ndarray) -> np.ndarray:
        """Condensed Jaccard distance vector (presence/absence)."""
        return self._condensed_distances(data, 'jaccard')

    def _sorensen_distance(self, data: np.ndarray) -> np.ndarray:
        """Condensed Sorensen (Dice) distance vector (presence/absence)."""
        return self._condensed_distances(data, 'sorensen')

    def _euclidean_distance(self, data: np.ndarray) -> np.ndarray:
        """Condensed Euclidean distance vector."""
        return self._condensed_distances(data, 'euclidean')

    def _manhattan_distance(self, data: np.ndarray) -> np.ndarray:
        """Condensed Manhattan distance vector."""
        return self._condensed_distances(data, 'manhattan')
