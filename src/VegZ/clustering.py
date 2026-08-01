"""
Comprehensive clustering analysis module for vegetation classification.

Copyright (c) 2025 Mohamed Z. Hatim
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import warnings

from ._compat import as_generator

#: VegZ distance names mapped onto SciPy pdist metrics.
_PDIST_ALIASES = {
    'bray_curtis': 'braycurtis',
    'braycurtis': 'braycurtis',
    'sorensen': 'dice',
    'manhattan': 'cityblock',
    'city_block': 'cityblock',
}


def _condensed_distances(values: np.ndarray, metric: str) -> np.ndarray:
    """Condensed distance vector honouring VegZ metric aliases."""
    key = str(metric).lower()
    scipy_metric = _PDIST_ALIASES.get(key, key)
    values = np.asarray(values, dtype=float)
    if scipy_metric in ('jaccard', 'dice'):
        values = (values > 0).astype(bool)
    return pdist(values, metric=scipy_metric)


def _total_sum_of_squares(data: np.ndarray) -> float:
    """
    Total within-cluster sum of squares for a single cluster (k = 1).

    This is the natural k=1 value of KMeans ``inertia_`` and must be on the
    same scale as it, otherwise the elbow curve is not monotone decreasing and
    every elbow detector is thrown off.
    """
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        return 0.0
    return float(np.sum((data - data.mean(axis=0)) ** 2))


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
                 cut_levels: Optional[List[float]] = None,
                 max_divisions: int = 6,
                 min_group_size: int = 5,
                 max_depth: Optional[int] = None,
                 min_eigenvalue: float = 0.05) -> Dict[str, Any]:
        """
        Two-Way Indicator Species Analysis (TWINSPAN).

        A divisive hierarchical classification. At each step the current group
        of sites is ordinated by correspondence analysis of its pseudospecies
        table and split at the centroid of the first (non-trivial) axis; the
        split is then refined using the pseudospecies that best discriminate
        the two halves.

        Parameters:
        -----------
        data : pd.DataFrame
            Species abundance matrix (sites x species), non-negative.
        cut_levels : list, optional
            Pseudospecies cut levels; the first level denotes mere presence.
            Defaults to ``[0, 2, 5, 10, 20]``.
        max_divisions : int
            Maximum number of splits performed in total.
        min_group_size : int
            A group is only split if both halves would have at least this many
            sites.
        max_depth : int, optional
            Maximum depth of the classification hierarchy. Defaults to
            unlimited (bounded by ``max_divisions``).
        min_eigenvalue : float
            Minimum first-axis eigenvalue required to accept a split. Groups
            with weaker internal structure become terminal.

        Returns:
        --------
        dict
            ``site_classification`` (terminal group per site, numbered from 1),
            ``classification_tree`` (divisions, terminal groups and their
            indicator pseudospecies) and the pseudospecies table.

        Notes
        -----
        This follows the logic of Hill's (1979) TWINSPAN but is not a
        line-by-line port of DECORANA/TWINSPAN: the species classification is
        derived from the indicator pseudospecies of each division rather than a
        separate two-way ordination, so group memberships can differ slightly
        from the original FORTRAN program.
        """
        if cut_levels is None:
            cut_levels = [0, 2, 5, 10, 20]
        if (data.values < 0).any():
            raise ValueError("TWINSPAN requires non-negative abundance data")
        if min_group_size < 2:
            raise ValueError("min_group_size must be at least 2")

        pseudo_species_data = self._create_pseudospecies(data, cut_levels)

        classification_tree: Dict[str, Any] = {
            'divisions': [],
            'groups': {},
            'indicator_species': {}
        }

        root = {
            'sites': data.index.tolist(),
            'level': 0,
            'parent': None,
            'eigenvalue': np.inf,   # always considered first
            'path': '',
        }

        pending = [root]
        divisions_done = 0
        group_id = 1

        while pending and divisions_done < max_divisions:
            # Split the group with the strongest internal gradient first.
            current = max(pending, key=lambda g: g['eigenvalue'])
            pending.remove(current)

            too_small = len(current['sites']) < 2 * min_group_size
            too_deep = max_depth is not None and current['level'] >= max_depth

            if too_small or too_deep:
                classification_tree['groups'][group_id] = current
                group_id += 1
                continue

            division = self._twinspan_division(
                pseudo_species_data.loc[current['sites']],
                min_group_size=min_group_size
            )

            if (division['eigenvalue'] < min_eigenvalue
                    or not division['group1_indices']
                    or not division['group2_indices']):
                classification_tree['groups'][group_id] = current
                group_id += 1
                continue

            sites = current['sites']
            children = []
            for side, indices in (('0', division['group1_indices']),
                                  ('1', division['group2_indices'])):
                children.append({
                    'sites': [sites[i] for i in indices],
                    'level': current['level'] + 1,
                    'parent': current['path'],
                    'eigenvalue': division['eigenvalue'],
                    'path': current['path'] + side,
                })

            classification_tree['divisions'].append({
                'division_id': divisions_done,
                'level': current['level'],
                'parent_path': current['path'],
                'child_paths': [c['path'] for c in children],
                'group_sizes': [len(c['sites']) for c in children],
                'eigenvalue': division['eigenvalue'],
                'indicator_species': division['indicator_species'],
            })
            classification_tree['indicator_species'][current['path'] or 'root'] = \
                division['indicator_species']

            pending.extend(children)
            divisions_done += 1

        # Anything still pending is terminal.
        for remaining in pending:
            classification_tree['groups'][group_id] = remaining
            group_id += 1

        site_classification = self._assign_final_groups(classification_tree, data.index)

        return {
            'site_classification': site_classification,
            'classification_tree': classification_tree,
            'pseudospecies_data': pseudo_species_data,
            'cut_levels': cut_levels,
            'n_divisions': divisions_done,
            'n_groups': int(site_classification.nunique()),
            'method': 'TWINSPAN'
        }

    def _create_pseudospecies(self, data: pd.DataFrame,
                              cut_levels: List[float]) -> pd.DataFrame:
        """
        Expand abundances into binary pseudospecies.

        The first cut level denotes presence (abundance > 0); each subsequent
        level ``c`` adds a pseudospecies flagging ``abundance >= c``. Encoding
        presence explicitly matters - without it a species recorded only at low
        cover contributes nothing to the classification.
        """
        columns = {}

        for species in data.columns:
            values = data[species]
            for i, cut_level in enumerate(cut_levels):
                if i == 0:
                    indicator = (values > 0)
                else:
                    indicator = (values >= cut_level)
                columns[f"{species}_{i + 1}"] = indicator.astype(int)

        pseudo = pd.DataFrame(columns, index=data.index)
        # Drop pseudospecies that are constant: they cannot discriminate.
        informative = pseudo.columns[(pseudo.sum(axis=0) > 0) &
                                     (pseudo.sum(axis=0) < len(pseudo))]
        return pseudo[informative] if len(informative) else pseudo

    def _twinspan_division(self, pseudo_data: pd.DataFrame,
                           min_group_size: int = 2) -> Dict[str, Any]:
        """
        One TWINSPAN division of a group of sites.

        Uses the first non-trivial correspondence-analysis axis of the
        pseudospecies table, cut at the weighted centroid, then refines the
        split with an indicator-pseudospecies score.
        """
        n_sites = len(pseudo_data)
        empty = {
            'group1_indices': [],
            'group2_indices': [],
            'eigenvalue': 0.0,
            'indicator_species': [],
            'ordination_scores': np.zeros(n_sites),
        }

        matrix = pseudo_data.values.astype(float)
        if n_sites < 2 * min_group_size or matrix.size == 0 or matrix.sum() == 0:
            return empty

        try:
            scores, eigenvalue = self._first_ca_axis(matrix)
        except (np.linalg.LinAlgError, ValueError) as exc:  # pragma: no cover
            warnings.warn(f"TWINSPAN ordination failed: {exc}")
            return empty

        if not np.isfinite(scores).all() or np.allclose(scores, scores[0]):
            return empty

        # Split at the centroid, then move the cut if either side is too small.
        order = np.argsort(scores)
        n_left = int(np.sum(scores < scores.mean()))
        n_left = int(np.clip(n_left, min_group_size, n_sites - min_group_size))

        group1_indices = order[:n_left].tolist()
        group2_indices = order[n_left:].tolist()

        indicator_species = self._identify_indicator_pseudospecies(
            pseudo_data, group1_indices, group2_indices
        )

        # Refinement: rescore sites on the indicator pseudospecies alone and
        # reassign, keeping the group-size constraint. This is what makes the
        # classification reproducible from a handful of named indicators, which
        # is the practical point of TWINSPAN.
        refined1, refined2 = self._refine_division(
            pseudo_data, indicator_species, group1_indices, group2_indices,
            min_group_size
        )

        return {
            'group1_indices': refined1,
            'group2_indices': refined2,
            'eigenvalue': float(eigenvalue),
            'indicator_species': indicator_species,
            'ordination_scores': scores,
        }

    @staticmethod
    def _first_ca_axis(matrix: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        First non-trivial correspondence-analysis axis of a count matrix.

        Returns ``(site_scores, eigenvalue)``. Subtracting the outer product of
        the row and column masses removes the trivial all-ones axis; a plain
        reciprocal-averaging power iteration converges to that trivial axis
        instead and yields no separation at all.
        """
        total = matrix.sum()
        if total <= 0:
            raise ValueError("empty pseudospecies table")

        P = matrix / total
        r = P.sum(axis=1)
        c = P.sum(axis=0)

        keep_rows = r > 0
        keep_cols = c > 0
        if keep_rows.sum() < 2 or keep_cols.sum() < 2:
            raise ValueError("insufficient non-empty rows/columns")

        expected = np.outer(r, c)
        with np.errstate(divide='ignore', invalid='ignore'):
            S = (P - expected) / np.sqrt(expected)
        S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)

        U, s, _ = np.linalg.svd(S, full_matrices=False)

        safe_r = np.where(r > 0, r, 1.0)
        scores = U[:, 0] / np.sqrt(safe_r)
        scores = np.where(r > 0, scores, 0.0)

        return scores, float(s[0] ** 2)

    @staticmethod
    def _refine_division(pseudo_data: pd.DataFrame,
                         indicator_species: List[Dict[str, Any]],
                         group1_indices: List[int],
                         group2_indices: List[int],
                         min_group_size: int) -> Tuple[List[int], List[int]]:
        """Reassign sites using a signed indicator-pseudospecies score."""
        if not indicator_species:
            return group1_indices, group2_indices

        values = pseudo_data.values.astype(float)
        columns = list(pseudo_data.columns)

        score = np.zeros(values.shape[0])
        for indicator in indicator_species:
            col = columns.index(indicator['species'])
            # Positive weight => species favours group 2.
            sign = 1.0 if indicator['group2_frequency'] > indicator['group1_frequency'] else -1.0
            score += sign * values[:, col]

        if np.allclose(score, score[0]):
            return group1_indices, group2_indices

        order = np.argsort(score, kind='stable')
        n_sites = values.shape[0]
        n_left = int(np.sum(score < score.mean()))
        n_left = int(np.clip(n_left, min_group_size, n_sites - min_group_size))

        return order[:n_left].tolist(), order[n_left:].tolist()

    def _identify_indicator_pseudospecies(self, pseudo_data: pd.DataFrame,
                                          group1_indices: List[int],
                                          group2_indices: List[int],
                                          max_indicators: int = 5,
                                          min_difference: float = 0.2) -> List[Dict[str, Any]]:
        """Pseudospecies whose frequency differs most between the two halves."""
        if not group1_indices or not group2_indices:
            return []

        values = pseudo_data.values.astype(float)
        freq1 = values[group1_indices].mean(axis=0)
        freq2 = values[group2_indices].mean(axis=0)
        difference = np.abs(freq1 - freq2)

        indicators = [
            {
                'species': column,
                'frequency_difference': float(difference[j]),
                'group1_frequency': float(freq1[j]),
                'group2_frequency': float(freq2[j]),
            }
            for j, column in enumerate(pseudo_data.columns)
            if difference[j] > min_difference
        ]

        indicators.sort(key=lambda x: x['frequency_difference'], reverse=True)
        return indicators[:max_indicators]

    
    def _assign_final_groups(self, classification_tree: Dict[str, Any],
                             site_index: pd.Index) -> pd.Series:
        """Assign sites to final (terminal) groups, numbered from 1."""
        # Start from 0 = unassigned rather than an uninitialised int Series,
        # which pandas fills with arbitrary memory contents.
        site_groups = pd.Series(0, index=site_index, dtype=int, name='twinspan_group')

        group_counter = 1
        for _group_id, group_info in classification_tree['groups'].items():
            for site in group_info['sites']:
                if site in site_groups.index:
                    site_groups.loc[site] = group_counter
            group_counter += 1

        unassigned = int((site_groups == 0).sum())
        if unassigned:
            warnings.warn(
                f"{unassigned} site(s) were not placed in a TWINSPAN group and "
                "are labelled 0."
            )

        return site_groups
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    
    def fuzzy_cmeans_clustering(self, data: pd.DataFrame,
                               n_clusters: int = 3,
                               fuzziness: float = 2.0,
                               max_iter: int = 100,
                               tol: float = 1e-4,
                               random_state: Optional[int] = 42) -> Dict[str, Any]:
        """
        Fuzzy C-means clustering.

        Parameters:
        -----------
        data : pd.DataFrame
            Data matrix
        n_clusters : int
            Number of clusters
        fuzziness : float
            Fuzziness parameter (must be > 1)
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        random_state : int, optional
            Seed for the random membership initialisation. Without this the
            algorithm returns a different partition on every call.

        Returns:
        --------
        dict
            Fuzzy clustering results
        """
        if fuzziness <= 1:
            raise ValueError("fuzziness must be greater than 1")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")

        X = data.values.astype(float)
        n_samples, n_features = X.shape

        if n_clusters < 1 or n_clusters > n_samples:
            raise ValueError(
                f"n_clusters must be between 1 and {n_samples} (got {n_clusters})"
            )

        rng = as_generator(random_state)
        membership = rng.random((n_samples, n_clusters))
        membership = membership / membership.sum(axis=1)[:, np.newaxis]

        centers = np.zeros((n_clusters, n_features))
        iteration = 0

        for iteration in range(max_iter):  # noqa: B007 - used after the loop
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
                       n_refs: int = 10,
                       random_state: Optional[int] = 42) -> Dict[str, Any]:
        """
        Gap statistic (Tibshirani, Walther & Hastie 2001).

        Parameters:
        -----------
        data : pd.DataFrame
            Data matrix
        k_range : range
            Candidate cluster counts
        n_refs : int
            Number of uniform reference data sets per k
        random_state : int, optional
            Seed for the reference data sets. Without it the gap statistic is
            not reproducible between calls.
        """
        rng = as_generator(random_state)
        X = np.asarray(data.values, dtype=float)
        lows, highs = X.min(axis=0), X.max(axis=0)

        k_values = list(k_range)
        gaps, errors = [], []

        for k in k_values:
            wk_actual = self._within_dispersion(X, k)

            wk_refs = np.empty(n_refs, dtype=float)
            for r in range(n_refs):
                reference = rng.uniform(lows, highs, size=X.shape)
                wk_refs[r] = self._within_dispersion(reference, k)

            log_refs = np.log(np.where(wk_refs > 0, wk_refs, np.finfo(float).tiny))
            gaps.append(float(np.mean(log_refs) - np.log(max(wk_actual, np.finfo(float).tiny))))
            errors.append(float(np.std(log_refs) * np.sqrt(1 + 1 / n_refs)))

        # Tibshirani's 1-SE rule: smallest k with Gap(k) >= Gap(k+1) - s(k+1).
        optimal_k = k_values[-1]
        for i in range(len(gaps) - 1):
            if gaps[i] >= gaps[i + 1] - errors[i + 1]:
                optimal_k = k_values[i]
                break

        return {
            'gap_values': gaps,
            'standard_errors': errors,
            'optimal_k': optimal_k,
            'k_range': k_values
        }

    @staticmethod
    def _within_dispersion(X: np.ndarray, k: int) -> float:
        """Pooled within-cluster sum of squares for a given k."""
        if k <= 1:
            return _total_sum_of_squares(X)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        return float(kmeans.inertia_)
    
    def _cophenetic_correlation(self, data: pd.DataFrame,
                                linkage_matrix: np.ndarray,
                                distances: Optional[np.ndarray] = None) -> float:
        """
        Cophenetic correlation coefficient.

        Parameters:
        -----------
        data : pd.DataFrame
            Data used for the clustering (ignored when ``distances`` is given).
        linkage_matrix : np.ndarray
            SciPy linkage matrix.
        distances : np.ndarray, optional
            The *same* condensed distance vector the linkage was built from.
            Supplying it matters: correlating the tree against Euclidean
            distances when it was built from, say, Bray-Curtis distances gives
            a meaningless number.
        """
        if distances is None:
            distances = pdist(np.asarray(data.values, dtype=float))

        correlation, _ = cophenet(linkage_matrix, np.asarray(distances, dtype=float))
        return float(correlation)
    
    def optimal_clusters_analysis(self, data: pd.DataFrame,
                                k_range: range = range(2, 11),
                                methods: Optional[List[str]] = None) -> Dict[str, Any]:
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
        if methods is None:
            methods = ['silhouette', 'gap_statistic']

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
            
            # Second difference of the WCSS curve; index i of the double diff
            # corresponds to k_values[i + 1], so the offset is +1 (not +2).
            k_values = list(k_range)
            if len(wcss) > 2:
                diff2 = np.diff(np.diff(wcss))
                elbow_idx = int(np.argmax(diff2)) + 1
                elbow_idx = min(elbow_idx, len(k_values) - 1)
                results['recommendations']['elbow'] = k_values[elbow_idx]

        return results
    
    def hierarchical_clustering(self, data: pd.DataFrame,
                                method: Optional[str] = None,
                                metric: Optional[str] = None,
                                n_clusters: Optional[int] = None,
                                linkage_method: Optional[str] = None,
                                distance_metric: Optional[str] = None) -> Dict[str, Any]:
        """
        Enhanced hierarchical clustering with validation.

        Parameters:
        -----------
        data : pd.DataFrame
            Data matrix, or a square distance matrix when the metric is
            ``'precomputed'``.
        method, linkage_method : str
            Linkage method ('ward', 'average', 'complete', 'single', ...).
            ``linkage_method`` is an alias kept in step with
            :meth:`VegZ.hierarchical_clustering`; default ``'ward'``.
        metric, distance_metric : str
            Distance metric; ``distance_metric`` is an alias. Default
            ``'euclidean'``.
        n_clusters : int, optional
            Number of clusters to extract

        Returns:
        --------
        dict
            Hierarchical clustering results with validation
        """
        if method is not None and linkage_method is not None and method != linkage_method:
            raise ValueError("Pass only one of 'method' or 'linkage_method'")
        if metric is not None and distance_metric is not None and metric != distance_metric:
            raise ValueError("Pass only one of 'metric' or 'distance_metric'")

        method = method or linkage_method or 'ward'
        metric = metric or distance_metric or 'euclidean'

        if metric == 'precomputed':
            values = np.asarray(data.values, dtype=float)
            distances = squareform(values, checks=False) if values.ndim == 2 and \
                values.shape[0] == values.shape[1] else values
            silhouette_input = None
        else:
            if method == 'ward' and metric != 'euclidean':
                warnings.warn(
                    "Ward linkage is only defined for Euclidean distances; "
                    "switching distance metric to 'euclidean'."
                )
                metric = 'euclidean'
            distances = _condensed_distances(data.values, metric)
            silhouette_input = data

        linkage_matrix = linkage(distances, method=method)

        results = {
            'linkage_matrix': linkage_matrix,
            'distances': distances,
            'method': method,
            'metric': metric,
            'linkage_method': method,
            'distance_metric': metric,
            'site_labels': data.index.tolist(),
        }

        if n_clusters:
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            results['cluster_labels'] = pd.Series(cluster_labels, index=data.index, name='cluster')

            if len(set(cluster_labels)) > 1 and silhouette_input is not None:
                silhouette_results = self._silhouette_analysis(silhouette_input, cluster_labels)
                results['silhouette_score'] = silhouette_results['mean_silhouette_score']
                results['silhouette_scores'] = silhouette_results['silhouette_scores']
                results['calinski_harabasz_score'] = self._calinski_harabasz_score(
                    silhouette_input, cluster_labels
                )

        # Correlate the tree against the distances it was actually built from.
        results['cophenetic_correlation'] = self._cophenetic_correlation(
            data, linkage_matrix, distances
        )

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
        
        cluster_centers = pd.DataFrame(
            kmeans.cluster_centers_,
            columns=data.columns,
            index=[f'Cluster_{i}' for i in range(n_clusters)]
        )

        results = {
            'cluster_labels': pd.Series(cluster_labels, index=data.index, name='cluster'),
            'cluster_centers': cluster_centers,
            # 'centroids' is the name used by VegZ.kmeans_clustering; expose both.
            'centroids': cluster_centers,
            'inertia': kmeans.inertia_,
            'n_iter': kmeans.n_iter_,
            'kmeans_object': kmeans,
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
                          methods: Optional[List[str]] = None) -> Dict[str, Any]:
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
        if methods is None:
            methods = ['elbow', 'silhouette', 'gap']

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
                                    methods: Optional[List[str]] = None,
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
        if methods is None:
            methods = ['knee_locator', 'derivative', 'variance_explained',
                       'distortion_jump']

        transformed_data = self._transform_for_clustering(data, transform)
        X = np.asarray(transformed_data.values, dtype=float)

        k_values = list(k_range)
        if not k_values:
            raise ValueError("k_range is empty")
        if min(k_values) < 1:
            raise ValueError("k_range must contain values >= 1")
        if max(k_values) > X.shape[0]:
            raise ValueError(
                f"k_range goes up to {max(k_values)} but the data only has "
                f"{X.shape[0]} sites"
            )

        inertias = []
        silhouette_scores = []
        calinski_scores = []
        davies_bouldin_scores = []

        for k in k_values:
            if k == 1:
                # Total sum of squares: the k=1 value of KMeans inertia_, so the
                # curve stays on one scale and remains monotone decreasing.
                inertias.append(_total_sum_of_squares(X))
                silhouette_scores.append(np.nan)
                calinski_scores.append(np.nan)
                davies_bouldin_scores.append(np.nan)
            else:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)

                inertias.append(kmeans.inertia_)

                if len(set(labels)) > 1:
                    silhouette_scores.append(silhouette_score(X, labels))
                    calinski_scores.append(calinski_harabasz_score(X, labels))
                    davies_bouldin_scores.append(davies_bouldin_score(X, labels))
                else:  # pragma: no cover - degenerate clustering
                    silhouette_scores.append(np.nan)
                    calinski_scores.append(np.nan)
                    davies_bouldin_scores.append(np.nan)

        results = {
            'k_values': k_values,
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
            elbow_k = self._knee_locator_method(k_values, inertias)
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
            elbow_k = self._variance_explained_elbow(k_values, inertias)
            results['elbow_points']['variance_explained'] = elbow_k
            results['method_details']['variance_explained'] = {
                'description': 'Point where additional clusters explain <10% more variance'
            }
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        if 'distortion_jump' in methods:
            elbow_k = self._distortion_jump_method(k_values, inertias)
            results['elbow_points']['distortion_jump'] = elbow_k
            results['method_details']['distortion_jump'] = {
                'description': 'Jump method based on distortion changes',
                'reference': 'Sugar & James (2003)'
            }
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        if 'l_method' in methods:
            elbow_k = self._l_method_elbow(k_values, inertias)
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
        
        # Best k by each validation index. NaN entries (k = 1, where these
        # indices are undefined) are skipped rather than assumed to be at
        # position 0, so the result is correct for any k_range.
        results['recommendations']['silhouette_optimal'] = self._best_k(
            k_values, silhouette_scores, maximise=True)
        results['recommendations']['calinski_optimal'] = self._best_k(
            k_values, calinski_scores, maximise=True)
        results['recommendations']['davies_bouldin_optimal'] = self._best_k(
            k_values, davies_bouldin_scores, maximise=False)
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        if plot_results:
            results['plots'] = self._create_elbow_plots(results)
        
        return results
    
    @staticmethod
    def _transform_for_clustering(data: pd.DataFrame, transform: str) -> pd.DataFrame:
        """Apply a transformation before clustering, preserving labels."""
        transform = (transform or 'none').lower()
        values = data.values.astype(float)

        if transform in ('none', 'raw'):
            out = values
        elif transform == 'hellinger':
            row_sums = values.sum(axis=1)
            row_sums[row_sums == 0] = 1.0
            out = np.sqrt(values / row_sums[:, None])
        elif transform == 'chord':
            norms = np.sqrt((values ** 2).sum(axis=1))
            norms[norms == 0] = 1.0
            out = values / norms[:, None]
        elif transform == 'log':
            out = np.log1p(values)
        elif transform == 'sqrt':
            out = np.sqrt(np.maximum(values, 0))
        else:
            raise ValueError(
                f"Unknown transformation '{transform}'. Valid options: none, "
                "hellinger, chord, log, sqrt."
            )

        return pd.DataFrame(out, index=data.index, columns=data.columns)

    @staticmethod
    def _best_k(k_values: List[int], scores: List[float], maximise: bool) -> Optional[int]:
        """Best k by a validation index, ignoring undefined (NaN) entries."""
        arr = np.asarray(scores, dtype=float)
        finite = np.isfinite(arr)
        if not finite.any():
            return None
        candidates = np.where(finite)[0]
        best = candidates[np.argmax(arr[candidates])] if maximise \
            else candidates[np.argmin(arr[candidates])]
        return k_values[int(best)]

    def _calculate_total_variance(self, data: np.ndarray) -> float:
        """
        Total within-cluster sum of squares (the k = 1 value of KMeans inertia).

        Named ``_calculate_total_variance`` for backwards compatibility; it
        returns a sum of squares, not a mean, so that it is directly comparable
        with ``KMeans.inertia_``.
        """
        return _total_sum_of_squares(data)
    
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
    
    def _variance_explained_elbow(self, k_values: List[int], inertias: List[float],
                                  threshold: float = 0.1) -> Optional[int]:
        """
        Smallest k after which an extra cluster explains < ``threshold`` more
        of the total variance.

        The reference point is the first inertia in ``k_values``, so the result
        is meaningful for any starting k (not only k = 1).
        """
        if len(inertias) < 3:
            return None

        baseline = float(inertias[0])
        if baseline <= 0:
            return None

        for i in range(len(inertias) - 1):
            improvement = (inertias[i] - inertias[i + 1]) / baseline
            if improvement < threshold:
                return k_values[i]

        # Every step still yields a large gain: the largest k is the best guess.
        return k_values[-1]

    def _distortion_jump_method(self, k_values: List[int], inertias: List[float],
                                n_features: Optional[int] = None) -> Optional[int]:
        """
        Sugar & James (2003) jump method.

        Distortions are transformed by ``d_k^(-Y)`` with the transformation
        power ``Y = p / 2`` (``p`` = dimensionality), and the selected k is the
        one maximising the jump ``d_k^(-Y) - d_{k-1}^(-Y)``. This is the actual
        published statistic; a plain second difference of the raw inertia
        (which is what a naive implementation computes) is a different and much
        less reliable criterion.

        Parameters:
        -----------
        k_values, inertias : list
            The elbow curve.
        n_features : int, optional
            Data dimensionality ``p``. Defaults to 2, giving ``Y = 1``.
        """
        if len(inertias) < 3:
            return None

        distortions = np.asarray(inertias, dtype=float)
        k_arr = np.asarray(k_values, dtype=float)

        # Sugar & James define distortion as the mean squared error per
        # dimension; inertia is proportional to it, and the transformation is
        # scale-equivariant in k, so proportionality is harmless.
        power = (n_features if n_features else 2) / 2.0

        with np.errstate(divide='ignore', invalid='ignore'):
            transformed = np.where(distortions > 0, distortions ** (-power), np.inf)

        if not np.all(np.isfinite(transformed)):
            return None

        jumps = np.diff(transformed)
        if jumps.size == 0:
            return None

        # jumps[i] compares k_values[i + 1] against k_values[i].
        best = int(np.argmax(jumps)) + 1
        if k_arr[best] < 1:
            return None
        return k_values[best]
    
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
            
            except (np.linalg.LinAlgError, ValueError):
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
                ax1.axvline(x=elbow_k, color=colors[i % len(colors)], 
                           linestyle='--', alpha=0.7, label=f'{method}: k={elbow_k}')
        ax1.legend()
        
        def plot_index(ax, scores, ylabel, title, colour, best='max'):
            """Plot one validation index, dropping k values where it is undefined."""
            finite = [(k, s) for k, s in zip(k_values, scores)
                      if s is not None and np.isfinite(s)]
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

            if not finite:
                ax.text(0.5, 0.5, 'not available', transform=ax.transAxes,
                        ha='center', va='center')
                return

            ks, values = zip(*finite)
            ax.plot(ks, values, colour, linewidth=2, markersize=8)
            best_k = ks[int(np.argmax(values))] if best == 'max' else ks[int(np.argmin(values))]
            ax.axvline(x=best_k, color='red', linestyle='--', label=f'Best: k={best_k}')
            ax.legend()

        plot_index(axes[0, 1], metrics['silhouette_scores'],
                   'Average Silhouette Score', 'Silhouette Analysis', 'go-', 'max')
        plot_index(axes[1, 0], metrics['calinski_harabasz_scores'],
                   'Calinski-Harabasz Index', 'Calinski-Harabasz Index', 'mo-', 'max')
        plot_index(axes[1, 1], metrics['davies_bouldin_scores'],
                   'Davies-Bouldin Index', 'Davies-Bouldin Index (Lower is Better)',
                   'co-', 'min')

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
    
    def _calculate_gap_statistic(self, data: pd.DataFrame, k_range: range,
                                 n_refs: int = 10,
                                 random_state: Optional[int] = 42) -> List[float]:
        """Gap values for each k (thin wrapper around :meth:`_gap_statistic`)."""
        return self._gap_statistic(data, k_range, n_refs=n_refs,
                                   random_state=random_state)['gap_values']