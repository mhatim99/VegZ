"""
Comprehensive multivariate analysis module for vegetation data.

Copyright (c) 2025 Mohamed Z. Hatim
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Tuple, Optional, Any
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import procrustes
from scipy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

from ._compat import make_mds, as_generator

#: VegZ distance-metric names mapped onto SciPy pdist metrics.
_PDIST_ALIASES = {
    'bray_curtis': 'braycurtis',
    'braycurtis': 'braycurtis',
    'jaccard': 'jaccard',
    'sorensen': 'dice',
    'euclidean': 'euclidean',
    'manhattan': 'cityblock',
    'cityblock': 'cityblock',
    'canberra': 'canberra',
}

_BINARY_METRICS = {'jaccard', 'dice'}


class MultivariateAnalyzer:
    """Comprehensive multivariate analysis for ecological communities."""

    def __init__(self):
        """Initialize multivariate analyzer."""
        self.available_methods = [
            'pca', 'ca', 'dca', 'cca', 'rda', 'nmds', 'pcoa'
        ]

        self.significance_tests = [
            'anova_rda', 'anova_cca', 'varpart', 'forward_selection'
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

    # ------------------------------------------------------------------
    # Unconstrained ordination
    # ------------------------------------------------------------------

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
            PCA results including site scores, species scores and explained variance
        """
        transformed_data = self._transform(data, transform)

        max_components = max(1, min(data.shape[0] - 1, data.shape[1]))
        if n_components is None:
            n_components = max_components
        n_components = min(n_components, max_components)

        pca = PCA(n_components=n_components)
        site_scores = pca.fit_transform(transformed_data)

        # Species scores scaled as correlations with the axes (biplot scaling 2).
        species_scores = pca.components_.T * np.sqrt(pca.explained_variance_)

        return {
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
            'total_variance': float(np.sum(pca.explained_variance_)),
            'transform': transform,
            'pca_object': pca
        }

    def nmds_analysis(self, data: pd.DataFrame,
                      distance_metric: str = 'bray_curtis',
                      n_dimensions: int = 2,
                      max_iterations: int = 300,
                      n_init: int = 10,
                      random_state: Optional[int] = 42) -> Dict[str, Any]:
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
            Maximum SMACOF iterations, by default 300
        n_init : int, optional
            Number of random restarts, by default 10
        random_state : int, optional
            Random state for reproducibility, by default 42

        Returns
        -------
        Dict[str, Any]
            NMDS results including site scores and stress value
        """
        distances = self._condensed(data.values, distance_metric)
        distance_matrix = squareform(distances)

        mds = make_mds(n_components=n_dimensions, metric=False, precomputed=True,
                       n_init=n_init, max_iter=max_iterations,
                       random_state=random_state)

        site_scores = mds.fit_transform(distance_matrix)

        return {
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

    def ca_analysis(self, data: pd.DataFrame,
                    scaling: int = 1) -> Dict[str, Any]:
        """
        Correspondence Analysis (CA).

        Alias for :meth:`correspondence_analysis`; prefer this abbreviated form
        in new code.

        Parameters:
        -----------
        data : pd.DataFrame
            Species abundance matrix (sites x species)
        scaling : int
            1 = site (row) principal coordinates, 2 = species (column) principal
            coordinates.

        Returns:
        --------
        dict
            CA results including scores, eigenvalues and diagnostics
        """
        return self.correspondence_analysis(data, scaling)

    def correspondence_analysis(self, data: pd.DataFrame, scaling: int = 1) -> Dict[str, Any]:
        """
        Correspondence Analysis of a contingency-like species matrix.

        Implements the standard SVD formulation (Legendre & Legendre 2012,
        section 9.4): with ``P = X / grand_total``, row masses ``r`` and column
        masses ``c``, the matrix
        ``S = D_r^{-1/2} (P - r c') D_c^{-1/2}`` is decomposed; eigenvalues are
        the squared singular values and sum to the total inertia
        (chi-square / N), so each is bounded by 1.
        """
        if (data.values < 0).any():
            raise ValueError("Correspondence analysis requires non-negative data")

        data_clean = data.loc[(data.sum(axis=1) > 0), (data.sum(axis=0) > 0)]

        if data_clean.empty:
            raise ValueError("Data matrix is empty after removing zero rows/columns")

        if data_clean.shape != data.shape:
            warnings.warn(
                f"Dropped {data.shape[0] - data_clean.shape[0]} empty site(s) and "
                f"{data.shape[1] - data_clean.shape[1]} absent species before CA."
            )

        X = data_clean.values.astype(float)
        grand_total = X.sum()

        P = X / grand_total
        r = P.sum(axis=1)          # row masses
        c = P.sum(axis=0)          # column masses

        expected = np.outer(r, c)
        S = (P - expected) / np.sqrt(expected)
        S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)

        U, s, Vt = svd(S, full_matrices=False)
        V = Vt.T

        eigenvalues = s ** 2
        total_inertia = float(eigenvalues.sum())

        # Drop trivial (numerically zero) axes.
        keep = eigenvalues > 1e-12
        U, s, V = U[:, keep], s[keep], V[:, keep]
        eigenvalues = eigenvalues[keep]

        explained_variance = (eigenvalues / total_inertia
                              if total_inertia > 0 else eigenvalues)

        # Standard coordinates.
        row_standard = U / np.sqrt(r)[:, None]
        col_standard = V / np.sqrt(c)[:, None]

        if scaling == 1:
            site_scores = row_standard * s           # row principal coordinates
            species_scores = col_standard            # species standard coordinates
        elif scaling == 2:
            site_scores = row_standard               # site standard coordinates
            species_scores = col_standard * s        # column principal coordinates
        else:
            raise ValueError("scaling must be 1 (sites) or 2 (species)")

        n_axes = site_scores.shape[1]

        return {
            'site_scores': pd.DataFrame(
                site_scores, index=data_clean.index,
                columns=[f'CA{i+1}' for i in range(n_axes)]
            ),
            'species_scores': pd.DataFrame(
                species_scores, index=data_clean.columns,
                columns=[f'CA{i+1}' for i in range(n_axes)]
            ),
            'row_masses': pd.Series(r, index=data_clean.index),
            'column_masses': pd.Series(c, index=data_clean.columns),
            'eigenvalues': eigenvalues,
            'explained_variance_ratio': explained_variance,
            'total_inertia': total_inertia,
            'chi_square': total_inertia * grand_total,
            'scaling': scaling,
            'method': 'CA'
        }

    def dca_analysis(self, data: pd.DataFrame,
                     segments: int = 26,
                     n_axes: int = 4) -> Dict[str, Any]:
        """
        Detrended Correspondence Analysis (DCA).

        Parameters:
        -----------
        data : pd.DataFrame
            Species abundance matrix
        segments : int
            Number of segments used for detrending-by-segments
        n_axes : int
            Number of axes to retain

        Returns:
        --------
        dict
            DCA results, including ``gradient_lengths`` in SD units.

        Notes
        -----
        This is a simplified DCA: it applies detrending-by-segments of each
        axis against the first, then rescales axes so that the mean
        within-segment standard deviation of site scores is 1, which is what
        makes the resulting gradient lengths interpretable in "SD" units. It is
        not a bit-for-bit reimplementation of Hill's DECORANA, so gradient
        lengths should be read as indicative (the usual >4 SD => unimodal,
        <3 SD => linear rule of thumb still applies).
        """
        ca_results = self.ca_analysis(data)

        site_index = ca_results['site_scores'].index
        species_index = ca_results['species_scores'].index
        abundances = data.loc[site_index, species_index].values.astype(float)

        site_scores = ca_results['site_scores'].values.copy()

        n_axes = min(n_axes, site_scores.shape[1])
        site_scores = site_scores[:, :n_axes]

        axis1 = site_scores[:, 0].copy()

        # Detrend every subsequent axis against axis 1 by segments.
        for axis in range(1, site_scores.shape[1]):
            site_scores[:, axis] = self._detrend_by_segments(
                axis1, site_scores[:, axis], segments
            )

        # Rescale each axis so that one unit equals one standard deviation of
        # species turnover (Hill & Gauch 1980), then read gradient length off
        # the rescaled range.
        species_scores = np.zeros((abundances.shape[1], site_scores.shape[1]))
        gradient_lengths = []

        eigenvalues = ca_results['eigenvalues']
        for axis in range(site_scores.shape[1]):
            scores = site_scores[:, axis]
            eigenvalue = float(eigenvalues[axis]) if axis < len(eigenvalues) else 0.0
            sd_unit, species_axis = self._turnover_sd_unit(abundances, scores, eigenvalue)
            if sd_unit > 0:
                scores = scores / sd_unit
                species_axis = species_axis / sd_unit
            site_scores[:, axis] = scores
            species_scores[:, axis] = species_axis
            gradient_lengths.append(float(scores.max() - scores.min()))

        return {
            'site_scores': pd.DataFrame(
                site_scores,
                index=site_index,
                columns=[f'DCA{i+1}' for i in range(site_scores.shape[1])]
            ),
            'species_scores': pd.DataFrame(
                species_scores,
                index=species_index,
                columns=[f'DCA{i+1}' for i in range(species_scores.shape[1])]
            ),
            'eigenvalues': ca_results['eigenvalues'][:n_axes],
            'explained_variance_ratio': ca_results['explained_variance_ratio'][:n_axes],
            'gradient_lengths': np.array(gradient_lengths),
            'segments': segments,
            'method': 'DCA'
        }

    @staticmethod
    def _turnover_sd_unit(abundances: np.ndarray, site_scores: np.ndarray,
                          eigenvalue: float = 1.0):
        """
        One "SD of species turnover" for a DCA axis.

        Species positions are the abundance-weighted averages (WA) of the site
        scores, de-shrunk by ``sqrt(eigenvalue)``. That correction matters: a
        raw WA systematically pulls species toward the centroid, and without
        undoing it the rescaling unit comes out too small and gradient lengths
        are inflated. The unit itself is the root-mean-square within-site
        spread of the species positions, so after dividing by it a distance of
        1 along the axis is roughly one standard deviation of species turnover
        - the units in which DCA gradient lengths are conventionally read.

        Returns ``(sd_unit, species_scores_on_this_axis)``.
        """
        species_totals = abundances.sum(axis=0)
        safe_species_totals = np.where(species_totals > 0, species_totals, 1.0)
        species_scores = (abundances * site_scores[:, None]).sum(axis=0) / safe_species_totals
        species_scores = np.where(species_totals > 0, species_scores, 0.0)

        # Undo weighted-averaging shrinkage so species and sites share a scale.
        if eigenvalue > 1e-12:
            species_scores = species_scores / np.sqrt(eigenvalue)

        site_totals = abundances.sum(axis=1)
        variances = []
        weights = []
        for i in range(abundances.shape[0]):
            total = site_totals[i]
            if total <= 0:
                continue
            weights_i = abundances[i] / total
            mean_i = float(np.sum(weights_i * species_scores))
            var_i = float(np.sum(weights_i * (species_scores - mean_i) ** 2))
            variances.append(var_i)
            weights.append(total)

        if not variances:
            return 0.0, species_scores

        mean_variance = float(np.average(variances, weights=weights))
        return float(np.sqrt(mean_variance)), species_scores

    @staticmethod
    def _segment_assignment(axis1: np.ndarray, segments: int) -> np.ndarray:
        """Assign each site to one of ``segments`` equal-width bins on axis 1."""
        lo, hi = float(axis1.min()), float(axis1.max())
        if hi <= lo:
            return np.zeros(axis1.shape[0], dtype=int)
        width = (hi - lo) / segments
        bins = np.floor((axis1 - lo) / width).astype(int)
        return np.clip(bins, 0, segments - 1)

    def _detrend_by_segments(self, axis1: np.ndarray, axis_k: np.ndarray,
                             segments: int) -> np.ndarray:
        """Subtract the within-segment mean of ``axis_k`` along ``axis1``."""
        assignment = self._segment_assignment(axis1, segments)
        detrended = axis_k.copy()
        for seg in np.unique(assignment):
            mask = assignment == seg
            if mask.sum() >= 1:
                detrended[mask] -= axis_k[mask].mean()
        return detrended

    def _mean_within_segment_sd(self, scores: np.ndarray, segments: int) -> float:
        """Mean within-segment SD of the scores, used as the DCA rescaling unit."""
        assignment = self._segment_assignment(scores, segments)
        sds = []
        for seg in np.unique(assignment):
            block = scores[assignment == seg]
            if block.size >= 2:
                sds.append(float(block.std(ddof=1)))
        if not sds:
            overall = float(scores.std(ddof=1)) if scores.size > 1 else 0.0
            return overall
        mean_sd = float(np.mean([sd for sd in sds if sd > 0])) if any(sd > 0 for sd in sds) else 0.0
        return mean_sd

    def detrended_correspondence_analysis(self, data: pd.DataFrame, segments: int = 26) -> Dict[str, Any]:
        """
        Detrended Correspondence Analysis (DCA) - alias for :meth:`dca_analysis`.

        Kept for backward compatibility; use ``dca_analysis()`` in new code.
        """
        return self.dca_analysis(data, segments)

    def pcoa_analysis(self, data: pd.DataFrame,
                      distance_metric: str = 'bray_curtis',
                      correction: Optional[str] = None) -> Dict[str, Any]:
        """
        Principal Coordinates Analysis (PCoA / classical MDS).

        Parameters:
        -----------
        data : pd.DataFrame
            Species abundance matrix
        distance_metric : str
            Distance metric to use
        correction : {'lingoes', 'cailliez', None}
            Optional correction for negative eigenvalues produced by
            non-Euclidean distances.

        Returns:
        --------
        dict
            PCoA results. ``negative_eigenvalue_fraction`` reports how much of
            the total absolute eigenvalue mass sits on negative axes - a large
            value means the chosen distance is far from Euclidean and the
            2-D picture understates the real structure.
        """
        distances = self._condensed(data.values, distance_metric)
        distance_matrix = squareform(distances)

        eigenvalues, eigenvectors = self._principal_coordinates(distance_matrix)

        if correction in ('lingoes', 'cailliez') and (eigenvalues < -1e-10).any():
            distance_matrix = self._correct_distances(distance_matrix, eigenvalues, correction)
            eigenvalues, eigenvectors = self._principal_coordinates(distance_matrix)

        negative_mass = float(np.sum(np.abs(eigenvalues[eigenvalues < 0])))
        total_abs = float(np.sum(np.abs(eigenvalues)))

        positive = eigenvalues > 1e-10
        pos_eigenvalues = eigenvalues[positive]
        coordinates = eigenvectors[:, positive] * np.sqrt(pos_eigenvalues)

        return {
            'coordinates': pd.DataFrame(
                coordinates,
                index=data.index,
                columns=[f'PCo{i+1}' for i in range(coordinates.shape[1])]
            ),
            # Alias so PCoA results work with the shared plotting helpers.
            'site_scores': pd.DataFrame(
                coordinates,
                index=data.index,
                columns=[f'PCo{i+1}' for i in range(coordinates.shape[1])]
            ),
            'eigenvalues': pos_eigenvalues,
            'all_eigenvalues': eigenvalues,
            'explained_variance_ratio': (pos_eigenvalues / pos_eigenvalues.sum()
                                         if pos_eigenvalues.sum() > 0 else pos_eigenvalues),
            'negative_eigenvalue_fraction': negative_mass / total_abs if total_abs > 0 else 0.0,
            'distance_matrix': distance_matrix,
            'distance_metric': distance_metric,
            'correction': correction,
            'method': 'PCoA'
        }

    @staticmethod
    def _principal_coordinates(distance_matrix: np.ndarray):
        """Gower-centre a distance matrix and return sorted eigen-decomposition."""
        n = distance_matrix.shape[0]
        centering = np.eye(n) - np.ones((n, n)) / n
        gower = -0.5 * centering @ (distance_matrix ** 2) @ centering
        gower = (gower + gower.T) / 2  # enforce exact symmetry

        # eigh is appropriate (and faster/stabler) for a symmetric matrix.
        eigenvalues, eigenvectors = np.linalg.eigh(gower)
        order = np.argsort(eigenvalues)[::-1]
        return eigenvalues[order], eigenvectors[:, order]

    @staticmethod
    def _correct_distances(distance_matrix: np.ndarray, eigenvalues: np.ndarray,
                           correction: str) -> np.ndarray:
        """Apply the Lingoes or Cailliez correction for negative eigenvalues."""
        c = float(abs(eigenvalues.min()))
        corrected = distance_matrix.copy()
        off_diagonal = ~np.eye(distance_matrix.shape[0], dtype=bool)

        if correction == 'lingoes':
            corrected[off_diagonal] = np.sqrt(distance_matrix[off_diagonal] ** 2 + 2 * c)
        else:  # cailliez
            corrected[off_diagonal] = distance_matrix[off_diagonal] + c
        return corrected

    def principal_coordinates_analysis(self, data: pd.DataFrame,
                                       distance_metric: str = 'bray_curtis') -> Dict[str, Any]:
        """
        Principal Coordinates Analysis (PCoA) - alias for :meth:`pcoa_analysis`.

        Kept for backward compatibility; use ``pcoa_analysis()`` in new code.
        """
        return self.pcoa_analysis(data, distance_metric)

    # ------------------------------------------------------------------
    # Constrained ordination
    # ------------------------------------------------------------------

    def canonical_correspondence_analysis(self, species_data: pd.DataFrame,
                                          env_data: pd.DataFrame,
                                          scaling: int = 1) -> Dict[str, Any]:
        """
        Canonical Correspondence Analysis (CCA).

        Implements the weighted-regression formulation (ter Braak 1986;
        Legendre & Legendre 2012, section 11.2):

        1. Build the chi-square standardised matrix
           ``S = D_r^{-1/2} (P - r c') D_c^{-1/2}``.
        2. Weighted-regress ``S`` on the (row-weighted, centred, standardised)
           environmental variables to obtain the fitted table ``S_hat``.
        3. SVD of ``S_hat`` gives the constrained axes; the eigenvalues sum to
           the constrained inertia.

        Parameters:
        -----------
        species_data : pd.DataFrame
            Species abundance matrix (sites x species), non-negative
        env_data : pd.DataFrame
            Environmental variables matrix
        scaling : int
            1 = site principal coordinates, 2 = species principal coordinates

        Returns:
        --------
        dict
            CCA results including LC/WA site scores, species scores,
            environmental biplot scores and the constrained inertia fraction.
        """
        species_complete, env_complete = self._align_constrained(species_data, env_data)

        if (species_complete.values < 0).any():
            raise ValueError("CCA requires non-negative species data")

        X = species_complete.values.astype(float)
        grand_total = X.sum()
        if grand_total <= 0:
            raise ValueError("Species matrix sums to zero")

        P = X / grand_total
        r = P.sum(axis=1)
        c = P.sum(axis=0)

        if np.any(r <= 0) or np.any(c <= 0):
            keep_rows = r > 0
            keep_cols = c > 0
            species_complete = species_complete.loc[keep_rows, keep_cols]
            env_complete = env_complete.loc[keep_rows]
            return self.canonical_correspondence_analysis(
                species_complete, env_complete, scaling
            )

        expected = np.outer(r, c)
        S = (P - expected) / np.sqrt(expected)
        S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
        total_inertia = float(np.sum(S ** 2))

        # Row-weighted centring and standardisation of the constraints.
        Z = env_complete.values.astype(float)
        weighted_mean = (r[:, None] * Z).sum(axis=0)
        Zc = Z - weighted_mean
        weighted_sd = np.sqrt((r[:, None] * Zc ** 2).sum(axis=0))
        weighted_sd[weighted_sd == 0] = 1.0
        Zc = Zc / weighted_sd

        # Weighted least squares projection: sqrt(r) absorbs the row weights.
        Zw = np.sqrt(r)[:, None] * Zc
        hat = Zw @ np.linalg.pinv(Zw.T @ Zw) @ Zw.T
        S_hat = hat @ S

        constrained_inertia = float(np.sum(S_hat ** 2))

        U, s, Vt = svd(S_hat, full_matrices=False)
        V = Vt.T

        eigenvalues = s ** 2
        keep = eigenvalues > 1e-12
        # A CCA can have at most as many axes as constraints.
        keep[env_complete.shape[1]:] = False
        U, s, V = U[:, keep], s[keep], V[:, keep]
        eigenvalues = eigenvalues[keep]

        row_standard = U / np.sqrt(r)[:, None]
        col_standard = V / np.sqrt(c)[:, None]

        if scaling == 1:
            site_scores = row_standard * s
            species_scores = col_standard
        elif scaling == 2:
            site_scores = row_standard
            species_scores = col_standard * s
        else:
            raise ValueError("scaling must be 1 (sites) or 2 (species)")

        # Site scores that are linear combinations of the constraints (LC scores)
        # versus weighted averages of the species scores (WA scores).
        lc_scores = site_scores
        with np.errstate(invalid='ignore', divide='ignore'):
            wa_scores = (P / r[:, None]) @ (col_standard)
        wa_scores = np.nan_to_num(wa_scores, nan=0.0)

        # Environmental biplot scores: weighted correlation of each constraint
        # with each constrained axis.
        env_scores = Zc.T @ (r[:, None] * lc_scores)
        axis_norm = np.sqrt((r[:, None] * lc_scores ** 2).sum(axis=0))
        axis_norm[axis_norm == 0] = 1.0
        env_scores = env_scores / axis_norm

        n_axes = eigenvalues.size
        columns = [f'CCA{i+1}' for i in range(n_axes)]

        return {
            'site_scores': pd.DataFrame(lc_scores, index=species_complete.index, columns=columns),
            'lc_scores': pd.DataFrame(lc_scores, index=species_complete.index, columns=columns),
            'wa_scores': pd.DataFrame(wa_scores[:, :n_axes], index=species_complete.index,
                                      columns=columns),
            'species_scores': pd.DataFrame(species_scores, index=species_complete.columns,
                                           columns=columns),
            'env_scores': pd.DataFrame(env_scores, index=env_complete.columns, columns=columns),
            'eigenvalues': eigenvalues,
            'explained_variance_ratio': (eigenvalues / constrained_inertia
                                         if constrained_inertia > 0 else eigenvalues),
            'total_inertia': total_inertia,
            'constrained_inertia': constrained_inertia,
            'unconstrained_inertia': total_inertia - constrained_inertia,
            'proportion_constrained': (constrained_inertia / total_inertia
                                       if total_inertia > 0 else 0.0),
            'scaling': scaling,
            'method': 'CCA'
        }

    def cca_analysis(self, species_data: pd.DataFrame, env_data: pd.DataFrame,
                     scaling: int = 1) -> Dict[str, Any]:
        """
        Canonical Correspondence Analysis (CCA).

        Alias for :meth:`canonical_correspondence_analysis`.
        """
        return self.canonical_correspondence_analysis(species_data, env_data, scaling)

    def rda_analysis(self, species_data: pd.DataFrame,
                     env_data: pd.DataFrame,
                     scale_species: bool = False) -> Dict[str, Any]:
        """
        Redundancy Analysis (RDA).

        Species data are centred (and optionally standardised, giving a
        correlation RDA); the environmental variables are standardised. The
        response table is regressed on the constraints and the fitted values
        are decomposed by SVD.

        Parameters:
        -----------
        species_data : pd.DataFrame
            Species abundance matrix
        env_data : pd.DataFrame
            Environmental variables matrix
        scale_species : bool
            Standardise species to unit variance (correlation RDA). Default
            ``False`` centres only, which is the usual choice for community
            data that has already been transformed.

        Returns:
        --------
        dict
            RDA results including constrained/unconstrained eigenvalues,
            LC and WA site scores, species scores and environmental biplot
            scores.
        """
        species_complete, env_complete = self._align_constrained(species_data, env_data)

        n = species_complete.shape[0]
        if n < 3:
            raise ValueError("RDA requires at least three complete sites")

        Y = species_complete.values.astype(float)
        Y = Y - Y.mean(axis=0)
        if scale_species:
            sd = Y.std(axis=0, ddof=1)
            sd[sd == 0] = 1.0
            Y = Y / sd

        X = StandardScaler().fit_transform(env_complete.values.astype(float))

        total_variance = float(np.sum(Y ** 2) / (n - 1))

        # Fitted (constrained) and residual (unconstrained) components.
        coefficients = np.linalg.pinv(X.T @ X) @ X.T @ Y
        Y_fitted = X @ coefficients
        Y_residual = Y - Y_fitted

        constrained_variance = float(np.sum(Y_fitted ** 2) / (n - 1))
        unconstrained_variance = float(np.sum(Y_residual ** 2) / (n - 1))

        U, s, Vt = svd(Y_fitted, full_matrices=False)
        V = Vt.T

        eigenvalues = (s ** 2) / (n - 1)
        keep = eigenvalues > 1e-12
        keep[env_complete.shape[1]:] = False  # at most one axis per constraint
        U, s, V = U[:, keep], s[keep], V[:, keep]
        eigenvalues = eigenvalues[keep]

        lc_scores = U * s                 # fitted site scores
        wa_scores = Y @ V                 # weighted-average site scores
        species_scores = V                # species loadings

        # Environmental biplot scores = correlation of each constraint with each axis.
        env_scores = np.zeros((X.shape[1], lc_scores.shape[1]))
        for j in range(X.shape[1]):
            for k in range(lc_scores.shape[1]):
                if np.std(lc_scores[:, k]) > 0:
                    env_scores[j, k] = np.corrcoef(X[:, j], lc_scores[:, k])[0, 1]

        # Residual (unconstrained) eigenvalues from a PCA of the residuals.
        residual_eigenvalues = np.array([])
        if unconstrained_variance > 0:
            _, s_res, _ = svd(Y_residual, full_matrices=False)
            residual_eigenvalues = (s_res ** 2) / (n - 1)
            residual_eigenvalues = residual_eigenvalues[residual_eigenvalues > 1e-12]

        n_axes = eigenvalues.size
        columns = [f'RDA{i+1}' for i in range(n_axes)]

        return {
            'site_scores': pd.DataFrame(lc_scores, index=species_complete.index, columns=columns),
            'lc_scores': pd.DataFrame(lc_scores, index=species_complete.index, columns=columns),
            'wa_scores': pd.DataFrame(wa_scores[:, :n_axes], index=species_complete.index,
                                      columns=columns),
            'species_scores': pd.DataFrame(species_scores, index=species_complete.columns,
                                           columns=columns),
            'env_scores': pd.DataFrame(env_scores, index=env_complete.columns, columns=columns),
            'coefficients': pd.DataFrame(coefficients, index=env_complete.columns,
                                         columns=species_complete.columns),
            'eigenvalues': eigenvalues,
            'residual_eigenvalues': residual_eigenvalues,
            # Fraction of the *constrained* variance on each axis.
            'explained_variance_ratio': (eigenvalues / constrained_variance
                                         if constrained_variance > 0 else eigenvalues),
            # Fraction of *total* variance on each axis.
            'total_variance_ratio': (eigenvalues / total_variance
                                     if total_variance > 0 else eigenvalues),
            'total_variance': total_variance,
            'constrained_variance': constrained_variance,
            'unconstrained_variance': unconstrained_variance,
            'proportion_constrained': (constrained_variance / total_variance
                                       if total_variance > 0 else 0.0),
            'adjusted_r_squared': self._adjusted_r2(
                constrained_variance / total_variance if total_variance > 0 else 0.0,
                n, env_complete.shape[1]
            ),
            'method': 'RDA'
        }

    @staticmethod
    def _adjusted_r2(r_squared: float, n: int, n_predictors: int) -> float:
        """Ezekiel-adjusted R-squared, the standard bias correction for RDA."""
        denom = n - n_predictors - 1
        if denom <= 0:
            return float('nan')
        return float(1 - (1 - r_squared) * (n - 1) / denom)

    @staticmethod
    def _align_constrained(species_data: pd.DataFrame,
                           env_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Align species and environmental tables on shared, complete sites."""
        common_index = species_data.index.intersection(env_data.index)
        if len(common_index) == 0:
            raise ValueError(
                "Species and environmental data share no site identifiers"
            )

        species_aligned = species_data.loc[common_index]
        env_aligned = env_data.loc[common_index]

        numeric_env = env_aligned.select_dtypes(include=[np.number])
        if numeric_env.shape[1] == 0:
            raise ValueError("No numeric environmental variables available")
        if numeric_env.shape[1] < env_aligned.shape[1]:
            dropped = set(env_aligned.columns) - set(numeric_env.columns)
            warnings.warn(f"Ignoring non-numeric environmental columns: {sorted(dropped)}")

        complete_cases = ~numeric_env.isnull().any(axis=1)
        species_complete = species_aligned.loc[complete_cases]
        env_complete = numeric_env.loc[complete_cases]

        if len(species_complete) == 0:
            raise ValueError("No complete cases found")

        return species_complete, env_complete

    def redundancy_analysis(self, species_data: pd.DataFrame, env_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Redundancy Analysis (RDA) - alias for :meth:`rda_analysis`.

        Kept for backward compatibility; use ``rda_analysis()`` in new code.
        """
        return self.rda_analysis(species_data, env_data)


    # ------------------------------------------------------------------
    # Significance tests for constrained ordination
    # ------------------------------------------------------------------

    def anova_rda(self, species_data: pd.DataFrame,
                  env_data: pd.DataFrame,
                  by: Optional[str] = None,
                  conditioning: Optional[pd.DataFrame] = None,
                  permutations: int = 999,
                  scale_species: bool = False,
                  random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        Permutation significance test for a redundancy analysis.

        The analogue of vegan's ``anova.cca`` for RDA. Without this, a
        constrained ordination can be fitted but not tested - and a constrained
        ordination that explains a large fraction of variance is unremarkable if
        the number of constraints is large relative to the number of sites.

        Parameters:
        -----------
        species_data : pd.DataFrame
            Response matrix (sites x species).
        env_data : pd.DataFrame
            Constraining variables.
        by : {None, 'axis', 'terms', 'margin'}
            ``None`` tests the whole constrained model. ``'axis'`` tests each
            canonical axis in turn (conditioning on the preceding axes),
            ``'terms'`` tests each constraint sequentially, ``'margin'`` tests
            each constraint with all others present.
        conditioning : pd.DataFrame, optional
            Covariates to partial out first (partial RDA). Their variance is
            removed before the constraints are tested.
        permutations : int
            Number of permutations.
        scale_species : bool
            Standardise species to unit variance (correlation RDA).
        random_state : int, optional
            Seed for reproducible permutations.

        Returns:
        --------
        dict
            ``table`` with Df, Variance, F and Pr(>F) per tested component.

        Notes
        -----
        With ``conditioning`` the permutation is of the residuals after the
        covariates have been removed (a reduced-model permutation), which is
        the appropriate scheme for a partial ordination.
        """
        return self._anova_constrained(
            species_data, env_data, method='rda', by=by,
            conditioning=conditioning, permutations=permutations,
            scale_species=scale_species, random_state=random_state)

    def anova_cca(self, species_data: pd.DataFrame,
                  env_data: pd.DataFrame,
                  by: Optional[str] = None,
                  conditioning: Optional[pd.DataFrame] = None,
                  permutations: int = 999,
                  random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        Permutation significance test for a canonical correspondence analysis.

        See :meth:`anova_rda` for the parameters; the only difference is that
        the response is the chi-square standardised species table and the
        regression is row-weighted, matching :meth:`cca_analysis`.
        """
        return self._anova_constrained(
            species_data, env_data, method='cca', by=by,
            conditioning=conditioning, permutations=permutations,
            random_state=random_state)

    def _anova_constrained(self, species_data: pd.DataFrame,
                           env_data: pd.DataFrame,
                           method: str,
                           by: Optional[str],
                           conditioning: Optional[pd.DataFrame],
                           permutations: int,
                           random_state: Optional[int],
                           scale_species: bool = False) -> Dict[str, Any]:
        """Shared permutation machinery for RDA and CCA significance tests."""
        if by not in (None, 'axis', 'terms', 'margin'):
            raise ValueError("by must be None, 'axis', 'terms' or 'margin'")

        species_complete, env_complete = self._align_constrained(species_data, env_data)

        condition_complete = None
        if conditioning is not None:
            index = species_complete.index.intersection(conditioning.index)
            species_complete = species_complete.loc[index]
            env_complete = env_complete.loc[index]
            condition_complete = conditioning.loc[index].select_dtypes(include=[np.number])
            keep = ~condition_complete.isnull().any(axis=1)
            species_complete = species_complete.loc[keep]
            env_complete = env_complete.loc[keep]
            condition_complete = condition_complete.loc[keep]

        response, weights = self._constrained_response(
            species_complete, method, scale_species)

        X = self._standardise(env_complete.values.astype(float), weights)
        Z = (self._standardise(condition_complete.values.astype(float), weights)
             if condition_complete is not None else None)

        rng = as_generator(random_state)
        n = response.shape[0]

        if by is None:
            rows = [self._anova_component(
                response, X, Z, 'Model', permutations, rng, n)]
        elif by == 'terms':
            rows = []
            for i, name in enumerate(env_complete.columns):
                previous = X[:, :i] if i > 0 else None
                condition = self._hstack(Z, previous)
                rows.append(self._anova_component(
                    response, X[:, i:i + 1], condition, name,
                    permutations, rng, n, full_constraints=X))
        elif by == 'margin':
            rows = []
            for i, name in enumerate(env_complete.columns):
                others = np.delete(X, i, axis=1)
                condition = self._hstack(Z, others if others.shape[1] else None)
                rows.append(self._anova_component(
                    response, X[:, i:i + 1], condition, name,
                    permutations, rng, n, full_constraints=X))
        else:  # by == 'axis'
            axes = self._canonical_axes(response, X, Z)
            rows = []
            for k in range(axes.shape[1]):
                condition = self._hstack(Z, axes[:, :k] if k else None)
                rows.append(self._anova_component(
                    response, axes[:, k:k + 1], condition, f'Axis{k + 1}',
                    permutations, rng, n, full_constraints=X))

        residual = self._partition(response, X, Z)
        rows.append({'term': 'Residual', 'Df': residual['df_residual'],
                     'Variance': residual['residual'], 'F': np.nan, 'Pr(>F)': np.nan})

        table = pd.DataFrame(rows).set_index('term')

        return {
            'table': table,
            'method': f'anova.{method}',
            'by': by or 'model',
            'permutations': permutations,
            'total_variance': residual['total'],
            'constrained_variance': residual['constrained'],
            'conditioned_variance': residual['conditioned'],
            'proportion_constrained': (residual['constrained'] / residual['total']
                                       if residual['total'] > 0 else 0.0),
        }

    @staticmethod
    def _hstack(first: Optional[np.ndarray],
                second: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Column-stack two optional blocks, returning None when both are absent."""
        blocks = [b for b in (first, second) if b is not None and b.shape[1] > 0]
        if not blocks:
            return None
        return np.column_stack(blocks)

    def _constrained_response(self, species: pd.DataFrame, method: str,
                              scale_species: bool):
        """
        Response matrix whose total sum of squares is the ordination's inertia.

        For RDA this is the centred (optionally standardised) species table. For
        CCA it is the chi-square standardised matrix, together with the row
        weights used by the weighted regression.
        """
        values = species.values.astype(float)

        if method == 'rda':
            centred = values - values.mean(axis=0)
            if scale_species:
                sd = centred.std(axis=0, ddof=1)
                sd[sd == 0] = 1.0
                centred = centred / sd
            return centred, None

        if (values < 0).any():
            raise ValueError("CCA requires non-negative species data")

        grand_total = values.sum()
        if grand_total <= 0:
            raise ValueError("Species matrix sums to zero")

        P = values / grand_total
        r = P.sum(axis=1)
        c = P.sum(axis=0)
        if np.any(r <= 0) or np.any(c <= 0):
            raise ValueError("CCA requires every site and species to be present")

        expected = np.outer(r, c)
        S = (P - expected) / np.sqrt(expected)
        return np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0), r

    @staticmethod
    def _standardise(matrix: np.ndarray, weights: Optional[np.ndarray]) -> np.ndarray:
        """Centre and scale predictors, using row weights for CCA."""
        if weights is None:
            centred = matrix - matrix.mean(axis=0)
        else:
            centred = matrix - (weights[:, None] * matrix).sum(axis=0)
            centred = np.sqrt(weights)[:, None] * centred

        sd = centred.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        return centred / sd

    @staticmethod
    def _project_out(response: np.ndarray, covariates: Optional[np.ndarray]) -> np.ndarray:
        """Residuals of ``response`` after regressing on ``covariates``."""
        if covariates is None or covariates.shape[1] == 0:
            return response
        hat = covariates @ np.linalg.pinv(covariates.T @ covariates) @ covariates.T
        return response - hat @ response

    def _partition(self, response: np.ndarray, X: np.ndarray,
                   Z: Optional[np.ndarray]) -> Dict[str, float]:
        """Split total inertia into conditioned, constrained and residual parts."""
        n = response.shape[0]
        total = float(np.sum(response ** 2))

        if Z is not None and Z.shape[1] > 0:
            residual_z = self._project_out(response, Z)
            conditioned = total - float(np.sum(residual_z ** 2))
            df_condition = int(np.linalg.matrix_rank(Z))
        else:
            residual_z = response
            conditioned = 0.0
            df_condition = 0

        X_adjusted = self._project_out(X, Z) if Z is not None else X
        fitted = X_adjusted @ np.linalg.pinv(X_adjusted.T @ X_adjusted) @ X_adjusted.T @ residual_z
        constrained = float(np.sum(fitted ** 2))
        df_constrained = int(np.linalg.matrix_rank(X_adjusted))

        residual = total - conditioned - constrained
        df_residual = n - 1 - df_condition - df_constrained

        return {'total': total, 'conditioned': conditioned,
                'constrained': constrained, 'residual': max(residual, 0.0),
                'df_condition': df_condition, 'df_constrained': df_constrained,
                'df_residual': max(df_residual, 0)}

    def _pseudo_f(self, response: np.ndarray, X: np.ndarray,
                  Z: Optional[np.ndarray], df_residual_full: int,
                  residual_full: float) -> float:
        """F for one constraint block against the full model's residual."""
        X_adjusted = self._project_out(X, Z) if Z is not None else X
        if X_adjusted.shape[1] == 0:
            return 0.0

        target = self._project_out(response, Z) if Z is not None else response
        hat = X_adjusted @ np.linalg.pinv(X_adjusted.T @ X_adjusted) @ X_adjusted.T
        explained = float(np.sum((hat @ target) ** 2))
        df = int(np.linalg.matrix_rank(X_adjusted))

        if df == 0 or df_residual_full <= 0 or residual_full <= 0:
            return 0.0
        return (explained / df) / (residual_full / df_residual_full)

    def _anova_component(self, response: np.ndarray, X: np.ndarray,
                         Z: Optional[np.ndarray], name: str,
                         permutations: int, rng, n: int,
                         full_constraints: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Test one constraint block by permutation."""
        all_constraints = full_constraints if full_constraints is not None else X
        full = self._partition(response, all_constraints, Z if full_constraints is None else None)
        residual_full = full['residual']
        df_residual_full = full['df_residual']

        X_adjusted = self._project_out(X, Z) if Z is not None else X
        target = self._project_out(response, Z) if Z is not None else response

        hat = (X_adjusted @ np.linalg.pinv(X_adjusted.T @ X_adjusted) @ X_adjusted.T
               if X_adjusted.shape[1] else np.zeros((n, n)))
        variance = float(np.sum((hat @ target) ** 2))
        df = int(np.linalg.matrix_rank(X_adjusted)) if X_adjusted.shape[1] else 0

        observed_f = ((variance / df) / (residual_full / df_residual_full)
                      if df > 0 and df_residual_full > 0 and residual_full > 0 else 0.0)

        # Permute the (conditioned) response rows: under the null the
        # constraints carry no information about it.
        permuted_f = np.empty(permutations, dtype=float)
        for i in range(permutations):
            order = rng.permutation(n)
            permuted_target = target[order]
            explained = float(np.sum((hat @ permuted_target) ** 2))
            resid = float(np.sum(permuted_target ** 2)) - explained
            permuted_f[i] = ((explained / df) / (resid / df_residual_full)
                             if df > 0 and df_residual_full > 0 and resid > 0 else 0.0)

        p_value = float((np.sum(permuted_f >= observed_f) + 1) / (permutations + 1))

        return {'term': name, 'Df': df, 'Variance': variance,
                'F': observed_f, 'Pr(>F)': p_value}

    def _canonical_axes(self, response: np.ndarray, X: np.ndarray,
                        Z: Optional[np.ndarray]) -> np.ndarray:
        """Site scores on the canonical axes (used for by-axis testing)."""
        target = self._project_out(response, Z) if Z is not None else response
        X_adjusted = self._project_out(X, Z) if Z is not None else X
        if X_adjusted.shape[1] == 0:
            return np.zeros((response.shape[0], 0))

        fitted = X_adjusted @ np.linalg.pinv(X_adjusted.T @ X_adjusted) @ X_adjusted.T @ target
        U, s, _ = svd(fitted, full_matrices=False)
        keep = s > 1e-10
        keep[X_adjusted.shape[1]:] = False
        return U[:, keep] * s[keep]

    # ------------------------------------------------------------------
    # Variation partitioning
    # ------------------------------------------------------------------

    def varpart(self, species_data: pd.DataFrame,
                *explanatory: pd.DataFrame,
                transform: str = 'hellinger',
                permutations: int = 999,
                random_state: Optional[int] = None,
                table_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Variation partitioning among two or three explanatory tables.

        Decomposes the variance of the response into the fractions uniquely
        explained by each explanatory table, the fractions they share, and the
        unexplained residual - the standard way to ask "how much of the
        community pattern is environmental and how much is spatial?".

        Parameters:
        -----------
        species_data : pd.DataFrame
            Response matrix (sites x species).
        *explanatory : pd.DataFrame
            Two or three tables of explanatory variables.
        transform : str
            Transformation applied to the response before partitioning;
            ``'hellinger'`` is the usual choice for community data.
        permutations : int
            Permutations used to test the testable (unique) fractions. Set to 0
            to skip testing.
        random_state : int, optional
            Seed for reproducible permutations.
        table_names : list of str, optional
            Labels for the explanatory tables (default ``X1``, ``X2``, ...).

        Returns:
        --------
        dict
            ``fractions`` (a DataFrame of adjusted R-squared per fraction),
            ``testable`` (permutation tests of the unique fractions) and the
            individual/combined adjusted R-squared values.

        Notes
        -----
        Fractions are **adjusted** R-squared (Peres-Neto et al. 2006). Using
        unadjusted R-squared inflates every fraction and makes tables with more
        variables look more important than they are. Shared fractions are
        obtained by subtraction and can legitimately be negative, which
        indicates the two tables explain the response better separately than
        their overlap suggests; such fractions cannot be tested.
        """
        if not 2 <= len(explanatory) <= 3:
            raise ValueError("varpart requires two or three explanatory tables")

        names = table_names or [f'X{i + 1}' for i in range(len(explanatory))]
        if len(names) != len(explanatory):
            raise ValueError("table_names must match the number of explanatory tables")

        # Align every table on the sites they share.
        index = species_data.index
        for table in explanatory:
            index = index.intersection(table.index)
        if len(index) == 0:
            raise ValueError("Species and explanatory tables share no sites")

        response = self._transform(species_data.loc[index], transform)
        response = response - response.mean(axis=0)
        n = response.shape[0]

        blocks = []
        for table in explanatory:
            numeric = table.loc[index].select_dtypes(include=[np.number])
            if numeric.shape[1] == 0:
                raise ValueError("Explanatory tables must contain numeric columns")
            blocks.append(self._standardise(numeric.values.astype(float), None))

        def adjusted_r2(selected: List[int]) -> float:
            if not selected:
                return 0.0
            design = np.column_stack([blocks[i] for i in selected])
            fitted = design @ np.linalg.pinv(design.T @ design) @ design.T @ response
            total = float(np.sum(response ** 2))
            if total <= 0:
                return 0.0
            r2 = float(np.sum(fitted ** 2)) / total
            k = int(np.linalg.matrix_rank(design))
            if n - k - 1 <= 0:
                return float('nan')
            return 1 - (1 - r2) * (n - 1) / (n - k - 1)

        if len(blocks) == 2:
            r2_1, r2_2 = adjusted_r2([0]), adjusted_r2([1])
            r2_12 = adjusted_r2([0, 1])
            fractions = {
                f'[a] {names[0]} unique': r2_12 - r2_2,
                f'[b] shared': r2_1 + r2_2 - r2_12,
                f'[c] {names[1]} unique': r2_12 - r2_1,
                '[d] residual': 1 - r2_12,
            }
            individual = {names[0]: r2_1, names[1]: r2_2}
            combined = {'+'.join(names): r2_12}
        else:
            r2_1, r2_2, r2_3 = adjusted_r2([0]), adjusted_r2([1]), adjusted_r2([2])
            r2_12, r2_13, r2_23 = adjusted_r2([0, 1]), adjusted_r2([0, 2]), adjusted_r2([1, 2])
            r2_123 = adjusted_r2([0, 1, 2])
            fractions = {
                f'[a] {names[0]} unique': r2_123 - r2_23,
                f'[b] {names[1]} unique': r2_123 - r2_13,
                f'[c] {names[2]} unique': r2_123 - r2_12,
                f'[d] {names[0]}+{names[1]}': r2_13 + r2_23 - r2_3 - r2_123,
                f'[e] {names[1]}+{names[2]}': r2_12 + r2_13 - r2_1 - r2_123,
                f'[f] {names[0]}+{names[2]}': r2_12 + r2_23 - r2_2 - r2_123,
                '[g] all three': r2_1 + r2_2 + r2_3 - r2_12 - r2_13 - r2_23 + r2_123,
                '[h] residual': 1 - r2_123,
            }
            individual = dict(zip(names, [r2_1, r2_2, r2_3]))
            combined = {'+'.join(names): r2_123}

        fraction_table = pd.DataFrame(
            {'adj_R2': list(fractions.values())}, index=list(fractions))
        fraction_table['testable'] = ['unique' in name or name.startswith('[a]')
                                      for name in fraction_table.index]

        testable = None
        if permutations and permutations > 0:
            records = []
            for i, name in enumerate(names):
                others = [j for j in range(len(blocks)) if j != i]
                condition = np.column_stack([blocks[j] for j in others])
                result = self._anova_component(
                    response, blocks[i], condition, f'{name} | others',
                    permutations, as_generator(random_state), n,
                    full_constraints=np.column_stack(blocks))
                records.append(result)
            testable = pd.DataFrame(records).set_index('term')

        return {
            'fractions': fraction_table,
            'individual_adj_r2': individual,
            'combined_adj_r2': combined,
            'testable': testable,
            'n_sites': n,
            'transform': transform,
            'method': 'varpart',
        }

    def forward_selection(self, species_data: pd.DataFrame,
                          env_data: pd.DataFrame,
                          alpha: float = 0.05,
                          permutations: int = 199,
                          max_variables: Optional[int] = None,
                          transform: str = 'hellinger',
                          random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        Forward selection of explanatory variables for an RDA.

        At each step the variable giving the largest increase in explained
        variance is tested by permutation; selection stops when no remaining
        variable is significant at ``alpha``.

        Parameters:
        -----------
        species_data : pd.DataFrame
            Response matrix.
        env_data : pd.DataFrame
            Candidate explanatory variables.
        alpha : float
            Significance threshold for entry.
        permutations : int
            Permutations per candidate test.
        max_variables : int, optional
            Stop after this many variables.
        transform : str
            Transformation applied to the response.
        random_state : int, optional
            Seed for reproducible permutations.

        Returns:
        --------
        dict
            ``selected`` variable names in entry order and a ``history`` table.
        """
        species_complete, env_complete = self._align_constrained(species_data, env_data)

        response = self._transform(species_complete, transform)
        response = response - response.mean(axis=0)
        n = response.shape[0]

        candidates = list(env_complete.columns)
        columns = {name: self._standardise(
            env_complete[[name]].values.astype(float), None) for name in candidates}

        rng = as_generator(random_state)
        selected: List[str] = []
        history = []
        limit = max_variables or len(candidates)

        while candidates and len(selected) < limit:
            best_name, best_result = None, None
            for name in candidates:
                condition = (np.column_stack([columns[s] for s in selected])
                             if selected else None)
                trial = np.column_stack([columns[s] for s in selected + [name]])
                result = self._anova_component(
                    response, columns[name], condition, name,
                    permutations, rng, n, full_constraints=trial)
                if best_result is None or result['F'] > best_result['F']:
                    best_name, best_result = name, result

            if best_result is None or best_result['Pr(>F)'] > alpha:
                if best_result is not None:
                    history.append({**best_result, 'selected': False})
                break

            selected.append(best_name)
            candidates.remove(best_name)
            history.append({**best_result, 'selected': True})

        return {
            'selected': selected,
            'history': pd.DataFrame(history) if history else pd.DataFrame(),
            'alpha': alpha,
            'method': 'forward_selection',
        }

    # ------------------------------------------------------------------
    # Distance measures
    # ------------------------------------------------------------------

    @staticmethod
    def _condensed(values: np.ndarray, metric: str) -> np.ndarray:
        """Condensed distance vector, honouring VegZ metric aliases."""
        key = str(metric).lower()
        values = np.asarray(values, dtype=float)

        if key == 'chord':
            norms = np.sqrt((values ** 2).sum(axis=1))
            norms[norms == 0] = 1.0
            return pdist(values / norms[:, None], metric='euclidean')
        if key == 'hellinger':
            row_sums = values.sum(axis=1)
            row_sums[row_sums == 0] = 1.0
            return pdist(np.sqrt(values / row_sums[:, None]), metric='euclidean')

        scipy_metric = _PDIST_ALIASES.get(key, key)
        if scipy_metric in _BINARY_METRICS:
            values = (values > 0).astype(bool)

        try:
            return pdist(values, metric=scipy_metric)
        except ValueError as exc:
            raise ValueError(
                f"Unknown distance metric '{metric}'. Supported names: "
                f"{', '.join(sorted(set(_PDIST_ALIASES) | {'chord', 'hellinger'}))} "
                "or any SciPy pdist metric."
            ) from exc

    def _bray_curtis_distance(self, data: np.ndarray) -> np.ndarray:
        """Condensed Bray-Curtis distance vector."""
        return self._condensed(data, 'bray_curtis')

    def _jaccard_distance(self, data: np.ndarray) -> np.ndarray:
        """Condensed Jaccard distance vector (presence/absence)."""
        return self._condensed(data, 'jaccard')

    def _sorensen_distance(self, data: np.ndarray) -> np.ndarray:
        """Condensed Sorensen (Dice) distance vector (presence/absence)."""
        return self._condensed(data, 'sorensen')

    def _euclidean_distance(self, data: np.ndarray) -> np.ndarray:
        """Condensed Euclidean distance vector."""
        return self._condensed(data, 'euclidean')

    def _manhattan_distance(self, data: np.ndarray) -> np.ndarray:
        """Condensed Manhattan (city-block) distance vector."""
        return self._condensed(data, 'manhattan')

    def _canberra_distance(self, data: np.ndarray) -> np.ndarray:
        """Condensed Canberra distance vector."""
        return self._condensed(data, 'canberra')

    def _chord_distance(self, data: np.ndarray) -> np.ndarray:
        """Condensed chord distance vector."""
        return self._condensed(data, 'chord')

    def _hellinger_distance(self, data: np.ndarray) -> np.ndarray:
        """Condensed Hellinger distance vector."""
        return self._condensed(data, 'hellinger')

    @staticmethod
    def _transform(data: pd.DataFrame, method: str) -> np.ndarray:
        """Apply a transformation for ordination and return a NumPy array."""
        method = (method or 'none').lower()
        values = data.values.astype(float)

        if method in ('none', 'raw'):
            return values
        if method == 'hellinger':
            row_sums = values.sum(axis=1)
            row_sums[row_sums == 0] = 1.0
            return np.sqrt(values / row_sums[:, None])
        if method == 'chord':
            norms = np.sqrt((values ** 2).sum(axis=1))
            norms[norms == 0] = 1.0
            return values / norms[:, None]
        if method == 'log':
            return np.log1p(values)
        if method == 'sqrt':
            return np.sqrt(np.maximum(values, 0))
        if method == 'standardize':
            return StandardScaler().fit_transform(values)

        raise ValueError(
            f"Unknown transformation '{method}'. Valid options: none, "
            "hellinger, chord, log, sqrt, standardize."
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def procrustes_analysis(self, data1: Union[np.ndarray, pd.DataFrame],
                            data2: Union[np.ndarray, pd.DataFrame],
                            permutations: int = 999,
                            random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        Procrustes analysis comparing two ordinations.

        Parameters:
        -----------
        data1, data2 : array-like
            Ordination score matrices with matching rows (sites).
        permutations : int
            Number of row permutations used for the PROTEST significance test
            (0 to skip).
        random_state : int, optional
            Seed for reproducible permutations.

        Returns:
        --------
        dict
            Procrustes results including the ``m12_squared`` disparity, the
            Procrustes correlation and a PROTEST p-value.
        """
        mat_a = np.asarray(data1.values if isinstance(data1, pd.DataFrame) else data1, dtype=float)
        mat_b = np.asarray(data2.values if isinstance(data2, pd.DataFrame) else data2, dtype=float)

        if mat_a.shape[0] != mat_b.shape[0]:
            raise ValueError("Both ordinations must contain the same number of rows")

        # scipy.spatial.procrustes requires matching column counts.
        n_cols = min(mat_a.shape[1], mat_b.shape[1])
        mat_a, mat_b = mat_a[:, :n_cols], mat_b[:, :n_cols]

        mtx1, mtx2, disparity = procrustes(mat_a, mat_b)
        correlation = float(np.sqrt(max(0.0, 1 - disparity)))

        p_value = np.nan
        if permutations and permutations > 0:
            rng = as_generator(random_state)
            count = 0
            for _ in range(permutations):
                permuted = mat_b[rng.permutation(mat_b.shape[0])]
                try:
                    _, _, perm_disparity = procrustes(mat_a, permuted)
                except ValueError:  # pragma: no cover - degenerate input
                    continue
                if perm_disparity <= disparity:
                    count += 1
            p_value = (count + 1) / (permutations + 1)

        return {
            'transformed_data1': mtx1,
            'transformed_data2': mtx2,
            'procrustes_disparity': disparity,
            'm12_squared': disparity,
            'correlation': correlation,
            'p_value': p_value,
            'permutations': permutations,
            'sum_of_squares': float(np.sum((mtx1 - mtx2) ** 2))
        }

    def environmental_fitting(self, ordination_scores: pd.DataFrame,
                              env_data: pd.DataFrame,
                              method: str = 'vector',
                              permutations: int = 999,
                              random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        Fit environmental vectors to an ordination (``envfit``-style).

        Each variable is regressed on the ordination axes; the resulting
        direction is normalised to unit length (the direction of steepest
        increase) and scaled by ``sqrt(R^2)`` for plotting, and significance is
        assessed by permuting the variable across sites.

        Parameters:
        -----------
        ordination_scores : pd.DataFrame
            Site scores (sites x axes)
        env_data : pd.DataFrame
            Environmental variables
        method : str
            Currently only ``'vector'`` is implemented.
        permutations : int
            Number of permutations for the significance test (0 to skip).
        random_state : int, optional
            Seed for reproducible permutations.

        Returns:
        --------
        dict
            ``vectors`` maps each variable to ``arrow_coords`` (unit direction
            scaled by sqrt(R^2)), ``direction`` (unit vector), ``r_squared``
            and ``p_value``. The flat ``environmental_vectors``, ``r_squared``
            and ``p_values`` dictionaries are retained for compatibility.
        """
        if method != 'vector':
            raise ValueError("Only method='vector' is currently supported")

        common_index = ordination_scores.index.intersection(env_data.index)
        if len(common_index) == 0:
            raise ValueError("Ordination scores and environmental data share no sites")

        scores_aligned = ordination_scores.loc[common_index]
        env_aligned = env_data.loc[common_index].select_dtypes(include=[np.number])

        rng = as_generator(random_state)

        results: Dict[str, Any] = {
            'method': method,
            'vectors': {},
            'environmental_vectors': {},
            'r_squared': {},
            'p_values': {},
            'n_sites': len(common_index),
        }

        for env_var in env_aligned.columns:
            valid = env_aligned[env_var].notna()
            if valid.sum() < 3:
                continue

            X = scores_aligned.loc[valid].values.astype(float)
            y = env_aligned.loc[valid, env_var].values.astype(float)

            r_squared, direction = self._fit_env_vector(X, y)
            if direction is None:
                continue

            p_value = np.nan
            if permutations and permutations > 0:
                count = 0
                for _ in range(permutations):
                    perm_r2, _ = self._fit_env_vector(X, rng.permutation(y))
                    if perm_r2 >= r_squared:
                        count += 1
                p_value = (count + 1) / (permutations + 1)

            arrow = direction * np.sqrt(max(r_squared, 0.0))

            results['vectors'][env_var] = {
                'arrow_coords': arrow,
                'direction': direction,
                'r_squared': r_squared,
                'p_value': p_value,
            }
            results['environmental_vectors'][env_var] = arrow
            results['r_squared'][env_var] = r_squared
            results['p_values'][env_var] = p_value

        return results

    @staticmethod
    def _fit_env_vector(X: np.ndarray, y: np.ndarray):
        """Least-squares fit of one variable on the ordination axes."""
        X_centred = X - X.mean(axis=0)
        y_centred = y - y.mean()

        ss_total = float(np.sum(y_centred ** 2))
        if ss_total == 0:
            return 0.0, None

        coefficients, *_ = np.linalg.lstsq(X_centred, y_centred, rcond=None)
        fitted = X_centred @ coefficients
        r_squared = float(1 - np.sum((y_centred - fitted) ** 2) / ss_total)

        norm = float(np.linalg.norm(coefficients))
        if norm == 0:
            return r_squared, None

        return r_squared, coefficients / norm

    def goodness_of_fit_test(self, ordination_results: Dict[str, Any],
                             original_data: pd.DataFrame,
                             distance_metric: str = 'bray_curtis') -> Dict[str, Any]:
        """
        Assess how faithfully an ordination reproduces the original distances.

        Parameters:
        -----------
        ordination_results : dict
            Results from any ordination method in this module.
        original_data : pd.DataFrame
            Original site-by-species matrix.
        distance_metric : str
            Distance metric for the original data.

        Returns:
        --------
        dict
            Pearson and Spearman correlations between original and ordination
            distances, Kruskal's stress-1 (computed after an optimal linear
            rescaling, so it is invariant to the arbitrary scale of the
            ordination) and Shepard-plot data.
        """
        for key in ('site_scores', 'scores', 'coordinates'):
            if key in ordination_results:
                ord_scores = ordination_results[key]
                break
        else:
            raise ValueError("No ordination scores found in results")

        orig_distances = self._condensed(original_data.values, distance_metric)
        ord_distances = pdist(np.asarray(ord_scores.values, dtype=float), metric='euclidean')

        correlation = float(np.corrcoef(orig_distances, ord_distances)[0, 1])
        spearman = float(stats.spearmanr(orig_distances, ord_distances)[0])

        # Optimal scale factor between the two distance sets, then Kruskal
        # stress-1. Without this rescaling the "stress" would depend on the
        # arbitrary units of the ordination axes.
        denom = float(np.sum(orig_distances ** 2))
        scale = float(np.sum(orig_distances * ord_distances) / denom) if denom > 0 else 1.0
        fitted = scale * orig_distances
        residual = float(np.sum((ord_distances - fitted) ** 2))
        total = float(np.sum(ord_distances ** 2))
        stress = float(np.sqrt(residual / total)) if total > 0 else 0.0

        shepard_data = pd.DataFrame({
            'original_distance': orig_distances,
            'ordination_distance': ord_distances,
            'fitted_distance': fitted,
        })

        return {
            'correlation': correlation,
            'spearman_correlation': spearman,
            'stress': stress,
            'r_squared': correlation ** 2,
            'shepard_plot_data': shepard_data,
            'distance_metric': distance_metric
        }

    def _calculate_gradient_lengths(self, site_scores: np.ndarray) -> np.ndarray:
        """Gradient length (range) of each axis, assuming SD-scaled scores."""
        return np.array([
            float(site_scores[:, axis].max() - site_scores[:, axis].min())
            for axis in range(site_scores.shape[1])
        ])
