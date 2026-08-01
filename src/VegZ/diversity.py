"""
Comprehensive diversity analysis module for vegetation data.

Copyright (c) 2025 Mohamed Z. Hatim
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional, Dict
from scipy.special import gammaln
import warnings


def _log_comb(n: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Log of the binomial coefficient C(n, k), vectorised and NaN-safe."""
    n = np.asarray(n, dtype=float)
    k = np.asarray(k, dtype=float)
    valid = (n >= k) & (k >= 0)
    out = np.full(np.broadcast(n, k).shape, -np.inf, dtype=float)
    with np.errstate(invalid='ignore'):
        vals = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    out = np.where(valid, vals, -np.inf)
    return out


class DiversityAnalyzer:
    """Comprehensive diversity analysis for ecological communities.

    Conventions
    -----------
    ``simpson`` returns Simpson's concentration/dominance index
    :math:`D = \\sum p_i^2` (low values mean high diversity). The complementary
    Gini-Simpson index :math:`1 - D` is available as ``gini_simpson`` and the
    reciprocal :math:`1/D` as ``simpson_inv``.

    ``jack1``/``jack2`` are incidence-based estimators of *dataset-level*
    richness: they are computed across all samples and therefore return a
    single number rather than one value per sample.
    """

    #: Indices returning one value per sample.
    per_sample_indices = [
        'shannon', 'simpson', 'gini_simpson', 'simpson_inv', 'richness',
        'evenness', 'fisher_alpha', 'berger_parker', 'mcintosh', 'brillouin',
        'menhinick', 'margalef', 'chao1', 'ace'
    ]

    #: Indices returning a single value for the whole data set.
    dataset_indices = ['jack1', 'jack2']

    def __init__(self):
        """Initialize diversity analyzer."""
        self.available_indices = list(self.per_sample_indices) + list(self.dataset_indices)

    def calculate_all_indices(self, data: pd.DataFrame,
                              include_dataset_level: bool = False) -> pd.DataFrame:
        """
        Calculate all available diversity indices.

        Parameters:
        -----------
        data : pd.DataFrame
            Species abundance matrix (samples x species)
        include_dataset_level : bool
            If True, also append the dataset-level incidence estimators
            (``jack1``, ``jack2``) as constant columns. They are excluded by
            default because repeating a single number on every row is easy to
            misread as a per-sample estimate.

        Returns:
        --------
        pd.DataFrame
            Diversity indices for each sample
        """
        results = pd.DataFrame(index=data.index)

        for index in self.per_sample_indices:
            try:
                results[index] = self.calculate_index(data, index)
            except Exception as e:  # pragma: no cover - defensive
                warnings.warn(f"Could not calculate {index}: {e}")
                results[index] = np.nan

        if include_dataset_level:
            for index in self.dataset_indices:
                try:
                    results[index] = float(self.calculate_index(data, index))
                except Exception as e:  # pragma: no cover - defensive
                    warnings.warn(f"Could not calculate {index}: {e}")
                    results[index] = np.nan

        return results

    def calculate_index(self, data: pd.DataFrame, index: str) -> Union[pd.Series, float]:
        """
        Calculate a specific diversity index.

        Parameters:
        -----------
        data : pd.DataFrame
            Species abundance matrix
        index : str
            Name of diversity index

        Returns:
        --------
        pd.Series or float
            Per-sample values, or a single value for dataset-level estimators
            (``jack1``, ``jack2``).
        """
        dispatch = {
            'shannon': self.shannon_diversity,
            'simpson': self.simpson_diversity,
            'gini_simpson': self.gini_simpson,
            'simpson_inv': self.simpson_inverse,
            'richness': self.species_richness,
            'evenness': self.pielou_evenness,
            'fisher_alpha': self.fisher_alpha,
            'berger_parker': self.berger_parker,
            'mcintosh': self.mcintosh_diversity,
            'brillouin': self.brillouin_diversity,
            'menhinick': self.menhinick_index,
            'margalef': self.margalef_index,
            'chao1': self.chao1_estimator,
            'ace': self.ace_estimator,
            'jack1': self.jackknife1_estimator,
            'jack2': self.jackknife2_estimator,
        }

        index_lower = index.lower()
        if index_lower not in dispatch:
            raise ValueError(f"Unknown diversity index: {index}")

        return dispatch[index_lower](data)

    # ------------------------------------------------------------------
    # Per-sample indices
    # ------------------------------------------------------------------

    def shannon_diversity(self, data: pd.DataFrame) -> pd.Series:
        """Shannon diversity index (H', natural log)."""
        def calculate_shannon(row):
            abundances = row[row > 0]
            if len(abundances) == 0:
                return 0.0
            proportions = abundances / abundances.sum()
            return float(-np.sum(proportions * np.log(proportions)))

        return data.apply(calculate_shannon, axis=1)

    def simpson_diversity(self, data: pd.DataFrame) -> pd.Series:
        """Simpson's concentration (dominance) index D = sum(p_i^2)."""
        def calculate_simpson(row):
            abundances = row[row > 0]
            if len(abundances) == 0:
                return 0.0
            proportions = abundances / abundances.sum()
            return float(np.sum(proportions ** 2))

        return data.apply(calculate_simpson, axis=1)

    def gini_simpson(self, data: pd.DataFrame) -> pd.Series:
        """Gini-Simpson index (1 - D); probability two draws differ."""
        simpson = self.simpson_diversity(data)
        richness = self.species_richness(data)
        gini = 1 - simpson
        # An empty sample has no diversity rather than a diversity of 1.
        return gini.where(richness > 0, 0.0)

    def simpson_inverse(self, data: pd.DataFrame) -> pd.Series:
        """Inverse Simpson diversity (1/D)."""
        simpson = self.simpson_diversity(data)
        return (1 / simpson.replace(0, np.nan)).fillna(0.0)

    def species_richness(self, data: pd.DataFrame) -> pd.Series:
        """Species richness (S)."""
        return (data > 0).sum(axis=1)

    def pielou_evenness(self, data: pd.DataFrame) -> pd.Series:
        """Pielou's evenness (J' = H'/ln S); 0 for samples with S < 2."""
        shannon = self.shannon_diversity(data)
        richness = self.species_richness(data)
        log_richness = np.log(richness.replace({0: np.nan, 1: np.nan}))
        evenness = shannon / log_richness
        return evenness.fillna(0.0)

    def fisher_alpha(self, data: pd.DataFrame) -> pd.Series:
        """Fisher's alpha diversity (solves S = alpha * ln(1 + N/alpha))."""
        def calculate_fisher(row):
            abundances = row[row > 0]
            if len(abundances) == 0:
                return 0.0

            S = len(abundances)          # Species richness
            N = float(abundances.sum())  # Total abundance

            if S <= 1 or N <= 0:
                return 0.0

            # Fixed-point iteration; converges quickly for ecological data.
            alpha = float(S)
            for _ in range(200):
                denom = np.log1p(N / alpha)
                if denom <= 0:
                    break
                alpha_new = S / denom
                if abs(alpha_new - alpha) < 1e-8:
                    alpha = alpha_new
                    break
                alpha = alpha_new

            return float(alpha)

        return data.apply(calculate_fisher, axis=1)

    def berger_parker(self, data: pd.DataFrame) -> pd.Series:
        """Berger-Parker dominance index."""
        def calculate_bp(row):
            abundances = row[row > 0]
            if len(abundances) == 0:
                return 0.0
            return float(abundances.max() / abundances.sum())

        return data.apply(calculate_bp, axis=1)

    def mcintosh_diversity(self, data: pd.DataFrame) -> pd.Series:
        """McIntosh diversity index."""
        def calculate_mcintosh(row):
            abundances = row[row > 0]
            if len(abundances) == 0:
                return 0.0

            N = float(abundances.sum())
            U = float(np.sqrt(np.sum(np.square(abundances.astype(float)))))

            denominator = N - np.sqrt(N)
            if denominator <= 0:
                # A single individual (N == 1) leaves no room for diversity.
                return 0.0

            return float((N - U) / denominator)

        return data.apply(calculate_mcintosh, axis=1)

    def brillouin_diversity(self, data: pd.DataFrame) -> pd.Series:
        """Brillouin diversity index (requires integer counts)."""
        def calculate_brillouin(row):
            abundances = np.asarray(row[row > 0], dtype=float)
            if abundances.size == 0:
                return 0.0

            counts = np.rint(abundances)
            N = counts.sum()
            if N <= 0:
                return 0.0

            return float((gammaln(N + 1) - np.sum(gammaln(counts + 1))) / N)

        return data.apply(calculate_brillouin, axis=1)

    def menhinick_index(self, data: pd.DataFrame) -> pd.Series:
        """Menhinick's richness index (S / sqrt(N))."""
        richness = self.species_richness(data)
        total_abundance = data.sum(axis=1)
        return richness / np.sqrt(total_abundance.replace(0, np.nan)).fillna(1.0)

    def margalef_index(self, data: pd.DataFrame) -> pd.Series:
        """Margalef's richness index ((S - 1) / ln N)."""
        richness = self.species_richness(data)
        total_abundance = data.sum(axis=1)
        log_abundance = np.log(total_abundance.replace({0: np.nan, 1: np.nan}))
        margalef = (richness - 1) / log_abundance
        return margalef.fillna(0.0)

    def chao1_estimator(self, data: pd.DataFrame) -> pd.Series:
        """Bias-corrected Chao1 richness estimator (per sample)."""
        def calculate_chao1(row):
            abundances = np.asarray(row[row > 0], dtype=float)
            if abundances.size == 0:
                return 0.0

            S_obs = abundances.size
            f1 = float(np.sum(np.isclose(abundances, 1)))   # Singletons
            f2 = float(np.sum(np.isclose(abundances, 2)))   # Doubletons

            # Bias-corrected form (Chao 1987); reduces to the classic
            # S_obs + f1^2 / (2 f2) when f2 is large.
            return float(S_obs + (f1 * (f1 - 1)) / (2 * (f2 + 1)))

        return data.apply(calculate_chao1, axis=1)

    def ace_estimator(self, data: pd.DataFrame, rare_threshold: int = 10) -> pd.Series:
        """
        ACE (Abundance-based Coverage Estimator).

        Follows Chao & Lee (1992): the coefficient-of-variation correction uses
        ``sum_{i=1..k} i (i - 1) f_i``.
        """
        def calculate_ace(row):
            abundances = np.asarray(row[row > 0], dtype=float)
            if abundances.size == 0:
                return 0.0

            rare = abundances[abundances <= rare_threshold]
            abundant = abundances[abundances > rare_threshold]

            S_rare = rare.size
            S_abund = abundant.size

            if S_rare == 0:
                return float(S_abund)

            N_rare = float(rare.sum())
            f1 = float(np.sum(np.isclose(rare, 1)))

            C_ace = 1 - (f1 / N_rare) if N_rare > 0 else 1.0

            if C_ace <= 0:
                # Every rare species is a singleton: coverage estimate breaks
                # down, fall back on the observed richness.
                return float(S_abund + S_rare)

            if N_rare <= 1:
                gamma_sq = 0.0
            else:
                counts = np.rint(rare).astype(int)
                i_vals = np.arange(1, rare_threshold + 1)
                f_i = np.array([np.sum(counts == i) for i in i_vals], dtype=float)
                sum_i_i1_fi = float(np.sum(i_vals * (i_vals - 1) * f_i))
                gamma_sq = (S_rare / C_ace) * (sum_i_i1_fi / (N_rare * (N_rare - 1))) - 1
                gamma_sq = max(gamma_sq, 0.0)

            return float(S_abund + (S_rare / C_ace) + ((f1 / C_ace) * gamma_sq))

        return data.apply(calculate_ace, axis=1)

    # ------------------------------------------------------------------
    # Dataset-level estimators
    # ------------------------------------------------------------------

    def jackknife1_estimator(self, data: pd.DataFrame) -> float:
        """First-order Jackknife richness estimator (incidence-based)."""
        incidence = (data > 0).astype(int)
        n_samples = len(data)

        if n_samples < 2:
            return float((data > 0).any(axis=0).sum())

        species_incidence = incidence.sum(axis=0)
        S_obs = float((species_incidence > 0).sum())
        Q1 = float((species_incidence == 1).sum())

        return float(S_obs + Q1 * (n_samples - 1) / n_samples)

    def jackknife2_estimator(self, data: pd.DataFrame) -> float:
        """Second-order Jackknife richness estimator (incidence-based)."""
        incidence = (data > 0).astype(int)
        n_samples = len(data)

        if n_samples < 3:
            return self.jackknife1_estimator(data)

        species_incidence = incidence.sum(axis=0)
        S_obs = float((species_incidence > 0).sum())
        Q1 = float((species_incidence == 1).sum())
        Q2 = float((species_incidence == 2).sum())

        jack2 = (S_obs
                 + Q1 * (2 * n_samples - 3) / n_samples
                 - Q2 * ((n_samples - 2) ** 2) / (n_samples * (n_samples - 1)))
        return float(jack2)

    # ------------------------------------------------------------------
    # Beta diversity
    # ------------------------------------------------------------------

    def beta_diversity(self, data: pd.DataFrame,
                       method: str = 'whittaker',
                       pairwise: bool = True) -> Union[float, pd.DataFrame]:
        """
        Calculate beta diversity between samples.

        Parameters:
        -----------
        data : pd.DataFrame
            Species abundance matrix
        method : {'whittaker', 'sorensen', 'jaccard'}
            Beta diversity index. All three are incidence based.
        pairwise : bool
            If True (default) return the full sample-by-sample dissimilarity
            matrix. If False, return a single whole-data-set value; only
            ``'whittaker'`` supports this, in which case the multiplicative
            beta ``gamma / mean(alpha)`` is returned.

        Returns:
        --------
        float or pd.DataFrame
            Dissimilarity matrix (``pairwise=True``) or a scalar.

        Notes
        -----
        For two samples, Whittaker's beta and Sorensen dissimilarity are
        algebraically identical; both are provided because users reach for the
        names in different contexts.
        """
        method_lower = method.lower()
        if method_lower not in ('whittaker', 'sorensen', 'jaccard'):
            raise ValueError(f"Unknown beta diversity method: {method}")

        if not pairwise:
            if method_lower != 'whittaker':
                raise ValueError(
                    "pairwise=False is only defined for method='whittaker'"
                )
            return self.whittaker_beta(data)

        if method_lower == 'jaccard':
            return self._jaccard_beta(data)
        # Whittaker's pairwise beta equals Sorensen dissimilarity.
        return self._sorensen_beta(data)

    def whittaker_beta(self, data: pd.DataFrame) -> float:
        """Whittaker's multiplicative beta diversity, gamma / mean(alpha)."""
        gamma = float((data.sum(axis=0) > 0).sum())
        alpha_mean = float(self.species_richness(data).mean())
        if alpha_mean == 0:
            return 0.0
        return gamma / alpha_mean

    def _incidence(self, data: pd.DataFrame) -> np.ndarray:
        return (data.values > 0)

    def _sorensen_beta(self, data: pd.DataFrame) -> pd.DataFrame:
        """Sorensen (Bray-Curtis on incidences) dissimilarity matrix."""
        presence = self._incidence(data).astype(float)
        shared = presence @ presence.T
        totals = presence.sum(axis=1)
        denom = totals[:, None] + totals[None, :]

        with np.errstate(invalid='ignore', divide='ignore'):
            beta = 1 - (2 * shared) / denom
        beta = np.where(denom > 0, beta, 0.0)
        np.fill_diagonal(beta, 0.0)

        return pd.DataFrame(beta, index=data.index, columns=data.index)

    def _jaccard_beta(self, data: pd.DataFrame) -> pd.DataFrame:
        """Jaccard dissimilarity matrix."""
        presence = self._incidence(data).astype(float)
        shared = presence @ presence.T
        totals = presence.sum(axis=1)
        union = totals[:, None] + totals[None, :] - shared

        with np.errstate(invalid='ignore', divide='ignore'):
            beta = 1 - shared / union
        beta = np.where(union > 0, beta, 0.0)
        np.fill_diagonal(beta, 0.0)

        return pd.DataFrame(beta, index=data.index, columns=data.index)


    # ------------------------------------------------------------------
    # Baselga beta-diversity partitioning
    # ------------------------------------------------------------------

    def beta_partition(self, data: pd.DataFrame,
                       family: str = 'sorensen') -> Dict[str, pd.DataFrame]:
        """
        Partition pairwise beta diversity into turnover and nestedness.

        Baselga (2010, 2012). Total dissimilarity between two sites is split
        into the part caused by species *replacement* (turnover) and the part
        caused by one site being a *subset* of the other (nestedness-resultant
        dissimilarity). Two assemblages can have identical total beta diversity
        for completely different ecological reasons, and only the partition
        tells them apart.

        Parameters:
        -----------
        data : pd.DataFrame
            Species abundance or incidence matrix (sites x species).
        family : {'sorensen', 'jaccard'}
            ``'sorensen'`` gives ``beta_sim`` (turnover, Simpson dissimilarity)
            and ``beta_sne`` (nestedness) summing to ``beta_sor``.
            ``'jaccard'`` gives ``beta_jtu`` and ``beta_jne`` summing to
            ``beta_jac``.

        Returns:
        --------
        dict of pd.DataFrame
            Three site-by-site matrices: ``turnover``, ``nestedness`` and
            ``total``. By construction ``turnover + nestedness == total``.

        Examples
        --------
        A species-poor site that is a perfect subset of a richer one has zero
        turnover: all of its dissimilarity is nestedness. Two equally rich sites
        sharing no species have turnover 1 and nestedness 0.
        """
        family = family.lower()
        if family not in ('sorensen', 'jaccard'):
            raise ValueError("family must be 'sorensen' or 'jaccard'")

        presence = (data.values > 0).astype(float)
        shared = presence @ presence.T                     # a
        totals = presence.sum(axis=1)
        only_i = totals[:, None] - shared                  # b
        only_j = totals[None, :] - shared                  # c

        min_bc = np.minimum(only_i, only_j)
        max_bc = np.maximum(only_i, only_j)

        with np.errstate(divide='ignore', invalid='ignore'):
            if family == 'sorensen':
                total = (only_i + only_j) / (2 * shared + only_i + only_j)
                turnover = min_bc / (shared + min_bc)
                # beta_sne = beta_sor - beta_sim, written out to stay exact.
                nestedness = ((max_bc - min_bc) / (2 * shared + only_i + only_j)) * \
                             (shared / (shared + min_bc))
            else:
                total = (only_i + only_j) / (shared + only_i + only_j)
                turnover = (2 * min_bc) / (shared + 2 * min_bc)
                nestedness = ((max_bc - min_bc) / (shared + only_i + only_j)) * \
                             (shared / (shared + 2 * min_bc))

        def clean(matrix: np.ndarray) -> pd.DataFrame:
            matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(matrix, 0.0)
            return pd.DataFrame(matrix, index=data.index, columns=data.index)

        return {
            'turnover': clean(turnover),
            'nestedness': clean(nestedness),
            'total': clean(total),
            'family': family,
        }

    def beta_partition_multisite(self, data: pd.DataFrame,
                                 family: str = 'sorensen') -> Dict[str, float]:
        """
        Multi-site beta-diversity partitioning (Baselga 2010).

        The whole-dataset counterpart of :meth:`beta_partition`: a single
        turnover and a single nestedness value for the entire set of sites,
        rather than one per pair.

        Parameters:
        -----------
        data : pd.DataFrame
            Species incidence/abundance matrix (sites x species).
        family : {'sorensen', 'jaccard'}
            Dissimilarity family.

        Returns:
        --------
        dict
            ``turnover``, ``nestedness`` and ``total``, which sum exactly.
        """
        family = family.lower()
        if family not in ('sorensen', 'jaccard'):
            raise ValueError("family must be 'sorensen' or 'jaccard'")

        presence = (data.values > 0).astype(float)
        n_sites = presence.shape[0]
        if n_sites < 2:
            return {'turnover': 0.0, 'nestedness': 0.0, 'total': 0.0, 'family': family}

        richness = presence.sum(axis=1)
        s_total = float((presence.sum(axis=0) > 0).sum())   # gamma
        sum_si = float(richness.sum())

        shared = presence @ presence.T
        totals = presence.sum(axis=1)
        only_i = totals[:, None] - shared
        only_j = totals[None, :] - shared

        i_idx, j_idx = np.triu_indices(n_sites, k=1)
        b = only_i[i_idx, j_idx]
        c = only_j[i_idx, j_idx]

        sum_min = float(np.minimum(b, c).sum())
        sum_max = float(np.maximum(b, c).sum())

        denominator_shared = sum_si - s_total   # sum over sites of Si minus St

        if family == 'sorensen':
            total = (sum_min + sum_max) / (2 * denominator_shared + sum_min + sum_max) \
                if (2 * denominator_shared + sum_min + sum_max) > 0 else 0.0
            turnover = sum_min / (denominator_shared + sum_min) \
                if (denominator_shared + sum_min) > 0 else 0.0
        else:
            total = (sum_min + sum_max) / (denominator_shared + sum_min + sum_max) \
                if (denominator_shared + sum_min + sum_max) > 0 else 0.0
            turnover = (2 * sum_min) / (denominator_shared + 2 * sum_min) \
                if (denominator_shared + 2 * sum_min) > 0 else 0.0

        return {
            'turnover': float(turnover),
            'nestedness': float(total - turnover),
            'total': float(total),
            'family': family,
        }

    # ------------------------------------------------------------------
    # Sample coverage and coverage-based standardisation
    # ------------------------------------------------------------------

    @staticmethod
    def _abundance_frequencies(counts: np.ndarray) -> Dict[str, float]:
        """Sample size and singleton/doubleton counts for a single sample."""
        counts = counts[counts > 0]
        return {
            'n': float(counts.sum()),
            'f1': float(np.sum(counts == 1)),
            'f2': float(np.sum(counts == 2)),
            's_obs': float(counts.size),
        }

    def sample_coverage(self, data: pd.DataFrame) -> pd.Series:
        """
        Estimated sample coverage of each sample (Chao & Jost 2012).

        Coverage is the proportion of the community's total individuals that
        belong to species actually detected - i.e. how complete the sample is.
        Comparing diversity between samples of equal *coverage* is the
        recommended alternative to comparing at equal *sample size*, because
        equal-size samples of communities with different evenness are not
        equally complete.

        Returns:
        --------
        pd.Series
            Estimated coverage in [0, 1] for each sample.
        """
        counts_matrix = self._integer_counts(data)

        coverage = []
        for row in counts_matrix:
            freq = self._abundance_frequencies(row)
            n, f1, f2 = freq['n'], freq['f1'], freq['f2']
            if n <= 0:
                coverage.append(0.0)
            elif f1 == 0:
                coverage.append(1.0)
            else:
                # C_hat = 1 - (f1/n) * ((n-1)f1 / ((n-1)f1 + 2f2))
                denominator = (n - 1) * f1 + 2 * f2
                factor = ((n - 1) * f1 / denominator) if denominator > 0 else 1.0
                coverage.append(float(max(0.0, min(1.0, 1 - (f1 / n) * factor))))

        return pd.Series(coverage, index=data.index, name='coverage')

    @staticmethod
    def _integer_counts(data: pd.DataFrame) -> np.ndarray:
        """Round abundances to counts, warning if they were not integers."""
        values = data.values.astype(float)
        if not np.allclose(values, np.rint(values)):
            warnings.warn(
                "Coverage and Hill-number estimators are defined for integer "
                "counts; non-integer abundances were rounded."
            )
        return np.rint(values).astype(np.int64)

    def coverage_at_size(self, counts: np.ndarray, m: int) -> float:
        """
        Expected coverage of a subsample of ``m`` individuals.

        ``C(m) = 1 - sum_i (x_i / n) * C(n - x_i, m) / C(n - 1, m)``.
        """
        counts = np.asarray(counts, dtype=float)
        counts = counts[counts > 0]
        n = counts.sum()
        if n <= 0 or m <= 0:
            return 0.0
        if m >= n:
            freq = self._abundance_frequencies(counts)
            f1, f2 = freq['f1'], freq['f2']
            if f1 == 0:
                return 1.0
            denominator = (n - 1) * f1 + 2 * f2
            factor = ((n - 1) * f1 / denominator) if denominator > 0 else 1.0
            return float(max(0.0, min(1.0, 1 - (f1 / n) * factor)))

        log_denominator = _log_comb(n - 1, m)
        with np.errstate(invalid='ignore'):
            terms = np.exp(_log_comb(n - counts, m) - log_denominator)
        terms = np.nan_to_num(terms, nan=0.0)
        return float(max(0.0, min(1.0, 1 - np.sum((counts / n) * terms))))

    def hill_rarefaction(self, data: pd.DataFrame,
                         sizes: Optional[List[int]] = None,
                         q_values: Optional[List[float]] = None,
                         n_points: int = 15,
                         extrapolate_to: Optional[int] = None) -> pd.DataFrame:
        """
        Hill-number rarefaction and extrapolation curves (Chao et al. 2014).

        Unlike :meth:`rarefaction_curve`, which only reports species richness,
        this gives the whole diversity profile: ``q = 0`` is richness, ``q = 1``
        the exponential of Shannon entropy and ``q = 2`` the inverse Simpson
        concentration. Higher ``q`` weights common species more, so the three
        curves together separate "many rare species" from "a few dominant ones".

        Parameters:
        -----------
        data : pd.DataFrame
            Species count matrix (sites x species).
        sizes : list of int, optional
            Sample sizes to evaluate. Defaults to ``n_points`` values spread up
            to the reference sample size (or ``extrapolate_to``).
        q_values : list of float, optional
            Diversity orders; only 0, 1 and 2 are supported. Default ``[0,1,2]``.
        n_points : int
            Number of default sizes.
        extrapolate_to : int, optional
            Extrapolate each sample up to this many individuals. Extrapolation
            beyond about twice the reference sample size is unreliable and is
            warned about.

        Returns:
        --------
        pd.DataFrame
            Columns ``sample_id``, ``q``, ``size``, ``diversity``, ``coverage``
            and ``method`` (``'rarefaction'``, ``'observed'`` or
            ``'extrapolation'``).
        """
        if q_values is None:
            q_values = [0, 1, 2]
        unsupported = [q for q in q_values if q not in (0, 1, 2)]
        if unsupported:
            raise ValueError(
                f"Only q = 0, 1 and 2 are supported; got {unsupported}")

        counts_matrix = self._integer_counts(data)
        records = []

        for row_index, sample_id in enumerate(data.index):
            counts = counts_matrix[row_index]
            counts = counts[counts > 0].astype(float)
            n = int(counts.sum())
            if n == 0:
                continue

            upper = extrapolate_to if extrapolate_to else n
            if extrapolate_to and extrapolate_to > 2 * n:
                warnings.warn(
                    f"Extrapolating sample '{sample_id}' from {n} to "
                    f"{extrapolate_to} individuals (more than double the "
                    "reference size); such estimates are unreliable."
                )

            if sizes is None:
                grid = sorted(set(
                    np.linspace(1, upper, min(n_points, upper)).astype(int).tolist()
                    + [n]
                ))
            else:
                grid = sorted(set(int(s) for s in sizes if s >= 1))

            for m in grid:
                for q in q_values:
                    if m < n:
                        value = self._hill_rarefy(counts, n, m, q)
                        kind = 'rarefaction'
                    elif m == n:
                        value = self._hill_observed(counts, q)
                        kind = 'observed'
                    else:
                        value = self._hill_extrapolate(counts, n, m - n, q)
                        kind = 'extrapolation'

                    records.append({
                        'sample_id': sample_id,
                        'q': q,
                        'size': m,
                        'diversity': value,
                        'coverage': self.coverage_at_size(counts, m)
                        if m <= n else self._coverage_extrapolated(counts, n, m - n),
                        'method': kind,
                    })

        return pd.DataFrame(records)

    @staticmethod
    def _hill_observed(counts: np.ndarray, q: float) -> float:
        """Hill number of the reference sample itself."""
        p = counts / counts.sum()
        if q == 0:
            return float(counts.size)
        if q == 1:
            return float(np.exp(-np.sum(p * np.log(p))))
        return float(1.0 / np.sum(p ** 2))

    def _hill_rarefy(self, counts: np.ndarray, n: int, m: int, q: float) -> float:
        """Expected Hill number of a subsample of ``m`` individuals."""
        if q == 0:
            log_denominator = _log_comb(n, m)
            absent = np.exp(_log_comb(n - counts, m) - log_denominator)
            absent = np.clip(np.nan_to_num(absent, nan=0.0), 0.0, 1.0)
            return float(np.sum(1 - absent))

        if q == 2:
            # Exact and closed-form: 1 / [1/m + (1 - 1/m) * sum x_i(x_i-1)/(n(n-1))]
            if n < 2:
                return 1.0
            concentration = float(np.sum(counts * (counts - 1)) / (n * (n - 1)))
            denominator = 1.0 / m + (1 - 1.0 / m) * concentration
            return float(1.0 / denominator) if denominator > 0 else float(m)

        # q == 1: expected Shannon entropy of a size-m subsample, evaluated in
        # log space so it is stable for large counts.
        k = np.arange(1, m + 1, dtype=float)
        entropy = 0.0
        log_denominator = _log_comb(n, m)
        for x in counts:
            valid = k <= min(x, m)
            if not valid.any():
                continue
            kk = k[valid]
            log_prob = (_log_comb(np.full(kk.shape, x), kk)
                        + _log_comb(np.full(kk.shape, n - x), m - kk)
                        - log_denominator)
            prob = np.exp(log_prob)
            ratio = kk / m
            entropy -= float(np.sum(prob * ratio * np.log(ratio)))
        return float(np.exp(entropy))

    @staticmethod
    def _chao_f0(counts: np.ndarray, n: int) -> float:
        """
        Chao1 estimate of the number of *undetected* species.

        Uses the ``(n-1)/n``-corrected form specified by Chao et al. (2014) for
        extrapolation. Note this differs slightly from
        :meth:`chao1_estimator`, which reports the bias-corrected
        ``S_obs + f1(f1-1)/(2(f2+1))`` variant; the two are both standard Chao1
        forms and converge for large samples, so the asymptote of a ``q = 0``
        extrapolation curve will not exactly equal ``chao1_estimator``.
        """
        f1 = float(np.sum(counts == 1))
        f2 = float(np.sum(counts == 2))
        if f2 > 0:
            return ((n - 1) / n) * f1 * f1 / (2 * f2)
        return ((n - 1) / n) * f1 * (f1 - 1) / 2

    @staticmethod
    def _coverage_deficit_A(counts: np.ndarray, n: int) -> float:
        """The quantity ``A`` used by the Chao et al. (2014) extrapolators."""
        f1 = float(np.sum(counts == 1))
        f2 = float(np.sum(counts == 2))
        if f1 == 0:
            return 1.0
        if f2 > 0:
            return float(2 * f2 / ((n - 1) * f1 + 2 * f2))
        return float(2.0 / ((n - 1) * (f1 - 1) + 2)) if (n - 1) * (f1 - 1) + 2 > 0 else 1.0

    def _coverage_extrapolated(self, counts: np.ndarray, n: int, m_star: int) -> float:
        """Expected coverage of an extrapolated sample of size n + m_star."""
        f1 = float(np.sum(counts == 1))
        f2 = float(np.sum(counts == 2))
        if f1 == 0:
            return 1.0
        denominator = (n - 1) * f1 + 2 * f2
        if denominator <= 0:
            return 1.0
        base = (n - 1) * f1 / denominator
        return float(max(0.0, min(1.0, 1 - (f1 / n) * base ** (m_star + 1))))

    def _hill_extrapolate(self, counts: np.ndarray, n: int, m_star: int,
                          q: float) -> float:
        """Extrapolated Hill number for ``m_star`` extra individuals."""
        if m_star <= 0:
            return self._hill_observed(counts, q)

        if q == 0:
            f0 = self._chao_f0(counts, n)
            f1 = float(np.sum(counts == 1))
            s_obs = float(counts.size)
            if f0 <= 0 or f1 <= 0:
                return s_obs
            return float(s_obs + f0 * (1 - (1 - f1 / (n * f0 + f1)) ** m_star))

        if q == 2:
            # The rarefaction formula extends directly to n + m_star.
            if n < 2:
                return 1.0
            size = n + m_star
            concentration = float(np.sum(counts * (counts - 1)) / (n * (n - 1)))
            denominator = 1.0 / size + (1 - 1.0 / size) * concentration
            return float(1.0 / denominator) if denominator > 0 else float(size)

        # q == 1: observed entropy plus the estimated increment, approached at
        # the rate the coverage deficit closes (Chao et al. 2014).
        p = counts / n
        observed_entropy = float(-np.sum(p * np.log(p)))
        f1 = float(np.sum(counts == 1))
        A = self._coverage_deficit_A(counts, n)

        if f1 == 0 or A >= 1.0:
            return float(np.exp(observed_entropy))

        r = np.arange(1, n, dtype=float)
        increment = (f1 / n) * (1 - A) ** (1 - n) * (
            -np.log(A) - float(np.sum((1 - A) ** r / r))
        )
        increment = max(increment, 0.0)
        entropy = observed_entropy + increment * (1 - (1 - A) ** m_star)
        return float(np.exp(entropy))

    def coverage_standardized_diversity(self, data: pd.DataFrame,
                                        coverage: Optional[float] = None,
                                        q_values: Optional[List[float]] = None,
                                        max_extrapolation_factor: float = 2.0
                                        ) -> pd.DataFrame:
        """
        Compare samples at equal *coverage* rather than equal sample size.

        Chao & Jost (2012). Equal-sized samples from communities that differ in
        evenness are not equally complete, so rarefying to a common sample size
        systematically under-represents the more diverse community.
        Standardising to common coverage removes that bias.

        Parameters:
        -----------
        data : pd.DataFrame
            Species count matrix (sites x species).
        coverage : float, optional
            Target coverage in (0, 1). Defaults to the smallest coverage
            achievable by every sample at twice its reference size (the usual
            "base coverage" recommendation).
        q_values : list of float, optional
            Diversity orders (0, 1, 2). Default ``[0, 1, 2]``.
        max_extrapolation_factor : float
            Never extrapolate a sample beyond this multiple of its own size.

        Returns:
        --------
        pd.DataFrame
            One row per sample and order, with the size needed to reach the
            target coverage, the achieved coverage, and the diversity there.
        """
        if q_values is None:
            q_values = [0, 1, 2]

        counts_matrix = self._integer_counts(data)
        samples = {}
        for row_index, sample_id in enumerate(data.index):
            counts = counts_matrix[row_index]
            counts = counts[counts > 0].astype(float)
            if counts.sum() > 0:
                samples[sample_id] = counts

        if not samples:
            return pd.DataFrame(
                columns=['sample_id', 'q', 'target_coverage', 'size',
                         'achieved_coverage', 'diversity'])

        if coverage is None:
            # Highest coverage every sample can reach within the extrapolation
            # limit, so no sample is pushed past what its data can support.
            reachable = [
                self._coverage_extrapolated(
                    counts, int(counts.sum()),
                    int(counts.sum() * (max_extrapolation_factor - 1)))
                for counts in samples.values()
            ]
            coverage = float(min(reachable))

        if not 0 < coverage < 1:
            raise ValueError("coverage must be strictly between 0 and 1")

        records = []
        for sample_id, counts in samples.items():
            n = int(counts.sum())
            size = self._size_for_coverage(
                counts, n, coverage, int(n * max_extrapolation_factor))
            achieved = (self.coverage_at_size(counts, size) if size <= n
                        else self._coverage_extrapolated(counts, n, size - n))

            for q in q_values:
                if size < n:
                    value = self._hill_rarefy(counts, n, size, q)
                elif size == n:
                    value = self._hill_observed(counts, q)
                else:
                    value = self._hill_extrapolate(counts, n, size - n, q)

                records.append({
                    'sample_id': sample_id,
                    'q': q,
                    'target_coverage': coverage,
                    'size': size,
                    'achieved_coverage': achieved,
                    'diversity': value,
                })

        return pd.DataFrame(records)

    def _size_for_coverage(self, counts: np.ndarray, n: int,
                           target: float, max_size: int) -> int:
        """Smallest sample size whose expected coverage reaches ``target``."""
        low, high = 1, max(max_size, n)

        def coverage_of(size: int) -> float:
            if size <= n:
                return self.coverage_at_size(counts, size)
            return self._coverage_extrapolated(counts, n, size - n)

        if coverage_of(high) < target:
            return high

        while low < high:
            middle = (low + high) // 2
            if coverage_of(middle) >= target:
                high = middle
            else:
                low = middle + 1
        return int(low)

    # ------------------------------------------------------------------
    # Hill numbers
    # ------------------------------------------------------------------

    def hill_numbers(self, data: pd.DataFrame,
                     q_values: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Calculate Hill numbers (diversity of order q).

        Parameters:
        -----------
        data : pd.DataFrame
            Species abundance matrix
        q_values : list
            Orders of diversity to calculate (default ``[0, 1, 2]``)

        Returns:
        --------
        pd.DataFrame
            Hill numbers for each sample and order
        """
        if q_values is None:
            q_values = [0, 1, 2]

        results = pd.DataFrame(index=data.index)

        for q in q_values:
            results[f'Hill_q{q}'] = self._calculate_hill_number(data, q)

        return results

    def _calculate_hill_number(self, data: pd.DataFrame, q: float) -> pd.Series:
        """Calculate Hill number of order q."""
        def hill_q(row):
            abundances = row[row > 0]
            if len(abundances) == 0:
                return 0.0

            proportions = np.asarray(abundances / abundances.sum(), dtype=float)

            if np.isclose(q, 0):
                return float(proportions.size)
            if np.isclose(q, 1):
                return float(np.exp(-np.sum(proportions * np.log(proportions))))
            return float(np.sum(proportions ** q) ** (1 / (1 - q)))

        return data.apply(hill_q, axis=1)

    # ------------------------------------------------------------------
    # Rarefaction / accumulation
    # ------------------------------------------------------------------

    def rarefaction_curve(self, data: pd.DataFrame,
                          sample_sizes: Optional[List[int]] = None,
                          n_points: int = 20) -> pd.DataFrame:
        """
        Individual-based (Hurlbert) rarefaction for each sample.

        Parameters:
        -----------
        data : pd.DataFrame
            Species abundance matrix of counts (samples x species)
        sample_sizes : list of int, optional
            Numbers of individuals to rarefy to. Defaults to ``n_points``
            values spread from 1 to the largest sample total.
        n_points : int
            Number of default sample sizes when ``sample_sizes`` is None.

        Returns:
        --------
        pd.DataFrame
            Long-format table with columns ``sample_id``, ``sample_size``,
            ``expected_species`` and ``variance``.

        Notes
        -----
        Uses Hurlbert's (1971) expectation
        ``E[S_m] = sum_i (1 - C(N - n_i, m) / C(N, m))`` evaluated in log space,
        so it is exact and stable for large counts. Non-integer abundances
        (e.g. cover percentages) are rounded to the nearest integer with a
        warning, because rarefaction is only defined for counts.
        """
        values = data.values.astype(float)
        if not np.allclose(values, np.rint(values)):
            warnings.warn(
                "Rarefaction is defined for integer counts; non-integer "
                "abundances were rounded to the nearest integer."
            )
        counts_matrix = np.rint(values).astype(np.int64)
        totals = counts_matrix.sum(axis=1)

        if sample_sizes is None:
            max_total = int(totals.max()) if totals.size else 0
            if max_total < 1:
                return pd.DataFrame(
                    columns=['sample_id', 'sample_size', 'expected_species', 'variance']
                )
            sample_sizes = sorted(set(
                np.linspace(1, max_total, min(n_points, max_total)).astype(int).tolist()
            ))

        records = []
        for row_idx, sample_id in enumerate(data.index):
            counts = counts_matrix[row_idx]
            counts = counts[counts > 0]
            N = int(counts.sum())
            if N == 0:
                continue

            for m in sample_sizes:
                m = int(m)
                if m < 1 or m > N:
                    continue
                expected, variance = self._hurlbert_expectation(counts, N, m)
                records.append({
                    'sample_id': sample_id,
                    'sample_size': m,
                    'expected_species': expected,
                    'variance': variance,
                })

        return pd.DataFrame(records)

    @staticmethod
    def _hurlbert_expectation(counts: np.ndarray, N: int, m: int):
        """Expected richness and its variance for a subsample of size ``m``."""
        log_denom = _log_comb(N, m)
        # Probability that species i is absent from a subsample of size m.
        log_absent = _log_comb(N - counts, m) - log_denom
        p_absent = np.exp(log_absent)
        p_absent = np.clip(np.nan_to_num(p_absent, nan=0.0), 0.0, 1.0)

        expected = float(np.sum(1 - p_absent))

        # Heck et al. (1975) variance of the rarefaction estimate.
        var = float(np.sum(p_absent * (1 - p_absent)))
        n_species = counts.size
        for i in range(n_species):
            for j in range(i + 1, n_species):
                log_joint = _log_comb(N - counts[i] - counts[j], m) - log_denom
                p_joint = float(np.exp(log_joint)) if np.isfinite(log_joint) else 0.0
                var += 2 * (p_joint - p_absent[i] * p_absent[j])

        return expected, max(var, 0.0)

    def species_accumulation_curve(self, data: pd.DataFrame,
                                   n_permutations: int = 100,
                                   random_state: Optional[int] = None) -> pd.DataFrame:
        """
        Sample-based species accumulation curve with permutation confidence bands.

        Parameters:
        -----------
        data : pd.DataFrame
            Species abundance matrix (samples x species)
        n_permutations : int
            Number of random sample orderings used to build the curve.
        random_state : int, optional
            Seed for reproducible permutations.

        Returns:
        --------
        pd.DataFrame
            Columns ``n_sites``, ``mean``, ``std``, ``ci_lower``, ``ci_upper``.
        """
        rng = np.random.default_rng(random_state)
        presence = (data.values > 0)
        n_samples = presence.shape[0]

        if n_samples == 0:
            return pd.DataFrame(columns=['n_sites', 'mean', 'std', 'ci_lower', 'ci_upper'])

        curves = np.zeros((n_permutations, n_samples), dtype=float)
        order = np.arange(n_samples)
        for p in range(n_permutations):
            rng.shuffle(order)
            cumulative = np.logical_or.accumulate(presence[order], axis=0)
            curves[p] = cumulative.sum(axis=1)

        return pd.DataFrame({
            'n_sites': np.arange(1, n_samples + 1),
            'mean': curves.mean(axis=0),
            'std': curves.std(axis=0, ddof=1) if n_permutations > 1 else np.zeros(n_samples),
            'ci_lower': np.percentile(curves, 2.5, axis=0),
            'ci_upper': np.percentile(curves, 97.5, axis=0),
        })
