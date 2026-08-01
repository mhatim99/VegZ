"""
Regression tests pinning the scientific behaviour corrected in VegZ 1.4.0.

Each test checks a property that has a known analytical answer, so a future
change that silently reintroduces one of the fixed defects will fail here
rather than quietly producing wrong numbers.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats
from scipy.stats import chi2_contingency

from VegZ.clustering import VegetationClustering
from VegZ.diversity import DiversityAnalyzer
from VegZ.environmental import EnvironmentalModeler
from VegZ.functional_traits import FunctionalTraitAnalyzer
from VegZ.multivariate import MultivariateAnalyzer
from VegZ.nestedness import NestednessAnalyzer, NullModels
from VegZ.spatial import SpatialAnalyzer
from VegZ.specialized_methods import (MetacommunityAnalyzer,
                                      PhylogeneticDiversityAnalyzer)
from VegZ.statistics import EcologicalStatistics
from VegZ.temporal import TemporalAnalyzer
from VegZ.data_management.transformations import DataTransformer
from VegZ import VegZ


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gradient_community():
    """Sites along a linear gradient with Gaussian species responses."""
    n_sites, n_species, span, tolerance = 30, 12, 10.0, 1.5
    gradient = np.linspace(0, span, n_sites)
    values = np.zeros((n_sites, n_species))
    for j in range(n_species):
        optimum = j * span / (n_species - 1)
        values[:, j] = 20 * np.exp(-0.5 * ((gradient - optimum) / tolerance) ** 2)

    data = pd.DataFrame(
        np.round(values, 4),
        index=[f'S{i:02d}' for i in range(n_sites)],
        columns=[f'sp{j:02d}' for j in range(n_species)],
    )
    env = pd.DataFrame({'gradient': gradient}, index=data.index)
    return data, env, span / tolerance


@pytest.fixture
def three_vegetation_types():
    """Three distinct community types, each with its own indicator species."""
    rng = np.random.default_rng(1)
    blocks = []
    for t in range(3):
        block = rng.poisson(0.3, (15, 12)).astype(float)
        block[:, t * 4:(t + 1) * 4] = rng.poisson(12, (15, 4))
        blocks.append(block)

    data = pd.DataFrame(
        np.vstack(blocks),
        index=[f'S{i:03d}' for i in range(45)],
        columns=[f'sp{i:02d}' for i in range(12)],
    )
    groups = pd.Series(np.repeat(['A', 'B', 'C'], 15), index=data.index)
    return data, groups


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------

class TestDiversityCorrectness:

    def test_rarefaction_at_full_size_equals_observed_richness(self):
        """Rarefying to N individuals must return exactly the observed richness."""
        data = pd.DataFrame({'a': [10], 'b': [5], 'c': [2], 'd': [0]}, index=['S1'])
        analyzer = DiversityAnalyzer()
        total = int(data.values.sum())

        curve = analyzer.rarefaction_curve(data, sample_sizes=[total])

        assert curve['expected_species'].iloc[0] == pytest.approx(3.0)
        assert curve['variance'].iloc[0] == pytest.approx(0.0, abs=1e-9)

    def test_rarefaction_is_monotone_increasing(self):
        data = pd.DataFrame({'a': [30], 'b': [20], 'c': [10], 'd': [5]}, index=['S1'])
        curve = DiversityAnalyzer().rarefaction_curve(data, sample_sizes=[5, 10, 25, 50])
        assert curve['expected_species'].is_monotonic_increasing

    def test_ace_uses_i_times_i_minus_one_weighting(self):
        """
        ACE's CV correction uses sum(i*(i-1)*f_i). With sum(i*f_i) the term
        collapses to N_rare and the estimator degenerates to S_rare/C.
        """
        counts = pd.DataFrame([[1, 1, 1, 2, 2, 5, 40]], columns=list('abcdefg'))
        analyzer = DiversityAnalyzer()

        ace = float(analyzer.ace_estimator(counts).iloc[0])
        observed = float(analyzer.species_richness(counts).iloc[0])

        rare = np.array([1, 1, 1, 2, 2, 5], dtype=float)
        n_rare, f1 = rare.sum(), 3.0
        s_rare, s_abund = 6.0, 1.0
        c_ace = 1 - f1 / n_rare
        i_vals = np.arange(1, 11)
        f_i = np.array([np.sum(rare == i) for i in i_vals], dtype=float)
        gamma = max((s_rare / c_ace) * (np.sum(i_vals * (i_vals - 1) * f_i)
                                        / (n_rare * (n_rare - 1))) - 1, 0.0)
        expected = s_abund + s_rare / c_ace + (f1 / c_ace) * gamma

        assert ace == pytest.approx(expected)
        assert ace >= observed

    def test_mcintosh_handles_single_individual(self):
        """N = 1 makes the denominator N - sqrt(N) zero; must not divide by zero."""
        data = pd.DataFrame({'a': [1], 'b': [0]}, index=['S1'])
        value = DiversityAnalyzer().mcintosh_diversity(data).iloc[0]
        assert np.isfinite(value)

    def test_beta_diversity_returns_matrices_for_every_method(self):
        data = pd.DataFrame({'a': [1, 0, 1], 'b': [1, 1, 0], 'c': [0, 1, 1]},
                            index=['S1', 'S2', 'S3'])
        analyzer = DiversityAnalyzer()

        for method in ('whittaker', 'sorensen', 'jaccard'):
            beta = analyzer.beta_diversity(data, method=method)
            assert isinstance(beta, pd.DataFrame)
            assert beta.shape == (3, 3)
            np.testing.assert_allclose(beta.values, beta.values.T)
            np.testing.assert_allclose(np.diag(beta.values), 0)

    def test_whittaker_scalar_still_available(self):
        data = pd.DataFrame({'a': [1, 0], 'b': [0, 1]}, index=['S1', 'S2'])
        # gamma = 2, mean alpha = 1 -> beta = 2
        assert DiversityAnalyzer().whittaker_beta(data) == pytest.approx(2.0)

    def test_gini_simpson_complements_simpson(self):
        data = pd.DataFrame({'a': [5, 1], 'b': [5, 9]}, index=['S1', 'S2'])
        analyzer = DiversityAnalyzer()
        simpson = analyzer.simpson_diversity(data)
        gini = analyzer.gini_simpson(data)
        np.testing.assert_allclose((simpson + gini).values, 1.0)

    def test_hill_numbers_decrease_with_q(self):
        data = pd.DataFrame({'a': [10, 1], 'b': [5, 1], 'c': [1, 1]}, index=['S1', 'S2'])
        hill = DiversityAnalyzer().hill_numbers(data, [0, 1, 2, 3])
        for _, row in hill.iterrows():
            assert np.all(np.diff(row.values) <= 1e-9)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStatisticsCorrectness:

    def test_permanova_reproduces_one_way_anova(self):
        """
        On univariate Euclidean distances PERMANOVA's pseudo-F is identical to
        the classical one-way ANOVA F. This pins the SS decomposition.
        """
        rng = np.random.default_rng(7)
        values = np.concatenate([rng.normal(0, 1, 10),
                                 rng.normal(1.5, 1, 10),
                                 rng.normal(3.0, 1, 10)])
        groups = np.repeat(['A', 'B', 'C'], 10)

        stats_engine = EcologicalStatistics()
        distances = stats_engine.calculate_distance_matrix(
            pd.DataFrame({'x': values}), 'euclidean')
        result = stats_engine.permanova(distances, groups, permutations=99, random_state=0)

        anova = scipy_stats.f_oneway(values[groups == 'A'],
                                     values[groups == 'B'],
                                     values[groups == 'C'])

        assert result['f_statistic'] == pytest.approx(anova.statistic)
        assert result['ss_total'] == pytest.approx(np.sum((values - values.mean()) ** 2))
        assert (result['ss_between'] + result['ss_within']
                == pytest.approx(result['ss_total']))

    def test_permanova_is_reproducible(self):
        rng = np.random.default_rng(3)
        data = pd.DataFrame(rng.random((12, 4)))
        groups = np.repeat(['A', 'B'], 6)
        engine = EcologicalStatistics()
        dm = engine.calculate_distance_matrix(data, 'bray_curtis')

        first = engine.permanova(dm, groups, permutations=99, random_state=11)
        second = engine.permanova(dm, groups, permutations=99, random_state=11)
        assert first['p_value'] == second['p_value']

    def test_anosim_r_reaches_one_for_perfect_separation(self):
        """R must be bounded by 1 and hit it when groups are perfectly separated."""
        engine = EcologicalStatistics()
        separated = pd.DataFrame({'x': [0.0, 0.1, 0.2, 100.0, 100.1, 100.2]})
        dm = engine.calculate_distance_matrix(separated, 'euclidean')

        result = engine.anosim(dm, ['A'] * 3 + ['B'] * 3, permutations=99, random_state=0)
        assert result['r_statistic'] == pytest.approx(1.0)

    def test_anosim_r_within_bounds(self):
        rng = np.random.default_rng(5)
        engine = EcologicalStatistics()
        dm = engine.calculate_distance_matrix(pd.DataFrame(rng.random((15, 3))), 'euclidean')
        r = engine.anosim(dm, np.repeat(['A', 'B', 'C'], 5), permutations=49,
                          random_state=0)['r_statistic']
        assert -1.0 <= r <= 1.0

    def test_indval_is_insensitive_to_group_size(self):
        """
        A species with identical mean abundance in both groups must score
        A = 0.5 each. Summing raw abundances instead of means makes this
        depend on how many sites happen to be in each group.
        """
        data = pd.DataFrame({'sp': [4.0] * 12}, index=range(12))
        groups = ['A'] * 10 + ['B'] * 2

        result = EcologicalStatistics().indicator_species_analysis(
            data, groups, permutations=0)

        details = result['sp']['group_details']
        assert details['A']['relative_abundance'] == pytest.approx(0.5)
        assert details['B']['relative_abundance'] == pytest.approx(0.5)

    def test_indval_perfect_indicator_scores_100(self, three_vegetation_types):
        data, groups = three_vegetation_types
        clean = pd.DataFrame({
            'perfect': [5, 5, 5, 0, 0, 0],
            'ubiquitous': [1, 1, 1, 1, 1, 1],
        }, index=range(6))
        result = EcologicalStatistics().indicator_species_analysis(
            clean, ['A'] * 3 + ['B'] * 3, permutations=0)

        assert result['perfect']['indval'] == pytest.approx(100.0)
        assert result['ubiquitous']['indval'] == pytest.approx(50.0)

    def test_simper_contributions_sum_to_dissimilarity(self):
        rng = np.random.default_rng(2)
        data = pd.DataFrame(rng.integers(0, 20, (10, 5)).astype(float),
                            columns=list('abcde'))
        groups = ['A'] * 5 + ['B'] * 5

        result = EcologicalStatistics().simper_analysis(data, groups)
        block = result['between_A_B']

        assert sum(block['species_contributions'].values()) == pytest.approx(
            block['average_dissimilarity'])
        table = block['species_table']
        assert table['contribution_pct'].sum() == pytest.approx(100.0)
        assert table['cumulative_pct'].iloc[-1] == pytest.approx(100.0)

    def test_mantel_alternative_options(self):
        rng = np.random.default_rng(4)
        engine = EcologicalStatistics()
        a = engine.calculate_distance_matrix(pd.DataFrame(rng.random((10, 3))), 'euclidean')
        b = engine.calculate_distance_matrix(pd.DataFrame(rng.random((10, 3))), 'euclidean')

        for alternative in ('greater', 'less', 'two-sided'):
            result = engine.mantel_test(a, b, permutations=99,
                                        alternative=alternative, random_state=0)
            assert 0 < result['p_value'] <= 1


# ---------------------------------------------------------------------------
# Ordination
# ---------------------------------------------------------------------------

class TestOrdinationCorrectness:

    def test_ca_total_inertia_equals_chi_square_over_n(self, gradient_community):
        data, _, _ = gradient_community
        result = MultivariateAnalyzer().ca_analysis(data)

        expected_chi2 = chi2_contingency(data.values)[0]
        assert result['chi_square'] == pytest.approx(expected_chi2, rel=1e-8)
        # CA eigenvalues are squared canonical correlations and cannot exceed 1.
        assert np.all(result['eigenvalues'] <= 1.0 + 1e-9)

    def test_pcoa_on_euclidean_matches_pca(self, gradient_community):
        data, _, _ = gradient_community
        analyzer = MultivariateAnalyzer()

        pcoa = analyzer.pcoa_analysis(data, distance_metric='euclidean')
        pca = analyzer.pca_analysis(data, transform='none')

        np.testing.assert_allclose(
            np.abs(pcoa['coordinates'].values[:, :3]),
            np.abs(pca['site_scores'].values[:, :3]),
            atol=1e-8,
        )
        assert pcoa['negative_eigenvalue_fraction'] == pytest.approx(0.0, abs=1e-10)

    def test_pcoa_correction_removes_negative_eigenvalues(self, gradient_community):
        data, _, _ = gradient_community
        analyzer = MultivariateAnalyzer()

        uncorrected = analyzer.pcoa_analysis(data, distance_metric='bray_curtis')
        corrected = analyzer.pcoa_analysis(data, distance_metric='bray_curtis',
                                           correction='lingoes')

        assert uncorrected['negative_eigenvalue_fraction'] > 0
        assert corrected['negative_eigenvalue_fraction'] == pytest.approx(0.0, abs=1e-10)

    def test_dca_gradient_length_tracks_species_turnover(self, gradient_community):
        data, _, expected_sd = gradient_community
        lengths = MultivariateAnalyzer().dca_analysis(data)['gradient_lengths']
        # Should be in SD units and near the true turnover, not a constant 4.
        assert lengths[0] == pytest.approx(expected_sd, rel=0.30)

    def test_dca_gradient_length_scales_with_turnover(self):
        analyzer = MultivariateAnalyzer()

        def build(span, tolerance, n_sites=40, n_species=15):
            gradient = np.linspace(0, span, n_sites)
            values = np.zeros((n_sites, n_species))
            for j in range(n_species):
                optimum = j * span / (n_species - 1)
                values[:, j] = 20 * np.exp(-0.5 * ((gradient - optimum) / tolerance) ** 2)
            return pd.DataFrame(np.round(values, 4))

        short = analyzer.dca_analysis(build(10, 2.5))['gradient_lengths'][0]
        long = analyzer.dca_analysis(build(20, 1.5))['gradient_lengths'][0]
        assert long > short

    def test_rda_variance_partition_adds_up(self, gradient_community):
        data, env, _ = gradient_community
        result = MultivariateAnalyzer().rda_analysis(data, env)

        assert (result['constrained_variance'] + result['unconstrained_variance']
                == pytest.approx(result['total_variance']))
        assert 0 <= result['proportion_constrained'] <= 1

    def test_cca_recovers_the_constraining_gradient(self, gradient_community):
        data, env, _ = gradient_community
        result = MultivariateAnalyzer().cca_analysis(data, env)

        # One constraint -> at most one constrained axis.
        assert len(result['eigenvalues']) <= env.shape[1]
        # The single constraint must load almost perfectly on axis 1.
        assert abs(result['env_scores'].loc['gradient', 'CCA1']) > 0.9
        assert 0 < result['proportion_constrained'] <= 1

    def test_procrustes_is_importable_and_exact_for_identical_configs(self):
        """scipy.spatial.procrustes - importing it from .distance raises ImportError."""
        rng = np.random.default_rng(0)
        config = rng.random((10, 2))
        result = MultivariateAnalyzer().procrustes_analysis(config, config.copy(),
                                                            permutations=0)
        assert result['m12_squared'] == pytest.approx(0.0, abs=1e-12)
        assert result['correlation'] == pytest.approx(1.0)

    def test_envfit_vectors_are_unit_length_and_detect_signal(self, gradient_community):
        data, env, _ = gradient_community
        rng = np.random.default_rng(0)
        env = env.assign(noise=rng.normal(0, 1, len(env)))

        analyzer = MultivariateAnalyzer()
        scores = analyzer.pca_analysis(data)['site_scores'].iloc[:, :2]
        fit = analyzer.environmental_fitting(scores, env, permutations=199, random_state=0)

        assert 'vectors' in fit
        for entry in fit['vectors'].values():
            assert np.linalg.norm(entry['direction']) == pytest.approx(1.0)

        assert fit['vectors']['gradient']['r_squared'] > 0.5
        assert fit['vectors']['gradient']['p_value'] < 0.05
        assert fit['vectors']['noise']['p_value'] > 0.05

    def test_nmds_is_non_metric(self, gradient_community):
        """NMDS must run *non*-metric MDS, not metric MDS."""
        data, _, _ = gradient_community
        result = MultivariateAnalyzer().nmds_analysis(data, n_dimensions=2)
        mds = result['mds_object']
        flag = getattr(mds, 'metric_mds', getattr(mds, 'metric', None))
        assert flag is False

    def test_core_nmds_is_non_metric(self, gradient_community):
        data, _, _ = gradient_community
        veg = VegZ()
        veg.species_matrix = data
        mds = veg.nmds_analysis()['mds_object']
        flag = getattr(mds, 'metric_mds', getattr(mds, 'metric', None))
        assert flag is False


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

class TestClusteringCorrectness:

    def test_elbow_inertia_curve_is_monotone_decreasing(self, three_vegetation_types):
        """
        The k=1 point must be the total sum of squares, on the same scale as
        KMeans inertia; using a mean variance breaks monotonicity.
        """
        data, _ = three_vegetation_types
        result = VegetationClustering().comprehensive_elbow_analysis(
            data, k_range=range(1, 8), plot_results=False)

        inertias = result['metrics']['inertia']
        assert np.all(np.diff(inertias) <= 1e-9)

    def test_elbow_analysis_works_when_k_range_excludes_one(self, three_vegetation_types):
        data, _ = three_vegetation_types
        result = VegetationClustering().comprehensive_elbow_analysis(
            data, k_range=range(2, 7), plot_results=False)

        assert result['k_values'] == list(range(2, 7))
        assert result['recommendations']['silhouette_optimal'] in result['k_values']

    def test_twinspan_recovers_known_vegetation_types(self, three_vegetation_types):
        """TWINSPAN must actually divide; it previously returned one group always."""
        from sklearn.metrics import adjusted_rand_score

        data, groups = three_vegetation_types
        result = VegetationClustering().twinspan(data, min_group_size=4, max_divisions=2)

        assert result['n_divisions'] == 2
        assert result['n_groups'] == 3
        assert adjusted_rand_score(groups.values,
                                   result['site_classification'].values) == pytest.approx(1.0)

    def test_twinspan_assigns_every_site(self, three_vegetation_types):
        data, _ = three_vegetation_types
        classification = VegetationClustering().twinspan(
            data, min_group_size=4)['site_classification']
        assert classification.dtype.kind in 'iu'
        assert (classification > 0).all()

    def test_kmeans_exposes_both_centroid_keys(self, three_vegetation_types):
        data, _ = three_vegetation_types
        result = VegetationClustering().kmeans_clustering(data, n_clusters=3)
        assert 'centroids' in result and 'cluster_centers' in result

    def test_hierarchical_accepts_distance_metric_alias(self, three_vegetation_types):
        data, _ = three_vegetation_types
        result = VegetationClustering().hierarchical_clustering(
            data, n_clusters=3, distance_metric='euclidean', linkage_method='average')
        assert 'cluster_labels' in result
        assert -1 <= result['cophenetic_correlation'] <= 1

    def test_cophenetic_uses_the_clustering_distances(self, three_vegetation_types):
        """A Bray-Curtis tree must be validated against Bray-Curtis distances."""
        from scipy.cluster.hierarchy import cophenet

        data, _ = three_vegetation_types
        result = VegetationClustering().hierarchical_clustering(
            data, n_clusters=3, distance_metric='bray_curtis', linkage_method='average')

        expected, _ = cophenet(result['linkage_matrix'], result['distances'])
        assert result['cophenetic_correlation'] == pytest.approx(expected)

    def test_fuzzy_cmeans_is_reproducible(self, three_vegetation_types):
        data, _ = three_vegetation_types
        clustering = VegetationClustering()
        first = clustering.fuzzy_cmeans_clustering(data, 3, random_state=7)
        second = clustering.fuzzy_cmeans_clustering(data, 3, random_state=7)
        pd.testing.assert_frame_equal(first['membership_matrix'],
                                      second['membership_matrix'])

    def test_gap_statistic_is_reproducible(self, three_vegetation_types):
        data, _ = three_vegetation_types
        clustering = VegetationClustering()
        first = clustering._gap_statistic(data, range(1, 5), n_refs=5, random_state=3)
        second = clustering._gap_statistic(data, range(1, 5), n_refs=5, random_state=3)
        assert first['gap_values'] == second['gap_values']

    def test_distortion_jump_finds_a_clear_elbow(self):
        clustering = VegetationClustering()
        assert clustering._distortion_jump_method(
            [1, 2, 3, 4, 5, 6], [100, 60, 20, 18, 17, 16]) == 3


# ---------------------------------------------------------------------------
# Nestedness
# ---------------------------------------------------------------------------

class TestNestednessCorrectness:

    @staticmethod
    def _perfectly_nested(n=8):
        matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            matrix[i, :n - i] = 1
        return pd.DataFrame(matrix,
                            index=[f'S{i}' for i in range(n)],
                            columns=[f'sp{i}' for i in range(n)])

    def test_nodf_is_100_for_perfectly_nested_matrix(self):
        analyzer = NestednessAnalyzer()
        analyzer.load_matrix(self._perfectly_nested())
        result = analyzer.calculate_nestedness_metrics(metrics=['nodf'])
        assert result['metrics']['nodf']['nodf_overall'] == pytest.approx(100.0)

    def test_nodf_is_zero_when_no_fill_gradient_exists(self):
        """Identical rows have no decreasing fill, so NODF must be 0."""
        data = pd.DataFrame(np.tile([1, 1, 1, 0, 0, 0], (6, 1)),
                            index=[f'S{i}' for i in range(6)],
                            columns=[f'sp{i}' for i in range(6)])
        analyzer = NestednessAnalyzer()
        analyzer.load_matrix(data)
        result = analyzer.calculate_nestedness_metrics(metrics=['nodf'])
        assert result['metrics']['nodf']['nodf_overall'] == pytest.approx(0.0)

    def test_nodf_stays_within_bounds(self):
        rng = np.random.default_rng(0)
        data = pd.DataFrame(rng.integers(0, 2, (12, 12)))
        analyzer = NestednessAnalyzer()
        analyzer.load_matrix(data)
        value = analyzer.calculate_nestedness_metrics(
            metrics=['nodf'])['metrics']['nodf']['nodf_overall']
        assert 0 <= value <= 100

    def test_temperature_is_zero_for_perfectly_nested_matrix(self):
        analyzer = NestednessAnalyzer()
        analyzer.load_matrix(self._perfectly_nested())
        result = analyzer.calculate_nestedness_metrics(metrics=['temperature'])
        assert result['metrics']['temperature']['temperature'] == pytest.approx(0.0)

    def test_constructing_analyzers_does_not_touch_global_rng(self):
        np.random.seed(123)
        expected = np.random.rand()

        np.random.seed(123)
        NestednessAnalyzer(random_state=7)
        NullModels(random_state=9)
        actual = np.random.rand()

        assert expected == actual


# ---------------------------------------------------------------------------
# Functional traits
# ---------------------------------------------------------------------------

class TestFunctionalTraitCorrectness:

    def test_feve_is_one_for_a_perfectly_even_community(self):
        traits = pd.DataFrame({'species': list('abcd'), 't1': [0.0, 1.0, 2.0, 3.0]})
        even = pd.DataFrame([[1.0, 1.0, 1.0, 1.0]], index=['even'], columns=list('abcd'))

        analyzer = FunctionalTraitAnalyzer()
        analyzer.load_trait_data(traits, even, 'species')
        feve = analyzer.calculate_functional_diversity()['site_diversity'].loc['even', 'FEve']

        assert feve == pytest.approx(1.0)

    def test_feve_is_lower_for_a_dominated_community(self):
        traits = pd.DataFrame({'species': list('abcd'), 't1': [0.0, 1.0, 2.0, 3.0]})
        even = pd.DataFrame([[1.0, 1.0, 1.0, 1.0]], index=['x'], columns=list('abcd'))
        uneven = pd.DataFrame([[100.0, 1.0, 1.0, 1.0]], index=['x'], columns=list('abcd'))

        def feve(abundance):
            analyzer = FunctionalTraitAnalyzer()
            analyzer.load_trait_data(traits, abundance, 'species')
            return analyzer.calculate_functional_diversity()['site_diversity'].loc['x', 'FEve']

        assert feve(uneven) < feve(even)

    def test_fd_indices_stay_within_bounds(self):
        rng = np.random.default_rng(0)
        names = [f'sp{i:02d}' for i in range(10)]
        traits = pd.DataFrame({'species': names,
                               'sla': rng.uniform(5, 30, 10),
                               'height': rng.uniform(0.1, 20, 10)})
        abundance = pd.DataFrame(rng.poisson(4, (12, 10)).astype(float),
                                 index=[f'S{i}' for i in range(12)], columns=names)

        analyzer = FunctionalTraitAnalyzer()
        analyzer.load_trait_data(traits, abundance, 'species')
        site_fd = analyzer.calculate_functional_diversity()['site_diversity']

        assert site_fd['FEve'].between(0, 1).all()
        assert site_fd['FDiv'].between(0, 1).all()
        assert (site_fd['FRic'] >= 0).all()
        assert (site_fd['RaoQ'] >= 0).all()

    def test_species_alignment_is_deterministic(self):
        """set() intersection ordering varies between runs; ours must not."""
        rng = np.random.default_rng(0)
        names = [f'sp{i:02d}' for i in range(8)]
        traits = pd.DataFrame({'species': names, 't': rng.uniform(0, 1, 8)})
        abundance = pd.DataFrame(rng.poisson(3, (5, 8)).astype(float),
                                 index=[f'S{i}' for i in range(5)], columns=names)

        orders = []
        for _ in range(3):
            analyzer = FunctionalTraitAnalyzer()
            analyzer.load_trait_data(traits.copy(), abundance.copy(), 'species')
            orders.append(list(analyzer.trait_data.index))

        assert orders[0] == orders[1] == orders[2] == names


# ---------------------------------------------------------------------------
# Spatial
# ---------------------------------------------------------------------------

class TestSpatialCorrectness:

    @staticmethod
    def _grid(values):
        coords = np.array([[i, j] for i in range(10) for j in range(10)], dtype=float)
        return pd.DataFrame({'longitude': coords[:, 0],
                             'latitude': coords[:, 1],
                             'response': values})

    def test_morans_i_detects_a_smooth_field(self):
        coords = np.array([[i, j] for i in range(10) for j in range(10)], dtype=float)
        smooth = coords[:, 0] + coords[:, 1]
        result = SpatialAnalyzer().spatial_autocorrelation(self._grid(smooth),
                                                           method='morans_i')
        assert result['morans_i'] > 0.3
        assert result['p_value'] < 0.01

    def test_morans_i_is_near_expectation_for_noise(self):
        rng = np.random.default_rng(0)
        result = SpatialAnalyzer().spatial_autocorrelation(
            self._grid(rng.normal(0, 1, 100)), method='morans_i')
        assert abs(result['morans_i'] - result['expected_i']) < 0.1
        assert result['p_value'] > 0.05

    def test_gearys_c_is_near_one_for_noise(self):
        """Geary's C has an expected value of exactly 1 under no autocorrelation."""
        rng = np.random.default_rng(0)
        result = SpatialAnalyzer().spatial_autocorrelation(
            self._grid(rng.normal(0, 1, 100)), method='gearys_c')
        assert result['gearys_c'] == pytest.approx(1.0, abs=0.15)

    def test_gearys_c_below_one_for_positive_autocorrelation(self):
        coords = np.array([[i, j] for i in range(10) for j in range(10)], dtype=float)
        result = SpatialAnalyzer().spatial_autocorrelation(
            self._grid(coords[:, 0] + coords[:, 1]), method='gearys_c')
        assert result['gearys_c'] < 1.0

    def test_landscape_shape_index_is_one_for_a_square(self):
        analyzer = SpatialAnalyzer()
        square = np.ones((20, 20), dtype=int)
        assert analyzer._landscape_shape_index(square, square, 1.0) == pytest.approx(1.0)

    def test_total_edge_length_of_a_square(self):
        analyzer = SpatialAnalyzer()
        square = np.ones((20, 20), dtype=int)
        assert analyzer._total_edge_length(square, 1.0) == pytest.approx(80.0)

    def test_contagion_bounds(self):
        analyzer = SpatialAnalyzer()
        rng = np.random.default_rng(0)
        single = np.zeros((40, 40), dtype=int)
        random_field = rng.integers(0, 2, (40, 40))

        assert analyzer._contagion_index(single, single, 1.0) == pytest.approx(100.0)
        assert analyzer._contagion_index(random_field, random_field, 1.0) < 5.0

    def test_interpolation_rejects_absurd_grid_sizes(self):
        coords = np.array([[i, j] for i in range(10) for j in range(10)], dtype=float)
        data = self._grid(coords[:, 0])
        with pytest.raises(ValueError, match='grid cells'):
            SpatialAnalyzer().spatial_interpolation(data, grid_resolution=0.0001)


# ---------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------

class TestTemporalCorrectness:

    @staticmethod
    def _series():
        rng = np.random.default_rng(2)
        dates = pd.date_range('2015-01-31', periods=72, freq='ME')
        t = np.arange(72)
        values = 10 + 0.08 * t + rng.normal(0, 0.6, 72)
        return pd.DataFrame({'date': dates, 'response': values})

    def test_mann_kendall_is_independent_of_row_order(self):
        """MK assumes time order; shuffling the input rows must not change it."""
        analyzer = TemporalAnalyzer()
        data = self._series()

        ordered = analyzer.trend_detection(data, 'date', 'response', method='mann_kendall')
        shuffled = analyzer.trend_detection(data.sample(frac=1, random_state=0),
                                            'date', 'response', method='mann_kendall')

        assert ordered['s_statistic'] == shuffled['s_statistic']
        assert ordered['p_value'] == shuffled['p_value']

    def test_mann_kendall_detects_a_real_trend(self):
        result = TemporalAnalyzer().trend_detection(
            self._series(), 'date', 'response', method='mann_kendall')
        assert result['trend'] == 'increasing'
        assert result['p_value'] < 0.01
        low, high = result['sens_slope_ci']
        assert low <= result['sens_slope'] <= high

    def test_spline_trend_handles_unsorted_input(self):
        data = self._series().sample(frac=1, random_state=1)
        result = TemporalAnalyzer().trend_detection(data, 'date', 'response', method='spline')
        assert np.isfinite(result['predicted_values']).all()

    def test_seasonality_detection_distinguishes_quarterly_from_daily(self):
        analyzer = TemporalAnalyzer()
        quarterly = pd.Series(np.arange(20),
                              index=pd.date_range('2015-03-31', periods=20, freq='QE'))
        daily = pd.Series(np.arange(30),
                          index=pd.date_range('2015-01-01', periods=30, freq='D'))

        assert analyzer._detect_seasonality(quarterly) == 4
        assert analyzer._detect_seasonality(daily) == 365

    def test_double_sigmoid_recovers_a_growing_season(self):
        """The senescence term must be an increasing logistic that is subtracted."""
        rng = np.random.default_rng(2)
        doy = np.arange(1, 366, 5)
        signal = (25 / (1 + np.exp(-0.1 * (doy - 120)))
                  - 25 / (1 + np.exp(-0.1 * (doy - 260))))
        data = pd.DataFrame({'doy': doy, 'resp': signal + rng.normal(0, 0.8, len(doy))})

        fit = TemporalAnalyzer().phenology_modeling(
            data, 'doy', 'resp', model_type='double_sigmoid')['results']['combined']

        assert fit['success']
        assert fit['r_squared'] > 0.95
        params = fit['phenological_parameters']
        assert params['green_up'] == pytest.approx(120, abs=10)
        assert params['senescence'] == pytest.approx(260, abs=10)

    def test_phenology_models_all_converge(self):
        rng = np.random.default_rng(2)
        doy = np.arange(1, 366, 5)
        resp = 30 * np.exp(-0.5 * ((doy - 180) / 40) ** 2) + rng.normal(0, 1, len(doy))
        data = pd.DataFrame({'doy': doy, 'resp': resp})

        analyzer = TemporalAnalyzer()
        for model in ('sigmoid', 'gaussian', 'double_sigmoid', 'beta', 'weibull'):
            fit = analyzer.phenology_modeling(
                data, 'doy', 'resp', model_type=model)['results']['combined']
            assert fit['success'], f"{model} failed to converge"

    def test_gaussian_phenology_recovers_peak_day(self):
        rng = np.random.default_rng(2)
        doy = np.arange(1, 366, 5)
        resp = 30 * np.exp(-0.5 * ((doy - 180) / 40) ** 2) + rng.normal(0, 1, len(doy))
        fit = TemporalAnalyzer().phenology_modeling(
            pd.DataFrame({'doy': doy, 'resp': resp}),
            'doy', 'resp', model_type='gaussian')['results']['combined']
        assert fit['phenological_parameters']['peak_time'] == pytest.approx(180, abs=5)


# ---------------------------------------------------------------------------
# Environmental modelling
# ---------------------------------------------------------------------------

class TestEnvironmentalCorrectness:

    def test_smoothers_handle_unsorted_integer_predictors(self):
        """UnivariateSpline needs increasing x; fitted values must stay float."""
        rng = np.random.default_rng(0)
        x = rng.permutation(np.arange(0, 60))
        y = 3.0 + 0.05 * x ** 1.5 + rng.normal(0, 2, 60)
        data = pd.DataFrame({'x': x, 'y': y})

        modeler = EnvironmentalModeler()
        for smoother in ('spline', 'lowess', 'polynomial', 'gaussian_process'):
            result = modeler.fit_gam(data, 'y', ['x'], smoother_types={'x': smoother})
            assert result['fitted_values'].dtype.kind == 'f'
            assert np.isfinite(result['fitted_values']).all()

    def test_smoothers_do_not_interpolate_noise(self):
        rng = np.random.default_rng(0)
        x = np.linspace(0, 10, 60)
        truth = 3.0 + 0.5 * x ** 1.5
        y = truth + rng.normal(0, 2, 60)

        modeler = EnvironmentalModeler()
        fitted = modeler._spline_smoother(x, y)['fitted_values']

        # Smoothing must recover the signal better than the raw observations do.
        assert np.mean((fitted - truth) ** 2) < np.mean((y - truth) ** 2)
        # And it must not simply interpolate every observation.
        assert np.corrcoef(fitted, y)[0, 1] ** 2 < 0.999

    def test_unknown_gam_family_raises(self):
        rng = np.random.default_rng(0)
        data = pd.DataFrame({'x': np.arange(20.0), 'y': rng.normal(0, 1, 20)})
        with pytest.raises(ValueError, match='Unknown family'):
            EnvironmentalModeler().fit_gam(data, 'y', ['x'], family='not-a-family')

    def test_gaussian_response_curve_recovers_parameters(self):
        rng = np.random.default_rng(0)
        gradient = np.linspace(0, 20, 60)
        abundance = (30 * np.exp(-0.5 * ((gradient - 12) / 3) ** 2) + 1
                     + rng.normal(0, 0.6, 60))

        result = EnvironmentalModeler().species_response_curves(
            pd.Series(abundance), pd.Series(gradient), 'gaussian')

        assert result['success']
        params = result['ecological_parameters']
        assert params['optimum'] == pytest.approx(12.0, abs=0.5)
        assert params['tolerance'] == pytest.approx(3.0, abs=0.5)


# ---------------------------------------------------------------------------
# Data transformations
# ---------------------------------------------------------------------------

class TestTransformationCorrectness:

    @pytest.fixture
    def frame(self):
        rng = np.random.default_rng(0)
        return pd.DataFrame(rng.integers(0, 20, (6, 5)).astype(float), columns=list('abcde'))

    def test_every_transformation_accepts_dataframes_and_arrays(self, frame):
        transformer = DataTransformer()
        for method in transformer.transformation_methods:
            assert transformer.transform(frame, method) is not None
            assert transformer.transform(frame.values, method) is not None

    def test_chord_rows_have_unit_norm(self, frame):
        result = DataTransformer().chord_transform(frame)
        np.testing.assert_allclose(np.sqrt((result.values ** 2).sum(axis=1)), 1.0)

    def test_hellinger_rows_have_unit_squared_norm(self, frame):
        result = DataTransformer().hellinger_transform(frame)
        np.testing.assert_allclose((result.values ** 2).sum(axis=1), 1.0)

    def test_chi_square_transform_yields_chi_square_distances(self, frame):
        """Euclidean distance on the transformed table == chi-square distance."""
        transformed = DataTransformer().chi_square_transform(frame).values
        X = frame.values
        grand_total, row_sums, col_sums = X.sum(), X.sum(1), X.sum(0)

        for i, j in [(0, 1), (2, 4), (3, 5)]:
            expected = np.sqrt(np.sum(
                (X[i] / row_sums[i] - X[j] / row_sums[j]) ** 2 / (col_sums / grand_total)))
            assert np.linalg.norm(transformed[i] - transformed[j]) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Specialized methods
# ---------------------------------------------------------------------------

class TestSpecializedCorrectness:

    def test_mntd_does_not_mutate_the_phylogeny(self):
        rng = np.random.default_rng(0)
        names = [f'sp{i}' for i in range(6)]
        matrix = rng.uniform(0, 1, (6, 6))
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 0)
        phylo = pd.DataFrame(matrix, index=names, columns=names)
        before = phylo.values.copy()

        analyzer = PhylogeneticDiversityAnalyzer()
        analyzer.load_phylogeny(phylo)
        analyzer.calculate_phylogenetic_diversity(
            pd.DataFrame(rng.integers(0, 4, (5, 6)), columns=names))

        np.testing.assert_array_equal(before, phylo.values)

    def test_metacommunity_clumped_structure_is_reachable(self):
        """
        The old range-size ratio could never fall below 1, making the 'clumped'
        classification unreachable; Morisita's index can.
        """
        blocks = np.zeros((12, 12), dtype=int)
        for b in range(3):
            blocks[b * 4:(b + 1) * 4, b * 4:(b + 1) * 4] = 1
        data = pd.DataFrame(blocks,
                            index=[f'S{i}' for i in range(12)],
                            columns=[f'sp{i}' for i in range(12)])

        result = MetacommunityAnalyzer().elements_of_metacommunity_structure(data)
        assert result['structure_type'] == 'clumped'
        assert result['boundary_clumping']['morisita_index'] > 1

    def test_metacommunity_nested_structure(self):
        nested = np.zeros((10, 10), dtype=int)
        for i in range(10):
            nested[i, :10 - i] = 1
        data = pd.DataFrame(nested,
                            index=[f'S{i}' for i in range(10)],
                            columns=[f'sp{i}' for i in range(10)])

        result = MetacommunityAnalyzer().elements_of_metacommunity_structure(data)
        assert result['structure_type'] == 'nested'


# ---------------------------------------------------------------------------
# Package hygiene
# ---------------------------------------------------------------------------

class TestPackageHygiene:

    def test_importing_vegz_emits_no_warnings(self):
        """A library must not warn merely because optional extras are absent."""
        import subprocess
        import sys

        completed = subprocess.run(
            [sys.executable, '-W', 'error::UserWarning', '-c', 'import VegZ'],
            capture_output=True, text=True,
        )
        assert completed.returncode == 0, completed.stderr

    def test_version_is_consistent(self):
        """
        The version is declared in three places. Read the one in pyproject.toml
        - the one that actually reaches PyPI - and check the others agree,
        rather than hard-coding a fourth copy that has to be edited each
        release and can itself drift.
        """
        import pathlib
        import re
        import VegZ

        pyproject = (pathlib.Path(__file__).resolve().parents[1]
                     / 'pyproject.toml').read_text(encoding='utf-8')
        match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M)
        assert match, 'no version found in pyproject.toml'
        declared = match.group(1)

        assert VegZ.__version__ == declared
        info = VegZ.get_version_info()
        assert info['version'] == declared
        major, minor, patch = (int(part) for part in declared.split('.'))
        assert (info['major'], info['minor'], info['patch']) == (major, minor, patch)

    def test_optional_dependency_flags_exist(self):
        from VegZ.data_management import standardization
        from VegZ import spatial, specialized_methods

        assert isinstance(standardization.FUZZYWUZZY_AVAILABLE, bool)
        assert isinstance(spatial.SPATIAL_LIBS_AVAILABLE, bool)
        assert isinstance(specialized_methods.NETWORKX_AVAILABLE, bool)

    def test_fuzzy_fallback_matches_identical_strings(self):
        from VegZ.data_management.standardization import _DifflibFuzz

        assert _DifflibFuzz.ratio('Quercus alba', 'Quercus alba') == 100
        assert _DifflibFuzz.ratio('Quercus alba', 'Quercus rubra') < 100
        assert _DifflibFuzz.token_sort_ratio('alba Quercus', 'Quercus alba') == 100
