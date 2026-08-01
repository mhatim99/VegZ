"""
Tests for the methods added in VegZ 1.5.0.

Each is checked against a result that is known independently of this code:
PERMDISP against Levene's test, adonis against classical ANOVA sums of squares,
anova_rda against the regression F, Baselga partitions against their defining
identities, and the Hill-number machinery against its own limiting cases.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from VegZ.diversity import DiversityAnalyzer
from VegZ.multivariate import MultivariateAnalyzer
from VegZ.statistics import EcologicalStatistics


@pytest.fixture
def gradient_data():
    """
    Sites along a linear gradient with Gaussian species responses.

    Observation noise is included deliberately. A noiseless community has
    essentially zero residual variance, which makes even a random predictor
    look significant and turns any p-value assertion into a fixture artefact.
    """
    rng = np.random.default_rng(0)
    n_sites, n_species, span, tolerance = 30, 12, 10.0, 1.5
    gradient = np.linspace(0, span, n_sites)

    values = np.zeros((n_sites, n_species))
    for j in range(n_species):
        optimum = j * span / (n_species - 1)
        values[:, j] = 20 * np.exp(-0.5 * ((gradient - optimum) / tolerance) ** 2)
    values = np.clip(values + rng.normal(0, 2.0, values.shape), 0, None)

    species = pd.DataFrame(np.round(values, 4),
                           index=[f'S{i:02d}' for i in range(n_sites)],
                           columns=[f'sp{j:02d}' for j in range(n_species)])
    environment = pd.DataFrame(
        {'gradient': gradient, 'noise': rng.normal(0, 1, n_sites)},
        index=species.index)
    return species, environment


# ---------------------------------------------------------------------------
# PERMDISP
# ---------------------------------------------------------------------------

class TestPermdisp:

    def test_matches_levene_on_univariate_euclidean_data(self):
        """
        With one variable and Euclidean distances, distance-to-centroid is
        |x - mean|, so PERMDISP reduces exactly to Levene's test.
        """
        rng = np.random.default_rng(11)
        values = np.concatenate([rng.normal(0, 1, 12), rng.normal(0, 3, 12),
                                 rng.normal(0, 1, 12)])
        groups = np.repeat(['A', 'B', 'C'], 12)

        engine = EcologicalStatistics()
        distances = engine.calculate_distance_matrix(
            pd.DataFrame({'x': values}), 'euclidean')
        result = engine.permdisp(distances, groups, permutations=99, random_state=0)

        levene = scipy_stats.levene(values[groups == 'A'], values[groups == 'B'],
                                    values[groups == 'C'], center='mean')
        assert result['f_statistic'] == pytest.approx(levene.statistic)

    def test_equal_dispersion_is_not_significant(self):
        rng = np.random.default_rng(3)
        engine = EcologicalStatistics()
        distances = engine.calculate_distance_matrix(
            pd.DataFrame({'x': rng.normal(0, 1, 36)}), 'euclidean')
        result = engine.permdisp(distances, np.repeat(['A', 'B', 'C'], 12),
                                 permutations=299, random_state=0)
        assert result['p_value'] > 0.05

    def test_unequal_dispersion_is_detected(self):
        rng = np.random.default_rng(5)
        values = np.concatenate([rng.normal(0, 0.2, 20), rng.normal(0, 5.0, 20)])
        engine = EcologicalStatistics()
        distances = engine.calculate_distance_matrix(
            pd.DataFrame({'x': values}), 'euclidean')
        result = engine.permdisp(distances, np.repeat(['A', 'B'], 20),
                                 permutations=299, random_state=0)
        assert result['p_value'] < 0.05

    def test_group_dispersions_track_the_true_spread(self):
        rng = np.random.default_rng(7)
        values = np.concatenate([rng.normal(0, 1, 40), rng.normal(0, 4, 40)])
        engine = EcologicalStatistics()
        distances = engine.calculate_distance_matrix(
            pd.DataFrame({'x': values}), 'euclidean')
        result = engine.permdisp(distances, np.repeat(['tight', 'wide'], 40),
                                 permutations=99, random_state=0)

        dispersions = result['group_dispersions']
        assert (dispersions['wide']['mean_distance_to_centroid']
                > dispersions['tight']['mean_distance_to_centroid'])

    def test_spatial_median_option(self, gradient_data):
        species, _ = gradient_data
        engine = EcologicalStatistics()
        distances = engine.calculate_distance_matrix(species, 'bray_curtis')
        result = engine.permdisp(distances, np.repeat(['A', 'B', 'C'], 10),
                                 permutations=49, centroid_type='median',
                                 random_state=0)
        assert result['centroid_type'] == 'median'
        assert (result['distances_to_centroid'] >= 0).all()

    def test_pairwise_comparisons_are_holm_adjusted(self, gradient_data):
        species, _ = gradient_data
        engine = EcologicalStatistics()
        distances = engine.calculate_distance_matrix(species, 'bray_curtis')
        result = engine.permdisp(distances, np.repeat(['A', 'B', 'C'], 10),
                                 permutations=49, pairwise=True, random_state=0)

        table = result['pairwise']
        assert len(table) == 3
        assert (table['p_value_holm'] >= table['p_value']).all()

    def test_betadisper_is_an_alias(self, gradient_data):
        species, _ = gradient_data
        engine = EcologicalStatistics()
        distances = engine.calculate_distance_matrix(species, 'bray_curtis')
        groups = np.repeat(['A', 'B', 'C'], 10)
        assert (engine.betadisper(distances, groups, permutations=49, random_state=1)
                ['f_statistic']
                == engine.permdisp(distances, groups, permutations=49, random_state=1)
                ['f_statistic'])

    def test_single_group_rejected(self, gradient_data):
        species, _ = gradient_data
        engine = EcologicalStatistics()
        distances = engine.calculate_distance_matrix(species, 'bray_curtis')
        with pytest.raises(ValueError, match='at least two groups'):
            engine.permdisp(distances, ['A'] * 30)

    def test_bad_centroid_type_rejected(self, gradient_data):
        species, _ = gradient_data
        engine = EcologicalStatistics()
        distances = engine.calculate_distance_matrix(species, 'bray_curtis')
        with pytest.raises(ValueError, match='centroid_type'):
            engine.permdisp(distances, np.repeat(['A', 'B', 'C'], 10),
                            centroid_type='barycentre')


# ---------------------------------------------------------------------------
# Multi-factor PERMANOVA
# ---------------------------------------------------------------------------

class TestAdonis:

    def test_one_way_reproduces_permanova(self):
        rng = np.random.default_rng(3)
        data = pd.DataFrame(rng.random((24, 6)))
        groups = np.repeat(['A', 'B', 'C'], 8)

        engine = EcologicalStatistics()
        distances = engine.calculate_distance_matrix(data, 'bray_curtis')
        permanova = engine.permanova(distances, groups, permutations=99, random_state=1)
        adonis = engine.adonis(distances, pd.DataFrame({'grp': groups}),
                               permutations=99, random_state=1)

        table = adonis['table']
        assert table.loc['grp', 'F'] == pytest.approx(permanova['f_statistic'])
        assert table.loc['grp', 'SumOfSqs'] == pytest.approx(permanova['ss_between'])
        assert table.loc['Total', 'SumOfSqs'] == pytest.approx(permanova['ss_total'])

    def test_two_way_matches_classical_sequential_anova(self):
        """Univariate Euclidean distances must give the textbook Type I SS."""
        rng = np.random.default_rng(5)
        a = np.repeat(['a1', 'a2'], 12)
        b = np.tile(np.repeat(['b1', 'b2'], 6), 2)
        y = (a == 'a2') * 3.0 + (b == 'b2') * 1.5 + rng.normal(0, 1, 24)

        engine = EcologicalStatistics()
        distances = engine.calculate_distance_matrix(pd.DataFrame({'y': y}), 'euclidean')
        table = engine.adonis(distances, pd.DataFrame({'A': a, 'B': b}),
                              terms=['A*B'], permutations=49, random_state=0)['table']

        def explained(columns):
            design = np.column_stack(columns)
            design = design - design.mean(axis=0)
            hat = design @ np.linalg.pinv(design.T @ design) @ design.T
            centred = y - y.mean()
            return float(centred @ hat @ centred)

        col_a = (a == 'a2').astype(float)
        col_b = (b == 'b2').astype(float)
        col_ab = col_a * col_b

        assert table.loc['A', 'SumOfSqs'] == pytest.approx(explained([col_a]))
        assert table.loc['B', 'SumOfSqs'] == pytest.approx(
            explained([col_a, col_b]) - explained([col_a]))
        assert table.loc['A:B', 'SumOfSqs'] == pytest.approx(
            explained([col_a, col_b, col_ab]) - explained([col_a, col_b]))

    def test_sums_of_squares_partition_exactly(self, gradient_data):
        species, environment = gradient_data
        engine = EcologicalStatistics()
        distances = engine.calculate_distance_matrix(species, 'bray_curtis')
        design = environment.assign(grp=np.repeat(['A', 'B', 'C'], 10))

        table = engine.adonis(distances, design, terms=['grp', 'gradient'],
                              permutations=49, random_state=0)['table']
        parts = table.drop(index='Total')
        assert parts['SumOfSqs'].sum() == pytest.approx(table.loc['Total', 'SumOfSqs'])
        assert parts['R2'].sum() == pytest.approx(1.0)

    def test_marginal_and_sequential_differ_for_correlated_terms(self):
        rng = np.random.default_rng(9)
        x1 = rng.normal(0, 1, 40)
        x2 = x1 * 0.8 + rng.normal(0, 0.4, 40)
        y = 2 * x1 + rng.normal(0, 1, 40)

        engine = EcologicalStatistics()
        distances = engine.calculate_distance_matrix(pd.DataFrame({'y': y}), 'euclidean')
        design = pd.DataFrame({'x1': x1, 'x2': x2})

        sequential = engine.adonis(distances, design, by='terms',
                                   permutations=49, random_state=0)['table']
        marginal = engine.adonis(distances, design, by='margin',
                                 permutations=49, random_state=0)['table']

        assert sequential.loc['x1', 'SumOfSqs'] != pytest.approx(
            marginal.loc['x1', 'SumOfSqs'])

    def test_strata_restrict_the_permutations(self, gradient_data):
        species, _ = gradient_data
        engine = EcologicalStatistics()
        distances = engine.calculate_distance_matrix(species, 'bray_curtis')
        design = pd.DataFrame({'grp': np.repeat(['A', 'B', 'C'], 10)})

        result = engine.adonis(distances, design, permutations=99,
                               strata=np.tile([0, 1], 15), random_state=0)
        assert 0 < result['table'].loc['grp', 'Pr(>F)'] <= 1

    def test_star_shorthand_expands(self, gradient_data):
        species, _ = gradient_data
        engine = EcologicalStatistics()
        distances = engine.calculate_distance_matrix(species, 'bray_curtis')
        design = pd.DataFrame({'A': np.repeat(['x', 'y'], 15),
                               'B': np.tile(['p', 'q'], 15)})
        result = engine.adonis(distances, design, terms=['A*B'],
                               permutations=19, random_state=0)
        assert result['terms'] == ['A', 'B', 'A:B']

    def test_bad_by_argument_rejected(self, gradient_data):
        species, _ = gradient_data
        engine = EcologicalStatistics()
        distances = engine.calculate_distance_matrix(species, 'bray_curtis')
        with pytest.raises(ValueError, match="by must be"):
            engine.adonis(distances, pd.DataFrame({'g': np.repeat(['A', 'B'], 15)}),
                          by='whatever')

    def test_mismatched_row_count_rejected(self, gradient_data):
        species, _ = gradient_data
        engine = EcologicalStatistics()
        distances = engine.calculate_distance_matrix(species, 'bray_curtis')
        with pytest.raises(ValueError, match='rows'):
            engine.adonis(distances, pd.DataFrame({'g': ['A', 'B']}))


# ---------------------------------------------------------------------------
# Constrained-ordination significance
# ---------------------------------------------------------------------------

class TestOrdinationSignificance:

    def test_rda_anova_equals_the_regression_f(self):
        """Univariate response, one predictor: F must equal the regression F."""
        rng = np.random.default_rng(4)
        n = 40
        x = rng.normal(0, 1, n)
        y = 2.0 * x + rng.normal(0, 1, n)

        result = MultivariateAnalyzer().anova_rda(
            pd.DataFrame({'sp': y}), pd.DataFrame({'x': x}),
            permutations=49, random_state=0)

        regression = scipy_stats.linregress(x, y)
        expected_f = (regression.rvalue ** 2) / ((1 - regression.rvalue ** 2) / (n - 2))
        assert result['table'].loc['Model', 'F'] == pytest.approx(expected_f)

    def test_pure_noise_is_not_significant(self):
        rng = np.random.default_rng(1)
        result = MultivariateAnalyzer().anova_rda(
            pd.DataFrame(rng.normal(0, 1, (40, 6))),
            pd.DataFrame({'noise': rng.normal(0, 1, 40)}),
            permutations=499, random_state=1)
        assert result['table'].loc['Model', 'Pr(>F)'] > 0.05

    @pytest.mark.parametrize('method', ['rda', 'cca'])
    def test_a_real_gradient_is_significant(self, gradient_data, method):
        species, environment = gradient_data
        analyzer = MultivariateAnalyzer()
        test = analyzer.anova_rda if method == 'rda' else analyzer.anova_cca
        result = test(species, environment[['gradient']], permutations=299,
                      random_state=0)
        assert result['table'].loc['Model', 'Pr(>F)'] < 0.05
        assert 0 < result['proportion_constrained'] <= 1

    @pytest.mark.parametrize('by', ['terms', 'margin', 'axis'])
    def test_by_options_rank_signal_above_noise(self, gradient_data, by):
        species, environment = gradient_data
        table = MultivariateAnalyzer().anova_rda(
            species, environment, by=by, permutations=199, random_state=0)['table']

        assert 'Residual' in table.index
        tested = table.drop(index='Residual')
        assert (tested['Pr(>F)'] > 0).all() and (tested['Pr(>F)'] <= 1).all()

        # The gradient component must dominate the noise component by a wide
        # margin, and be significant. Asserting that the noise component falls
        # on a particular side of 0.05 would be testing the fixture's luck.
        ranked = tested.sort_values('F', ascending=False)
        assert ranked['Pr(>F)'].iloc[0] < 0.05
        assert ranked['F'].iloc[0] > 5 * ranked['F'].iloc[-1]

    def test_named_terms_identify_the_real_driver(self, gradient_data):
        species, environment = gradient_data
        table = MultivariateAnalyzer().anova_rda(
            species, environment, by='margin', permutations=199,
            random_state=0)['table']
        assert table.loc['gradient', 'F'] > table.loc['noise', 'F']
        assert table.loc['gradient', 'Pr(>F)'] < 0.05

    def test_partial_rda_removes_the_covariate_first(self, gradient_data):
        species, environment = gradient_data
        result = MultivariateAnalyzer().anova_rda(
            species, environment[['gradient']],
            conditioning=environment[['noise']], permutations=199, random_state=0)
        assert result['conditioned_variance'] > 0
        assert result['table'].loc['Model', 'Pr(>F)'] < 0.05

    def test_bad_by_argument_rejected(self, gradient_data):
        species, environment = gradient_data
        with pytest.raises(ValueError, match='by must be'):
            MultivariateAnalyzer().anova_rda(species, environment, by='magic')


class TestVarpart:

    @pytest.fixture
    def three_tables(self, gradient_data):
        species, environment = gradient_data
        rng = np.random.default_rng(2)
        n = len(species)
        spatial = pd.DataFrame({'x': rng.normal(0, 1, n), 'y': rng.normal(0, 1, n)},
                               index=species.index)
        soil = pd.DataFrame({'ph': rng.normal(0, 1, n)}, index=species.index)
        return species, environment[['gradient']], spatial, soil

    def test_two_table_fractions_sum_to_one(self, three_tables):
        species, environment, spatial, _ = three_tables
        result = MultivariateAnalyzer().varpart(
            species, environment, spatial, table_names=['env', 'space'],
            permutations=99, random_state=0)
        assert result['fractions']['adj_R2'].sum() == pytest.approx(1.0)

    def test_three_table_fractions_sum_to_one(self, three_tables):
        species, environment, spatial, soil = three_tables
        result = MultivariateAnalyzer().varpart(
            species, environment, spatial, soil,
            table_names=['env', 'space', 'soil'], permutations=49, random_state=0)
        assert result['fractions']['adj_R2'].sum() == pytest.approx(1.0)

    def test_the_real_driver_gets_the_large_unique_fraction(self, three_tables):
        species, environment, spatial, _ = three_tables
        result = MultivariateAnalyzer().varpart(
            species, environment, spatial, table_names=['env', 'space'],
            permutations=199, random_state=0)

        fractions = result['fractions']['adj_R2']
        assert fractions['[a] env unique'] > fractions['[c] space unique']
        testable = result['testable']
        assert testable.loc['env | others', 'Pr(>F)'] < 0.05
        assert testable.loc['space | others', 'Pr(>F)'] > 0.05

    def test_wrong_number_of_tables_rejected(self, three_tables):
        species, environment, *_ = three_tables
        with pytest.raises(ValueError, match='two or three'):
            MultivariateAnalyzer().varpart(species, environment)

    def test_mismatched_table_names_rejected(self, three_tables):
        species, environment, spatial, _ = three_tables
        with pytest.raises(ValueError, match='table_names'):
            MultivariateAnalyzer().varpart(species, environment, spatial,
                                           table_names=['only-one'])


class TestForwardSelection:

    def test_selects_the_driver_and_rejects_noise(self, gradient_data):
        species, environment = gradient_data
        result = MultivariateAnalyzer().forward_selection(
            species, environment, permutations=199, random_state=0)
        assert 'gradient' in result['selected']
        assert 'noise' not in result['selected']

    def test_respects_the_variable_limit(self, gradient_data):
        species, environment = gradient_data
        result = MultivariateAnalyzer().forward_selection(
            species, environment, permutations=49, max_variables=1, random_state=0)
        assert len(result['selected']) <= 1

    def test_strict_alpha_selects_nothing(self, gradient_data):
        species, environment = gradient_data
        result = MultivariateAnalyzer().forward_selection(
            species, environment, alpha=0.0, permutations=19, random_state=0)
        assert result['selected'] == []


# ---------------------------------------------------------------------------
# Baselga beta partitioning
# ---------------------------------------------------------------------------

class TestBetaPartition:

    @pytest.mark.parametrize('family', ['sorensen', 'jaccard'])
    def test_nested_pair_is_pure_nestedness(self, family):
        nested = pd.DataFrame({'a': [1, 1], 'b': [1, 1], 'c': [1, 0], 'd': [1, 0]},
                              index=['rich', 'poor'])
        result = DiversityAnalyzer().beta_partition(nested, family)
        assert result['turnover'].iloc[0, 1] == pytest.approx(0.0)
        assert result['nestedness'].iloc[0, 1] == pytest.approx(
            result['total'].iloc[0, 1])

    @pytest.mark.parametrize('family', ['sorensen', 'jaccard'])
    def test_disjoint_equal_richness_is_pure_turnover(self, family):
        disjoint = pd.DataFrame({'a': [1, 0], 'b': [1, 0], 'c': [0, 1], 'd': [0, 1]},
                                index=['s1', 's2'])
        result = DiversityAnalyzer().beta_partition(disjoint, family)
        assert result['turnover'].iloc[0, 1] == pytest.approx(1.0)
        assert result['nestedness'].iloc[0, 1] == pytest.approx(0.0)

    def test_identical_sites_have_no_beta_diversity(self):
        same = pd.DataFrame({'a': [1, 1], 'b': [1, 1]}, index=['s1', 's2'])
        result = DiversityAnalyzer().beta_partition(same)
        for key in ('total', 'turnover', 'nestedness'):
            assert result[key].iloc[0, 1] == pytest.approx(0.0)

    @pytest.mark.parametrize('family', ['sorensen', 'jaccard'])
    def test_components_sum_to_the_total_for_every_pair(self, family):
        rng = np.random.default_rng(0)
        data = pd.DataFrame(rng.integers(0, 2, (12, 20)),
                            index=[f'S{i}' for i in range(12)])
        result = DiversityAnalyzer().beta_partition(data, family)
        np.testing.assert_allclose(
            result['turnover'].values + result['nestedness'].values,
            result['total'].values, atol=1e-12)

    def test_matrices_are_symmetric_with_a_zero_diagonal(self):
        rng = np.random.default_rng(1)
        data = pd.DataFrame(rng.integers(0, 2, (8, 15)))
        result = DiversityAnalyzer().beta_partition(data)
        for key in ('total', 'turnover', 'nestedness'):
            values = result[key].values
            np.testing.assert_allclose(values, values.T, atol=1e-12)
            np.testing.assert_allclose(np.diag(values), 0, atol=1e-12)

    def test_unknown_family_rejected(self):
        with pytest.raises(ValueError, match='family'):
            DiversityAnalyzer().beta_partition(pd.DataFrame({'a': [1, 0]}), 'kulczynski')

    @pytest.mark.parametrize('family', ['sorensen', 'jaccard'])
    def test_multisite_components_sum_to_the_total(self, family):
        rng = np.random.default_rng(2)
        data = pd.DataFrame(rng.integers(0, 2, (10, 25)))
        result = DiversityAnalyzer().beta_partition_multisite(data, family)
        assert (result['turnover'] + result['nestedness']
                == pytest.approx(result['total']))

    def test_multisite_nested_set_has_no_turnover(self):
        nested = pd.DataFrame(np.tril(np.ones((6, 6)))[:, ::-1],
                              index=[f'S{i}' for i in range(6)])
        result = DiversityAnalyzer().beta_partition_multisite(nested)
        assert result['turnover'] == pytest.approx(0.0)
        assert result['nestedness'] > 0

    def test_single_site_returns_zeros(self):
        result = DiversityAnalyzer().beta_partition_multisite(
            pd.DataFrame({'a': [1], 'b': [1]}))
        assert result['total'] == 0.0


# ---------------------------------------------------------------------------
# Coverage-based rarefaction and Hill numbers
# ---------------------------------------------------------------------------

class TestCoverageAndHillNumbers:

    @pytest.fixture
    def counts_frame(self):
        counts = np.array([50.0, 30, 15, 8, 5, 3, 2, 2, 1, 1, 1, 1])
        return pd.DataFrame([counts], index=['S1'],
                            columns=[f'sp{i}' for i in range(len(counts))])

    def test_coverage_is_a_probability(self, counts_frame):
        coverage = DiversityAnalyzer().sample_coverage(counts_frame)
        assert 0 <= coverage.iloc[0] <= 1

    def test_coverage_increases_with_sample_size(self, counts_frame):
        analyzer = DiversityAnalyzer()
        counts = counts_frame.values[0]
        values = [analyzer.coverage_at_size(counts, m) for m in (5, 20, 50, 119)]
        assert all(np.diff(values) >= -1e-12)

    def test_coverage_of_a_complete_sample_is_one(self):
        """With no singletons the sample is treated as fully covered."""
        frame = pd.DataFrame([[10.0, 10.0, 10.0]], index=['S1'])
        assert DiversityAnalyzer().sample_coverage(frame).iloc[0] == pytest.approx(1.0)

    def test_hill_q0_matches_the_hurlbert_curve(self, counts_frame):
        analyzer = DiversityAnalyzer()
        sizes = [10, 25, 50, 80]
        hill = analyzer.hill_rarefaction(counts_frame, sizes=sizes, q_values=[0])
        hurlbert = analyzer.rarefaction_curve(counts_frame, sample_sizes=sizes)
        np.testing.assert_allclose(hill['diversity'].values,
                                   hurlbert['expected_species'].values)

    def test_at_the_reference_size_the_observed_value_is_returned(self, counts_frame):
        analyzer = DiversityAnalyzer()
        n = int(counts_frame.values.sum())
        table = analyzer.hill_rarefaction(counts_frame, sizes=[n], q_values=[0, 1, 2])

        counts = counts_frame.values[0]
        counts = counts[counts > 0]
        expected = [analyzer._hill_observed(counts, q) for q in (0, 1, 2)]
        np.testing.assert_allclose(table['diversity'].values, expected)
        assert (table['method'] == 'observed').all()

    def test_hill_numbers_decrease_with_q(self, counts_frame):
        table = DiversityAnalyzer().hill_rarefaction(
            counts_frame, sizes=[20, 50, 80], q_values=[0, 1, 2])
        for size in (20, 50, 80):
            values = table[table['size'] == size].sort_values('q')['diversity'].values
            assert np.all(np.diff(values) <= 1e-9)

    def test_rarefaction_is_monotone_increasing(self, counts_frame):
        table = DiversityAnalyzer().hill_rarefaction(
            counts_frame, sizes=[5, 20, 50, 100], q_values=[0])
        assert table['diversity'].is_monotonic_increasing

    @pytest.mark.parametrize('q', [0, 1, 2])
    def test_extrapolation_is_continuous_and_monotone(self, counts_frame, q):
        analyzer = DiversityAnalyzer()
        counts = counts_frame.values[0]
        counts = counts[counts > 0]
        n = int(counts.sum())

        observed = analyzer._hill_observed(counts, q)
        extrapolated = [analyzer._hill_extrapolate(counts, n, m, q)
                        for m in (0, 10, 50, 200, 1000)]

        assert extrapolated[0] == pytest.approx(observed)
        assert all(np.diff(extrapolated) >= -1e-9)

    def test_extrapolation_beyond_double_warns(self, counts_frame):
        with pytest.warns(UserWarning, match='unreliable'):
            DiversityAnalyzer().hill_rarefaction(counts_frame, q_values=[0],
                                                 extrapolate_to=10_000)

    def test_unsupported_q_rejected(self, counts_frame):
        with pytest.raises(ValueError, match='q = 0, 1 and 2'):
            DiversityAnalyzer().hill_rarefaction(counts_frame, q_values=[3])

    def test_coverage_standardisation_reaches_the_target(self):
        rng = np.random.default_rng(7)
        even = rng.multinomial(400, np.ones(60) / 60).astype(float)
        uneven_p = np.arange(1, 61) ** -2.0
        uneven = rng.multinomial(400, uneven_p / uneven_p.sum()).astype(float)
        frame = pd.DataFrame([even, uneven], index=['even', 'uneven'])

        table = DiversityAnalyzer().coverage_standardized_diversity(
            frame, coverage=0.95, q_values=[0])
        assert (table['achieved_coverage'] >= 0.95 - 1e-6).all()

    def test_coverage_standardisation_needs_different_sizes_per_sample(self):
        """That is the whole point: equal size is not equal completeness."""
        rng = np.random.default_rng(7)
        even = rng.multinomial(400, np.ones(60) / 60).astype(float)
        uneven_p = np.arange(1, 61) ** -2.0
        uneven = rng.multinomial(400, uneven_p / uneven_p.sum()).astype(float)
        frame = pd.DataFrame([even, uneven], index=['even', 'uneven'])

        table = DiversityAnalyzer().coverage_standardized_diversity(
            frame, coverage=0.95, q_values=[0]).set_index('sample_id')
        assert table.loc['even', 'size'] != table.loc['uneven', 'size']

    def test_automatic_base_coverage(self):
        rng = np.random.default_rng(3)
        frame = pd.DataFrame(rng.integers(0, 15, (4, 20)).astype(float))
        table = DiversityAnalyzer().coverage_standardized_diversity(frame, q_values=[0])
        assert table['target_coverage'].nunique() == 1
        assert 0 < table['target_coverage'].iloc[0] < 1

    def test_invalid_target_coverage_rejected(self, counts_frame):
        with pytest.raises(ValueError, match='between 0 and 1'):
            DiversityAnalyzer().coverage_standardized_diversity(counts_frame,
                                                                coverage=1.5)

    def test_non_integer_abundances_warn(self):
        frame = pd.DataFrame([[1.5, 2.5, 3.5]], index=['S1'])
        with pytest.warns(UserWarning, match='integer counts'):
            DiversityAnalyzer().sample_coverage(frame)
