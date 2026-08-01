"""
Tests for the ecology modules with the largest remaining coverage gaps:
functional traits, spatial analysis, specialized methods, temporal analysis,
environmental modelling and the core VegZ facade.
"""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from VegZ import VegZ  # noqa: E402
from VegZ.environmental import EnvironmentalModeler  # noqa: E402
from VegZ.functional_traits import (  # noqa: E402
    FunctionalTraitAnalyzer, TraitSyndromes, quick_functional_diversity,
    quick_functional_groups)
from VegZ.spatial import SpatialAnalyzer  # noqa: E402
from VegZ.specialized_methods import (  # noqa: E402
    CommunityAssemblyAnalyzer, MetacommunityAnalyzer, NetworkAnalyzer,
    PhylogeneticDiversityAnalyzer, quick_cooccurrence_network,
    quick_metacommunity_analysis, quick_phylogenetic_diversity)
from VegZ.temporal import TemporalAnalyzer  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


@pytest.fixture
def species_names():
    return [f'sp{i:02d}' for i in range(10)]


@pytest.fixture
def abundance(species_names):
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.integers(1, 12, (15, 10)).astype(float),
                        index=[f'S{i:02d}' for i in range(15)],
                        columns=species_names)


@pytest.fixture
def traits(species_names):
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        'species': species_names,
        'sla': rng.uniform(5, 30, 10),
        'height': rng.uniform(0.1, 20, 10),
        'seed_mass': rng.uniform(0.1, 100, 10),
    })


@pytest.fixture
def phylogeny(species_names):
    rng = np.random.default_rng(2)
    matrix = rng.uniform(0, 1, (10, 10))
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0)
    return pd.DataFrame(matrix, index=species_names, columns=species_names)


# ---------------------------------------------------------------------------
# Functional traits
# ---------------------------------------------------------------------------

class TestFunctionalTraits:

    @pytest.fixture
    def analyzer(self, traits, abundance):
        a = FunctionalTraitAnalyzer()
        a.load_trait_data(traits, abundance, 'species')
        return a

    def test_indices_are_within_their_defined_bounds(self, analyzer):
        table = analyzer.calculate_functional_diversity()['site_diversity']
        assert table['FEve'].between(0, 1).all()
        assert table['FDiv'].between(0, 1).all()
        assert (table['FRic'] >= 0).all()
        assert (table['FDis'] >= 0).all()
        assert (table['RaoQ'] >= 0).all()

    def test_unstandardised_traits_still_work(self, analyzer):
        table = analyzer.calculate_functional_diversity(
            standardize=False)['site_diversity']
        assert len(table) == 15

    def test_specific_sites_and_traits_can_be_selected(self, analyzer):
        table = analyzer.calculate_functional_diversity(
            sites=['S00', 'S01'], traits=['sla', 'height'])['site_diversity']
        assert len(table) == 2

    def test_missing_trait_data_raises(self):
        with pytest.raises(ValueError, match='Trait data not loaded'):
            FunctionalTraitAnalyzer().calculate_functional_diversity()

    def test_rao_quadratic_entropy_matches_its_definition(self, analyzer):
        """RaoQ is w' D w over the species present at a site."""
        results = analyzer.calculate_functional_diversity()
        distances = results['trait_distances']
        site = analyzer.abundance_data.iloc[0]
        present = site[site > 0].index
        weights = (site[present] / site[present].sum()).values
        expected = float(weights @ distances.loc[present, present].values @ weights)
        assert results['site_diversity'].iloc[0]['RaoQ'] == pytest.approx(expected)

    @pytest.mark.parametrize('method', ['hierarchical', 'kmeans'])
    def test_functional_groups(self, analyzer, method):
        result = analyzer.identify_functional_groups(n_groups=3, method=method)
        assert result['n_groups'] == 3
        assert len(result['functional_groups']) == 10

    @pytest.mark.parametrize('method', ['hierarchical', 'kmeans'])
    def test_automatic_group_count_does_not_overrun(self, analyzer, method):
        """The elbow index used to run off the end of the candidate list."""
        result = analyzer.identify_functional_groups(method=method)
        assert result['n_groups'] >= 2

    def test_unknown_clustering_method_raises(self, analyzer):
        with pytest.raises(ValueError, match='Unknown clustering method'):
            analyzer.identify_functional_groups(n_groups=2, method='telepathy')

    def test_trait_environment_relationships(self, analyzer, abundance):
        rng = np.random.default_rng(3)
        environment = pd.DataFrame({'elev': rng.uniform(0, 2000, 15),
                                    'ph': rng.uniform(4, 8, 15)},
                                   index=abundance.index)
        result = analyzer.trait_environment_relationships(environment)
        assert result['correlations'].shape == (3, 2)
        assert ((result['p_values'] >= 0) & (result['p_values'] <= 1)).all().all()

    def test_trait_environment_requires_abundance(self, traits):
        analyzer = FunctionalTraitAnalyzer()
        analyzer.load_trait_data(traits, species_column='species')
        with pytest.raises(ValueError, match='trait and abundance data'):
            analyzer.trait_environment_relationships(pd.DataFrame({'e': [1.0]}))

    def test_functional_beta_diversity(self, analyzer):
        result = analyzer.calculate_functional_beta_diversity()
        assert 'gamma_diversity' in result
        assert 'beta_diversity' in result

    def test_plots(self, analyzer):
        analyzer.identify_functional_groups(n_groups=3)
        assert analyzer.plot_functional_space() is not None
        assert analyzer.plot_functional_space(color_by='functional_group') is not None
        assert analyzer.plot_trait_distributions() is not None

    def test_trait_syndromes(self, analyzer):
        syndromes = TraitSyndromes(analyzer)
        result = syndromes.identify_trait_syndromes()
        assert 'loadings' in result and 'syndromes' in result

        with pytest.raises(ValueError, match='Unknown method'):
            syndromes.identify_trait_syndromes(method='astrology')

    def test_trait_trade_offs(self, analyzer):
        result = TraitSyndromes(analyzer).analyze_trait_trade_offs()
        assert sum(result['summary'].values()) == len(result['trade_offs'])

    def test_quick_helpers(self, traits, abundance):
        assert 'site_diversity' in quick_functional_diversity(traits, abundance)
        assert 'n_groups' in quick_functional_groups(traits, n_groups=2)

    def test_no_common_species_warns(self, abundance):
        traits = pd.DataFrame({'species': ['nothing_here'], 'sla': [1.0]})
        analyzer = FunctionalTraitAnalyzer()
        with pytest.warns(UserWarning, match='No common species'):
            analyzer.load_trait_data(traits, abundance, 'species')


# ---------------------------------------------------------------------------
# Spatial analysis
# ---------------------------------------------------------------------------

class TestSpatialAnalyzer:

    @pytest.fixture
    def analyzer(self):
        return SpatialAnalyzer()

    @pytest.fixture
    def points(self):
        rng = np.random.default_rng(0)
        n = 30
        return pd.DataFrame({
            'longitude': rng.uniform(-80, -79, n),
            'latitude': rng.uniform(40, 41, n),
            'response': rng.uniform(0, 50, n),
        })

    @pytest.mark.parametrize('method', ['idw', 'kriging', 'rbf', 'nearest',
                                        'linear', 'cubic'])
    def test_interpolation_methods(self, analyzer, points, method):
        result = analyzer.spatial_interpolation(points, method=method,
                                                grid_resolution=0.2)
        assert result['Z_grid'].shape == result['X_grid'].shape

    def test_unknown_interpolation_method_raises(self, analyzer, points):
        with pytest.raises(ValueError, match='Unknown interpolation method'):
            analyzer.spatial_interpolation(points, method='ouija',
                                           grid_resolution=0.2)

    def test_too_few_points_raises(self, analyzer):
        with pytest.raises(ValueError, match='at least 3'):
            analyzer.spatial_interpolation(
                pd.DataFrame({'longitude': [0.0], 'latitude': [0.0],
                              'response': [1.0]}))

    def test_absurd_grid_resolution_is_refused(self, analyzer, points):
        with pytest.raises(ValueError, match='grid cells'):
            analyzer.spatial_interpolation(points, grid_resolution=1e-5)

    def test_non_positive_resolution_is_refused(self, analyzer, points):
        with pytest.raises(ValueError, match='must be positive'):
            analyzer.spatial_interpolation(points, grid_resolution=0)

    def test_idw_reproduces_values_at_the_data_points(self, analyzer):
        points = np.array([[0.0, 0.0], [1.0, 1.0]])
        values = np.array([10.0, 20.0])
        interpolated = analyzer._inverse_distance_weighting(points, values, points)
        np.testing.assert_allclose(interpolated, values)

    @pytest.mark.parametrize('method', ['morans_i', 'gearys_c', 'variogram'])
    def test_autocorrelation_methods(self, analyzer, points, method):
        assert analyzer.spatial_autocorrelation(points, method=method)

    def test_unknown_autocorrelation_method_raises(self, analyzer, points):
        with pytest.raises(ValueError, match='Unknown method'):
            analyzer.spatial_autocorrelation(points, method='dowsing')

    def test_too_few_points_for_autocorrelation(self, analyzer):
        with pytest.raises(ValueError, match='at least 3'):
            analyzer.spatial_autocorrelation(
                pd.DataFrame({'longitude': [0.0, 1.0], 'latitude': [0.0, 1.0],
                              'response': [1.0, 2.0]}))

    def test_variogram_semivariance_rises_with_distance(self, analyzer):
        grid = np.array([[i, j] for i in range(8) for j in range(8)], dtype=float)
        frame = pd.DataFrame({'longitude': grid[:, 0], 'latitude': grid[:, 1],
                              'response': grid[:, 0] + grid[:, 1]})
        result = analyzer.spatial_autocorrelation(frame, method='variogram')
        values = result['variogram_values']
        assert values[-1] > values[0]

    def test_fragmentation_analysis(self, analyzer):
        rng = np.random.default_rng(0)
        landscape = rng.integers(0, 4, (30, 30))
        result = analyzer.fragmentation_analysis(landscape)
        assert 'landscape_level' in result
        assert any(key.startswith('patch_type_') for key in result)

    def test_patch_metrics_on_a_known_landscape(self, analyzer):
        """A single solid block: one patch, 100% of the class, LSI = 1."""
        landscape = np.ones((10, 10), dtype=int)
        result = analyzer.fragmentation_analysis(landscape)['patch_type_1']
        assert result['number_of_patches'] == 1
        assert result['percentage_of_landscape'] == pytest.approx(100.0)
        assert result['landscape_shape_index'] == pytest.approx(1.0)

    def test_habitat_suitability_modelling(self, analyzer, points):
        presence = points[['longitude', 'latitude']].copy()
        presence['presence'] = (points['response'] > 25).astype(int)
        environment = points[['longitude', 'latitude', 'response']].rename(
            columns={'response': 'ndvi'})

        result = analyzer.habitat_suitability_modeling(presence, environment)
        assert 'performance_metrics' in result
        assert len(result['variable_importance']) >= 1

    def test_habitat_suitability_rejects_unknown_method(self, analyzer, points):
        presence = points[['longitude', 'latitude']].copy()
        presence['presence'] = 1
        environment = points[['longitude', 'latitude', 'response']]
        with pytest.raises(ValueError, match='Unknown method'):
            analyzer.habitat_suitability_modeling(presence, environment,
                                                  method='augury')


# ---------------------------------------------------------------------------
# Specialized methods
# ---------------------------------------------------------------------------

class TestSpecializedMethods:

    def test_phylogenetic_diversity_metrics(self, abundance, phylogeny):
        analyzer = PhylogeneticDiversityAnalyzer()
        analyzer.load_phylogeny(phylogeny)
        result = analyzer.calculate_phylogenetic_diversity(abundance)
        metrics = result['metrics']
        for column in ('pd', 'mpd', 'mntd', 'nri', 'nti'):
            assert column in metrics.columns
        assert (metrics['pd'] >= 0).all()

    def test_phylogeny_is_not_mutated(self, abundance, phylogeny):
        before = phylogeny.values.copy()
        analyzer = PhylogeneticDiversityAnalyzer()
        analyzer.load_phylogeny(phylogeny)
        analyzer.calculate_phylogenetic_diversity(abundance)
        np.testing.assert_array_equal(before, phylogeny.values)

    def test_null_expectations_are_cached_by_richness(self, abundance, phylogeny):
        analyzer = PhylogeneticDiversityAnalyzer()
        analyzer.load_phylogeny(phylogeny)
        analyzer.calculate_phylogenetic_diversity(abundance)
        assert analyzer._null_cache

    def test_missing_phylogeny_raises(self, abundance):
        with pytest.raises(ValueError, match='not loaded'):
            PhylogeneticDiversityAnalyzer().calculate_phylogenetic_diversity(abundance)

    def test_unknown_phylogeny_format_raises(self, phylogeny):
        with pytest.raises(ValueError, match='Unknown phylogenetic data format'):
            PhylogeneticDiversityAnalyzer().load_phylogeny(phylogeny, format='nexus')

    def test_no_shared_species_raises(self, phylogeny):
        community = pd.DataFrame({'other_species': [1, 2]})
        analyzer = PhylogeneticDiversityAnalyzer()
        analyzer.load_phylogeny(phylogeny)
        with pytest.raises(ValueError, match='No common species'):
            analyzer.calculate_phylogenetic_diversity(community)

    def test_mpd_matches_its_definition(self, phylogeny):
        analyzer = PhylogeneticDiversityAnalyzer()
        subset = phylogeny.iloc[:4, :4]
        weights = pd.Series([1.0, 2.0, 3.0, 4.0], index=subset.index)

        i_idx, j_idx = np.triu_indices(4, k=1)
        pair_weights = weights.values[i_idx] * weights.values[j_idx]
        expected = float(np.sum(subset.values[i_idx, j_idx] * pair_weights)
                         / pair_weights.sum())
        assert analyzer._calculate_mpd(subset, weights) == pytest.approx(expected)

    @pytest.mark.parametrize('structure,expected', [
        ('nested', 'nested'),
        ('clumped', 'clumped'),
    ])
    def test_metacommunity_structure_types_are_reachable(self, structure, expected):
        if structure == 'nested':
            matrix = np.zeros((10, 10), dtype=int)
            for i in range(10):
                matrix[i, :10 - i] = 1
        else:
            matrix = np.zeros((12, 12), dtype=int)
            for block in range(3):
                matrix[block * 4:(block + 1) * 4, block * 4:(block + 1) * 4] = 1

        frame = pd.DataFrame(matrix,
                             index=[f'S{i}' for i in range(matrix.shape[0])],
                             columns=[f'sp{i}' for i in range(matrix.shape[1])])
        result = MetacommunityAnalyzer().elements_of_metacommunity_structure(frame)
        assert result['structure_type'] == expected
        assert result['interpretation']

    def test_boundary_clumping_uses_morisita(self, abundance):
        result = MetacommunityAnalyzer().elements_of_metacommunity_structure(abundance)
        clumping = result['boundary_clumping']
        assert 'morisita_index' in clumping
        assert clumping['morisita_index'] >= 0

    @pytest.mark.parametrize('method', ['correlation', 'jaccard'])
    def test_cooccurrence_network(self, abundance, method):
        result = NetworkAnalyzer().build_cooccurrence_network(
            abundance, method=method, threshold=0.3)
        properties = result['network_properties']
        assert properties['n_nodes'] == abundance.shape[1]
        assert 0 <= properties['density'] <= 1

    def test_unknown_network_method_raises(self, abundance):
        with pytest.raises(ValueError, match='Unknown method'):
            NetworkAnalyzer().build_cooccurrence_network(abundance, method='vibes')

    @pytest.mark.parametrize('method', ['correlation', 'jaccard'])
    def test_self_associations_are_zeroed_without_mutating_the_input(
            self, abundance, method):
        """
        The diagonal used to be zeroed by writing through `.values`, which is
        only a view by accident in pandas 2 and is read-only in pandas 3.
        """
        before = abundance.to_numpy(copy=True)
        result = NetworkAnalyzer().build_cooccurrence_network(
            abundance, method=method, threshold=0.3)

        associations = result['associations']
        np.testing.assert_allclose(np.diag(associations.to_numpy()), 0.0)
        np.testing.assert_array_equal(abundance.to_numpy(), before)

        # A zeroed diagonal means no species is joined to itself.
        adjacency = result['adjacency_matrix']
        np.testing.assert_array_equal(np.diag(adjacency.to_numpy()), 0)

    def test_community_assembly_analysis(self, abundance, traits, phylogeny):
        result = CommunityAssemblyAnalyzer().assembly_process_analysis(
            abundance, traits.set_index('species'), phylogeny)
        assert 'taxonomic' in result
        assert 'interpretation' in result

    def test_constructing_analyzers_leaves_the_global_rng_alone(self):
        np.random.seed(99)
        expected = np.random.rand()
        np.random.seed(99)
        PhylogeneticDiversityAnalyzer(7)
        CommunityAssemblyAnalyzer(11)
        assert np.random.rand() == expected

    def test_quick_helpers(self, abundance, phylogeny):
        assert quick_phylogenetic_diversity(abundance, phylogeny)
        assert quick_metacommunity_analysis(abundance)
        assert quick_cooccurrence_network(abundance, threshold=0.3)


# ---------------------------------------------------------------------------
# Temporal analysis
# ---------------------------------------------------------------------------

class TestTemporalAnalyzer:

    @pytest.fixture
    def series(self):
        rng = np.random.default_rng(2)
        dates = pd.date_range('2015-01-31', periods=72, freq='ME')
        t = np.arange(72)
        values = (10 + 0.08 * t + 3 * np.sin(t / 12 * 2 * np.pi)
                  + rng.normal(0, 0.6, 72))
        return pd.DataFrame({'date': dates, 'response': values})

    @pytest.mark.parametrize('method', ['linear', 'polynomial', 'spline',
                                        'lowess', 'mann_kendall'])
    def test_trend_methods(self, series, method):
        assert TemporalAnalyzer().trend_detection(series, 'date', 'response',
                                                  method=method)

    def test_unknown_trend_method_raises(self, series):
        with pytest.raises(ValueError, match='Unknown trend method'):
            TemporalAnalyzer().trend_detection(series, 'date', 'response',
                                               method='haruspicy')

    def test_linear_trend_recovers_a_positive_slope(self, series):
        result = TemporalAnalyzer().trend_detection(series, 'date', 'response',
                                                    method='linear')
        assert result['slope'] > 0
        assert result['trend_direction'] == 'increasing'
        assert result['p_value'] < 0.05

    def test_mann_kendall_is_order_invariant(self, series):
        analyzer = TemporalAnalyzer()
        ordered = analyzer.trend_detection(series, 'date', 'response',
                                           method='mann_kendall')
        shuffled = analyzer.trend_detection(series.sample(frac=1, random_state=0),
                                            'date', 'response', method='mann_kendall')
        assert ordered['s_statistic'] == shuffled['s_statistic']

    def test_mann_kendall_on_a_short_series(self):
        frame = pd.DataFrame({'t': [1, 2], 'y': [1.0, 2.0]})
        result = TemporalAnalyzer().trend_detection(frame, 't', 'y',
                                                    method='mann_kendall')
        assert result['trend'] == 'insufficient data'

    @pytest.mark.parametrize('method', ['classical', 'stl', 'x11'])
    def test_seasonal_decomposition(self, series, method):
        result = TemporalAnalyzer().seasonal_decomposition(
            series, 'date', 'response', method=method, period=12)
        for component in ('trend', 'seasonal', 'residual'):
            assert component in result

    def test_unknown_decomposition_method_raises(self, series):
        with pytest.raises(ValueError, match='Unknown decomposition method'):
            TemporalAnalyzer().seasonal_decomposition(series, 'date', 'response',
                                                      method='tea-leaves')

    @pytest.mark.parametrize('freq,expected', [
        ('D', 365), ('ME', 12), ('QE', 4), ('W', 52),
    ])
    def test_seasonality_detection_by_frequency(self, freq, expected):
        index = pd.date_range('2015-01-01', periods=30, freq=freq)
        series = pd.Series(np.arange(30), index=index)
        assert TemporalAnalyzer()._detect_seasonality(series) == expected

    @pytest.mark.parametrize('model', ['sigmoid', 'gaussian', 'double_sigmoid',
                                       'beta', 'weibull'])
    def test_phenology_models_converge(self, model):
        rng = np.random.default_rng(2)
        doy = np.arange(1, 366, 5)
        response = 30 * np.exp(-0.5 * ((doy - 180) / 40) ** 2) + rng.normal(0, 1, len(doy))
        frame = pd.DataFrame({'doy': doy, 'resp': response})

        fit = TemporalAnalyzer().phenology_modeling(
            frame, 'doy', 'resp', model_type=model)['results']['combined']
        assert fit['success'], f'{model} failed to converge'

    def test_phenology_by_species(self):
        rng = np.random.default_rng(4)
        doy = np.tile(np.arange(1, 366, 10), 2)
        frame = pd.DataFrame({
            'doy': doy,
            'resp': 20 * np.exp(-0.5 * ((doy - 180) / 40) ** 2) + rng.normal(0, 1, len(doy)),
            'species': np.repeat(['a', 'b'], len(doy) // 2),
        })
        results = TemporalAnalyzer().phenology_modeling(
            frame, 'doy', 'resp', model_type='gaussian', species_col='species')
        assert set(results['results']) == {'a', 'b'}

    def test_phenology_requires_the_named_columns(self):
        with pytest.raises(ValueError, match='Required columns'):
            TemporalAnalyzer().phenology_modeling(pd.DataFrame({'x': [1]}),
                                                  'date', 'response')

    def test_unknown_phenology_model_raises(self):
        frame = pd.DataFrame({'doy': np.arange(20), 'resp': np.arange(20.0)})
        with pytest.raises(ValueError, match='Unknown model type'):
            TemporalAnalyzer().phenology_modeling(frame, 'doy', 'resp',
                                                  model_type='crystal-ball')

    @pytest.mark.parametrize('curve', ['logistic', 'gompertz', 'von_bertalanffy',
                                       'exponential', 'power'])
    def test_growth_curves(self, curve):
        t = np.arange(1, 25)
        frame = pd.DataFrame({'time': t,
                              'size': 100 / (1 + np.exp(-0.4 * (t - 12)))})
        result = TemporalAnalyzer().growth_curve_fitting(frame, 'time', 'size',
                                                         curve_type=curve)
        assert 'combined' in result['results']

    def test_unknown_growth_curve_raises(self):
        frame = pd.DataFrame({'time': [1, 2, 3, 4], 'size': [1.0, 2, 3, 4]})
        with pytest.raises(ValueError, match='Unknown curve type'):
            TemporalAnalyzer().growth_curve_fitting(frame, 'time', 'size',
                                                    curve_type='hockey-stick')

    def test_climate_vegetation_response(self, series):
        rng = np.random.default_rng(5)
        climate = pd.DataFrame({'date': series['date'],
                                'temp': rng.normal(15, 5, len(series))})
        result = TemporalAnalyzer().climate_vegetation_response(
            series, climate, 'date', 'response')
        assert 'temp' in result['results']


# ---------------------------------------------------------------------------
# Environmental modelling
# ---------------------------------------------------------------------------

class TestEnvironmentalModeler:

    @pytest.fixture
    def gam_frame(self):
        rng = np.random.default_rng(0)
        n = 60
        elevation = rng.uniform(0, 2000, n)
        return pd.DataFrame({
            'elevation': elevation,
            'ph': rng.uniform(4, 8, n),
            'response': 0.01 * elevation + rng.normal(0, 3, n),
        })

    @pytest.mark.parametrize('smoother', ['spline', 'lowess', 'polynomial',
                                          'gaussian_process'])
    def test_gam_smoothers(self, gam_frame, smoother):
        result = EnvironmentalModeler().fit_gam(
            gam_frame, 'response', ['elevation'],
            smoother_types={'elevation': smoother})
        assert result['fitted_values'].dtype.kind == 'f'
        assert np.isfinite(result['fitted_values']).all()

    def test_unknown_smoother_falls_back_with_a_warning(self, gam_frame):
        with pytest.warns(UserWarning, match='Unknown smoother'):
            EnvironmentalModeler().fit_gam(gam_frame, 'response', ['elevation'],
                                           smoother_types={'elevation': 'runes'})

    @pytest.mark.parametrize('family', ['gaussian', 'poisson'])
    def test_gam_families(self, gam_frame, family):
        frame = gam_frame.copy()
        frame['response'] = np.abs(frame['response']).round()
        result = EnvironmentalModeler().fit_gam(frame, 'response', ['elevation'],
                                                family=family)
        assert 'diagnostics' in result

    def test_unknown_family_raises(self, gam_frame):
        with pytest.raises(ValueError, match='Unknown family'):
            EnvironmentalModeler().fit_gam(gam_frame, 'response', ['elevation'],
                                           family='fortune-telling')

    def test_insufficient_data_raises(self):
        frame = pd.DataFrame({'x': [1.0, 2.0], 'y': [1.0, 2.0]})
        with pytest.raises(ValueError, match='Insufficient data'):
            EnvironmentalModeler().fit_gam(frame, 'y', ['x'])

    def test_gam_anova_is_produced(self, gam_frame):
        result = EnvironmentalModeler().fit_gam(gam_frame, 'response',
                                                ['elevation', 'ph'])
        assert set(result['anova']) == {'elevation', 'ph'}

    @pytest.mark.parametrize('curve', ['gaussian', 'skewed_gaussian', 'beta',
                                       'linear', 'threshold', 'unimodal'])
    def test_response_curves(self, curve):
        rng = np.random.default_rng(0)
        gradient = np.linspace(0, 20, 60)
        abundance = (30 * np.exp(-0.5 * ((gradient - 12) / 3) ** 2)
                     + rng.normal(0, 0.6, 60))
        result = EnvironmentalModeler().species_response_curves(
            pd.Series(abundance), pd.Series(gradient), curve)
        assert result['success'], curve

    def test_unknown_curve_type_raises(self):
        with pytest.raises(ValueError, match='Unknown curve type'):
            EnvironmentalModeler().species_response_curves(
                pd.Series([1.0] * 10), pd.Series(range(10)), 'sinusoid')

    def test_too_few_points_for_a_curve_raises(self):
        with pytest.raises(ValueError, match='Insufficient data'):
            EnvironmentalModeler().species_response_curves(
                pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]), 'gaussian')

    @pytest.mark.parametrize('method', ['cca', 'dca', 'rda', 'pca_env'])
    def test_gradient_analyses(self, abundance, method):
        rng = np.random.default_rng(1)
        environment = pd.DataFrame({'elev': rng.uniform(0, 2000, len(abundance)),
                                    'ph': rng.uniform(4, 8, len(abundance))},
                                   index=abundance.index)
        result = EnvironmentalModeler().environmental_gradient_analysis(
            abundance, environment, method=method)
        assert result['method']

    def test_unknown_gradient_method_raises(self, abundance):
        with pytest.raises(ValueError, match='Unknown gradient method'):
            EnvironmentalModeler().environmental_gradient_analysis(
                abundance, pd.DataFrame(), method='scrying')


# ---------------------------------------------------------------------------
# Core facade
# ---------------------------------------------------------------------------

class TestCoreFacade:

    def test_load_data_warns_without_species_columns(self, abundance, tmp_path):
        path = tmp_path / 'data.csv'
        abundance.to_csv(path)
        with pytest.warns(UserWarning, match='species_cols'):
            VegZ().load_data(str(path))

    def test_load_data_with_explicit_columns_is_quiet(self, abundance, tmp_path):
        import warnings
        path = tmp_path / 'data.csv'
        abundance.to_csv(path)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            VegZ().load_data(str(path), species_cols=list(abundance.columns))

    def test_unsupported_format_raises(self, tmp_path):
        path = tmp_path / 'data.csv'
        path.write_text('a,b\n1,2\n', encoding='utf-8')
        with pytest.raises(ValueError, match='Unsupported format'):
            VegZ().load_data(str(path), format_type='parquet')

    def test_methods_require_a_species_matrix(self):
        analyzer = VegZ()
        for call in (analyzer.calculate_diversity, analyzer.pca_analysis,
                     analyzer.summary_statistics, analyzer.rarefaction_curve,
                     analyzer.species_accumulation_curve):
            with pytest.raises(ValueError, match='not available'):
                call()

    def test_filter_rare_species_records_what_it_did(self, abundance):
        analyzer = VegZ()
        analyzer.species_matrix = abundance
        analyzer.filter_rare_species(min_occurrences=100)
        record = analyzer.metadata['filter_rare_species']
        assert record['n_original'] == abundance.shape[1]
        assert record['n_retained'] == 0

    def test_standardize_species_names(self):
        analyzer = VegZ()
        analyzer.data = pd.DataFrame({'species': ['quercus ALBA L.', 'pinus strobus']})
        result = analyzer.standardize_species_names('species')
        assert result['species_clean'].iloc[0] == 'Quercus alba'

    def test_standardize_species_names_requires_the_column(self):
        analyzer = VegZ()
        analyzer.data = pd.DataFrame({'other': [1]})
        with pytest.raises(ValueError, match='species column not found'):
            analyzer.standardize_species_names('species')

    def test_pca_and_nmds_expose_both_score_key_names(self, abundance):
        analyzer = VegZ()
        analyzer.species_matrix = abundance
        for result in (analyzer.pca_analysis(),
                       analyzer.nmds_analysis(random_state=0)):
            assert 'scores' in result and 'site_scores' in result

    def test_export_results_writes_csv(self, abundance, tmp_path):
        analyzer = VegZ()
        analyzer.species_matrix = abundance
        diversity = analyzer.calculate_diversity()
        analyzer.export_results({'diversity': diversity}, str(tmp_path / 'out'))
        assert (tmp_path / 'out_diversity.csv').exists()

    def test_export_rejects_unknown_format(self, abundance, tmp_path):
        analyzer = VegZ()
        analyzer.species_matrix = abundance
        with pytest.raises(ValueError, match='Unsupported export format'):
            analyzer.export_results({}, str(tmp_path / 'out'), format_type='xml')

    def test_plot_helpers(self, abundance):
        analyzer = VegZ()
        analyzer.species_matrix = abundance
        diversity = analyzer.calculate_diversity()
        assert analyzer.plot_diversity(diversity, 'shannon')
        assert analyzer.plot_ordination(analyzer.pca_analysis())
        assert analyzer.plot_cluster_dendrogram(
            analyzer.hierarchical_clustering(n_clusters=3))
        assert analyzer.plot_species_accumulation(
            analyzer.rarefaction_curve(sample_sizes=[5, 10, 20]))

    def test_plot_ordination_needs_two_axes(self, abundance):
        analyzer = VegZ()
        analyzer.species_matrix = abundance
        single = {'scores': pd.DataFrame({'PC1': [1.0, 2.0]})}
        with pytest.raises(ValueError, match='at least two axes'):
            analyzer.plot_ordination(single)

    def test_transform_rejects_unknown_methods(self, abundance):
        analyzer = VegZ()
        analyzer.species_matrix = abundance
        with pytest.raises(ValueError, match='Unknown transformation'):
            analyzer._transform_data(abundance, 'alchemy')

    def test_nmds_refuses_incompatible_transform_and_metric(self, abundance):
        analyzer = VegZ()
        analyzer.species_matrix = abundance
        with pytest.raises(ValueError, match='non-negative'):
            analyzer.nmds_analysis(distance_metric='bray_curtis',
                                   transform='standardize')
