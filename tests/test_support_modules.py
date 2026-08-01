"""
Tests for the remaining low-coverage support modules: nestedness, plotting,
interactive dashboards, remote sensing, coordinate systems and temporal
validation.
"""

import pathlib
import warnings
from unittest import mock

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from VegZ.data_management import remote_sensing as rs  # noqa: E402
from VegZ.data_management.coordinate_systems import CoordinateTransformer  # noqa: E402
from VegZ.data_quality.temporal_validation import TemporalValidator  # noqa: E402
from VegZ.interactive_viz import (InteractiveVisualizer, ReportGenerator,  # noqa: E402
                                  quick_analysis_report,
                                  quick_diversity_dashboard)
from VegZ.nestedness import (CooccurrenceAnalysis, NestednessAnalyzer,  # noqa: E402
                             NestednessSignificance, NullModels)
from VegZ.visualization import VegetationPlotter  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


@pytest.fixture
def perfectly_nested():
    """A staircase matrix: the textbook perfectly nested community."""
    n = 10
    matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        matrix[i, :n - i] = 1
    return pd.DataFrame(matrix, index=[f'S{i}' for i in range(n)],
                        columns=[f'sp{i}' for i in range(n)])


@pytest.fixture
def community():
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.integers(0, 2, (12, 10)),
                        index=[f'S{i:02d}' for i in range(12)],
                        columns=[f'sp{i:02d}' for i in range(10)])


# ---------------------------------------------------------------------------
# Nestedness
# ---------------------------------------------------------------------------

class TestNestedness:

    def test_perfect_nestedness_scores_at_the_extremes(self, perfectly_nested):
        """
        A staircase matrix is maximally nested: NODF approaches 100 and the
        Atmar-Patterson temperature approaches 0.
        """
        analyzer = NestednessAnalyzer()
        analyzer.load_matrix(perfectly_nested)
        metrics = analyzer.calculate_nestedness_metrics(
            metrics=['nodf', 'temperature'])['metrics']

        assert metrics['nodf']['nodf_overall'] > 90
        assert metrics['temperature']['temperature'] < 15

    def test_all_metrics_run(self, community):
        analyzer = NestednessAnalyzer()
        analyzer.load_matrix(community)
        metrics = analyzer.calculate_nestedness_metrics(
            metrics=['nodf', 'temperature', 'c_score', 'togetherness',
                     'wine'])['metrics']
        assert set(metrics) == {'nodf', 'temperature', 'c_score',
                                'togetherness', 'wine'}

    def test_nodf_components_bracket_the_total(self, community):
        analyzer = NestednessAnalyzer()
        analyzer.load_matrix(community)
        nodf = analyzer.calculate_nestedness_metrics(metrics=['nodf'])['metrics']['nodf']
        low = min(nodf['nodf_rows'], nodf['nodf_columns'])
        high = max(nodf['nodf_rows'], nodf['nodf_columns'])
        assert low - 1e-9 <= nodf['nodf_overall'] <= high + 1e-9

    @pytest.mark.parametrize('sort_by', ['total', 'richness', 'abundance'])
    def test_sorting_options_preserve_the_matrix_contents(self, community, sort_by):
        analyzer = NestednessAnalyzer()
        analyzer.load_matrix(community)
        sorted_matrix = analyzer.calculate_nestedness_metrics(
            sort_by=sort_by, metrics=['nodf'])['sorted_matrix']
        assert sorted_matrix.values.sum() == community.values.sum()
        assert set(sorted_matrix.index) == set(community.index)

    def test_binary_conversion_on_abundance_data(self):
        abundance = pd.DataFrame([[0.0, 3.5], [1.2, 0.0]])
        analyzer = NestednessAnalyzer()
        analyzer.load_matrix(abundance)
        assert set(np.unique(analyzer.matrix_data.values)) <= {0, 1}

    def test_unloaded_matrix_raises(self):
        with pytest.raises(ValueError, match='not loaded'):
            NestednessAnalyzer().calculate_nestedness_metrics()

    @pytest.mark.parametrize('model', ['equiprobable', 'proportional',
                                       'fixed_fixed', 'sequential_swap'])
    def test_null_models_preserve_what_they_promise(self, community, model):
        nulls = NullModels(random_state=0).generate_null_matrices(
            community, n_iterations=5, model_type=model)
        assert len(nulls) == 5
        for null in nulls:
            assert null.shape == community.shape
            if model in ('fixed_fixed', 'sequential_swap'):
                # These models are defined by holding both marginal totals.
                np.testing.assert_array_equal(null.values.sum(axis=1),
                                              community.values.sum(axis=1))
                np.testing.assert_array_equal(null.values.sum(axis=0),
                                              community.values.sum(axis=0))
            else:
                assert null.values.sum() == community.values.sum()

    @pytest.mark.parametrize('marginals', ['rows', 'columns', 'both', 'none'])
    def test_equiprobable_marginal_options(self, community, marginals):
        nulls = NullModels(random_state=1).generate_null_matrices(
            community, n_iterations=3, model_type='equiprobable',
            fixed_marginals=marginals)
        assert all(n.shape == community.shape for n in nulls)

    def test_unknown_null_model_raises(self, community):
        with pytest.raises(ValueError, match='Unknown'):
            NullModels().generate_null_matrices(community, n_iterations=1,
                                                model_type='wishful thinking')

    def test_null_models_are_reproducible(self, community):
        first = NullModels(random_state=7).generate_null_matrices(
            community, n_iterations=3, model_type='equiprobable')
        second = NullModels(random_state=7).generate_null_matrices(
            community, n_iterations=3, model_type='equiprobable')
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a.values, b.values)

    def test_a_staircase_has_a_degenerate_null_and_says_so(self, perfectly_nested):
        """
        A staircase matrix is the *only* matrix with its row and column totals
        (Gale-Ryser uniqueness for threshold graphs), so a marginal-preserving
        null model can only reproduce it. The null distribution is therefore a
        point mass and SES must be reported as 0 rather than as inf or NaN.
        """
        results = NestednessSignificance(random_state=0).test_nestedness_significance(
            perfectly_nested, metrics=['nodf'], n_iterations=19)
        nodf = results['significance_tests']['nodf']
        assert nodf['observed'] == pytest.approx(100.0)
        assert nodf['null_std'] == pytest.approx(0.0)
        assert nodf['ses'] == 0
        assert not nodf['significant']

    def test_nestedness_is_detected_against_an_unconstrained_null(self):
        """With marginals free, a nested matrix beats its null."""
        n = 12
        matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            matrix[i, :n - i] = 1
        frame = pd.DataFrame(matrix, index=[f'S{i}' for i in range(n)],
                             columns=[f'sp{i}' for i in range(n)])

        analyzer = NestednessAnalyzer()
        analyzer.load_matrix(frame)
        observed = analyzer.calculate_nestedness_metrics(
            metrics=['nodf'])['metrics']['nodf']['nodf_overall']

        nulls = NullModels(random_state=0).generate_null_matrices(
            frame, n_iterations=49, model_type='equiprobable',
            fixed_marginals='none')
        null_nodf = []
        for null in nulls:
            analyzer.load_matrix(null)
            null_nodf.append(analyzer.calculate_nestedness_metrics(
                metrics=['nodf'])['metrics']['nodf']['nodf_overall'])

        assert observed > np.mean(null_nodf)

    def test_significance_reports_a_two_sided_p_value_in_range(self, community):
        results = NestednessSignificance(random_state=0).test_nestedness_significance(
            community, metrics=['nodf', 'temperature'], n_iterations=49)
        for test in results['significance_tests'].values():
            for key in ('p_greater', 'p_lesser', 'p_two_tailed'):
                assert 0 <= test[key] <= 1

    def test_null_distribution_plot(self, community):
        significance = NestednessSignificance(random_state=0)
        results = significance.test_nestedness_significance(
            community, metrics=['nodf'], n_iterations=29)
        assert significance.plot_null_distributions(results) is not None

    @pytest.mark.parametrize('method', ['jaccard', 'ochiai', 'dice'])
    def test_species_associations(self, community, method):
        result = CooccurrenceAnalysis().species_associations(community,
                                                             method=method)
        matrix = result['associations']
        assert matrix.shape == (10, 10)
        np.testing.assert_allclose(np.diag(matrix.values), 1.0)
        np.testing.assert_allclose(matrix.values, matrix.values.T)

    def test_jaccard_association_matches_hand_calculation(self):
        # sp0 in sites 0,1; sp1 in sites 1,2 -> intersection 1, union 3.
        matrix = pd.DataFrame({'sp0': [1, 1, 0], 'sp1': [0, 1, 1]})
        result = CooccurrenceAnalysis().species_associations(matrix, 'jaccard')
        assert result['associations'].loc['sp0', 'sp1'] == pytest.approx(1 / 3)

    def test_unknown_association_method_raises(self, community):
        with pytest.raises(ValueError, match='Unknown'):
            CooccurrenceAnalysis().species_associations(community, method='tarot')

    def test_checkerboard_analysis(self, community):
        result = CooccurrenceAnalysis().checkerboard_analysis(community)
        assert result['mean_c_score'] >= 0
        assert result['total_checkerboard_units'] >= 0

    def test_checkerboard_of_a_perfect_checkerboard(self):
        """Two species that never co-occur give the maximum checkerboard count."""
        matrix = pd.DataFrame({'sp0': [1, 1, 0, 0], 'sp1': [0, 0, 1, 1]})
        result = CooccurrenceAnalysis().checkerboard_analysis(matrix)
        # 2 sites unique to sp0 x 2 sites unique to sp1 = 4 checkerboard units.
        assert result['total_checkerboard_units'] == 4
        assert result['n_pairs_with_checkerboards'] == 1


# ---------------------------------------------------------------------------
# Static plotting
# ---------------------------------------------------------------------------

class TestVegetationPlotter:

    @pytest.fixture
    def plotter(self):
        return VegetationPlotter()

    @pytest.fixture
    def abundance(self):
        rng = np.random.default_rng(0)
        return pd.DataFrame(rng.integers(0, 20, (12, 8)).astype(float),
                            index=[f'S{i:02d}' for i in range(12)],
                            columns=[f'sp{i}' for i in range(8)])

    @pytest.fixture
    def diversity(self, abundance):
        rng = np.random.default_rng(1)
        return pd.DataFrame({'shannon': rng.uniform(0.5, 2.5, 12),
                             'simpson': rng.uniform(0.2, 0.9, 12),
                             'richness': rng.integers(3, 8, 12)},
                            index=abundance.index)

    @pytest.fixture
    def ordination(self, abundance):
        rng = np.random.default_rng(2)
        return {
            'site_scores': pd.DataFrame(rng.normal(size=(12, 2)),
                                        index=abundance.index,
                                        columns=['PC1', 'PC2']),
            'species_scores': pd.DataFrame(rng.normal(size=(8, 2)),
                                           index=abundance.columns,
                                           columns=['PC1', 'PC2']),
            'explained_variance_ratio': np.array([0.4, 0.25]),
            'method': 'PCA',
        }

    @pytest.mark.parametrize('style', ['default', 'publication', 'presentation'])
    def test_styles(self, style):
        assert VegetationPlotter(style=style) is not None

    def test_diversity_indices_plot(self, plotter, diversity):
        assert plotter.plot_diversity_indices(diversity) is not None
        assert plotter.plot_diversity_indices(diversity,
                                              indices=['shannon']) is not None

    def test_diversity_plot_without_site_labels(self, plotter, diversity):
        assert plotter.plot_diversity_indices(diversity,
                                              site_labels=False) is not None

    def test_diversity_plot_accepts_the_dict_form(self, plotter, diversity):
        """calculate_all_indices returns a frame; some callers pass a dict."""
        assert plotter.plot_diversity_indices(
            {'diversity_indices': diversity}) is not None

    def test_diversity_plot_rejects_indices_that_are_not_present(self, plotter,
                                                                 diversity):
        with pytest.raises(ValueError):
            plotter.plot_diversity_indices(diversity, indices=['charisma'])

    def test_species_accumulation_plot(self, plotter, abundance):
        """Feed it exactly what DiversityAnalyzer produces."""
        from VegZ.diversity import DiversityAnalyzer
        curve = DiversityAnalyzer().species_accumulation_curve(
            abundance, n_permutations=20, random_state=0)
        assert plotter.plot_species_accumulation(curve) is not None
        assert plotter.plot_species_accumulation(curve, show_ci=False) is not None

    def test_rank_abundance_plot(self, plotter, abundance):
        assert plotter.plot_rank_abundance(abundance) is not None
        assert plotter.plot_rank_abundance(abundance, log_scale=False) is not None

    def test_ordination_plot(self, plotter, ordination):
        assert plotter.plot_ordination(ordination) is not None
        assert plotter.plot_ordination(ordination, show_species=True) is not None

    def test_ordination_plot_coloured_by_a_grouping(self, plotter, ordination):
        groups = pd.Series(['x', 'y'] * 6, index=ordination['site_scores'].index)
        assert plotter.plot_ordination(ordination, color_by=groups) is not None
        assert plotter.plot_ordination(ordination, show_labels=False) is not None

    def test_dendrogram_plot(self, plotter, abundance):
        from scipy.cluster.hierarchy import linkage
        result = {'linkage_matrix': linkage(abundance.values, method='ward'),
                  'labels': list(abundance.index)}
        assert plotter.plot_dendrogram(result) is not None

    def test_clusters_on_ordination(self, plotter, ordination, abundance):
        labels = pd.Series([0, 1, 2] * 4, index=abundance.index)
        assert plotter.plot_clusters_on_ordination(ordination, labels) is not None

    def test_environmental_vectors(self, plotter, ordination, abundance):
        """Feed it exactly what environmental_fitting produces."""
        from VegZ.multivariate import MultivariateAnalyzer
        rng = np.random.default_rng(3)
        environment = pd.DataFrame({'elev': rng.uniform(0, 1000, 12),
                                    'ph': rng.uniform(4, 8, 12)},
                                   index=abundance.index)
        fitted = MultivariateAnalyzer().environmental_fitting(
            ordination['site_scores'], environment, random_state=0)
        assert plotter.plot_environmental_vectors(ordination, fitted) is not None

    def test_correlation_matrix(self, plotter, abundance):
        assert plotter.plot_correlation_matrix(abundance) is not None
        assert plotter.plot_correlation_matrix(abundance,
                                               method='spearman') is not None

    def test_functional_space(self, plotter):
        rng = np.random.default_rng(4)
        traits = pd.DataFrame(rng.normal(size=(8, 3)),
                              columns=['sla', 'height', 'seed'])
        assert plotter.plot_functional_space(traits) is not None


# ---------------------------------------------------------------------------
# Interactive dashboards (matplotlib fallback when plotly is absent)
# ---------------------------------------------------------------------------

class TestInteractiveVisualizer:

    @pytest.fixture
    def visualizer(self):
        return InteractiveVisualizer()

    @pytest.fixture
    def diversity_results(self):
        rng = np.random.default_rng(0)
        index = [f'S{i:02d}' for i in range(10)]
        return {'diversity_indices': pd.DataFrame(
            {'shannon': rng.uniform(0.5, 2.5, 10),
             'simpson': rng.uniform(0.2, 0.9, 10),
             'richness': rng.integers(3, 9, 10)}, index=index)}

    def test_diversity_dashboard(self, visualizer, diversity_results):
        dashboard = visualizer.create_diversity_dashboard(diversity_results)
        assert dashboard

    def test_ordination_dashboard(self, visualizer):
        rng = np.random.default_rng(1)
        results = {'site_scores': pd.DataFrame(rng.normal(size=(10, 2)),
                                               columns=['PC1', 'PC2']),
                   'explained_variance_ratio': np.array([0.5, 0.3])}
        assert visualizer.create_ordination_dashboard(results)

    def test_clustering_dashboard_over_an_ordination(self, visualizer):
        rng = np.random.default_rng(3)
        clustering = {'cluster_labels': np.array([0, 0, 1, 1, 2, 2])}
        ordination = {'site_scores': pd.DataFrame(rng.normal(size=(6, 2)),
                                                  columns=['PC1', 'PC2'])}
        assert visualizer.create_clustering_dashboard(clustering, ordination)

    def test_clustering_dashboard_with_silhouette_scores(self, visualizer):
        results = {'cluster_labels': np.array([0, 0, 1, 1]),
                   'validation_metrics': {
                       'silhouette_scores': np.array([0.6, 0.5, 0.4, 0.3])}}
        assert visualizer.create_clustering_dashboard(results)

    @pytest.mark.parametrize('build,expected', [
        ('create_diversity_dashboard', 'diversity'),
        ('create_ordination_dashboard', 'ordination'),
        ('create_clustering_dashboard', 'clustering'),
        ('create_trait_dashboard', 'trait'),
    ])
    def test_every_dashboard_says_so_when_it_finds_nothing_to_plot(
            self, visualizer, build, expected):
        """
        Each panel sits behind an `if key in results` guard, so a results dict
        with unexpected key names used to yield a silent empty dashboard.
        """
        with pytest.warns(UserWarning, match=f'No plottable content.*{expected}'):
            result = getattr(visualizer, build)({'unrelated_key': 1})
        assert result == {}

    def test_trait_dashboard(self, visualizer):
        rng = np.random.default_rng(2)
        site_diversity = pd.DataFrame(
            {'FRic': rng.uniform(0, 5, 6), 'FEve': rng.uniform(0, 1, 6),
             'FDiv': rng.uniform(0, 1, 6), 'FDis': rng.uniform(0, 2, 6)},
            index=[f'S{i}' for i in range(6)])
        assert visualizer.create_trait_dashboard({'site_diversity': site_diversity})

    def test_trait_dashboard_falls_back_to_matplotlib(self, visualizer):
        """Without plotly the trait dashboard used to return {} in silence."""
        rng = np.random.default_rng(4)
        traits = pd.DataFrame({'sla': rng.uniform(5, 30, 8),
                               'height': rng.uniform(0.1, 20, 8)},
                              index=[f'sp{i}' for i in range(8)])
        results = {
            'site_diversity': pd.DataFrame({'FRic': rng.uniform(0, 5, 6)},
                                           index=[f'S{i}' for i in range(6)]),
            'functional_groups': {'functional_groups': [0, 0, 1, 1, 2, 2, 0, 1]},
        }
        with mock.patch('VegZ.interactive_viz.PLOTLY_AVAILABLE', False):
            plots = visualizer.create_trait_dashboard(results, traits)
        assert set(plots) == {'functional_diversity', 'trait_space'}
        assert all(isinstance(fig, plt.Figure) for fig in plots.values())

    def test_save_dashboard(self, visualizer, diversity_results, tmp_path):
        dashboard = visualizer.create_diversity_dashboard(diversity_results)
        saved = visualizer.save_dashboard(dashboard, str(tmp_path / 'dash'))
        assert saved

    def test_quick_diversity_dashboard(self, diversity_results):
        assert quick_diversity_dashboard(diversity_results)

    @pytest.mark.parametrize('fmt', ['html', 'markdown'])
    def test_report_generation(self, diversity_results, fmt):
        report = ReportGenerator().generate_analysis_report(
            {'diversity': diversity_results}, output_format=fmt)
        assert isinstance(report, str) and report.strip()

    def test_report_includes_a_supplied_data_summary(self, diversity_results):
        report = ReportGenerator().generate_analysis_report(
            {'diversity': diversity_results},
            data_summary={'n_sites': 10, 'n_species': 42})
        assert isinstance(report, str) and report.strip()

    def test_save_report(self, diversity_results, tmp_path):
        generator = ReportGenerator()
        content = generator.generate_analysis_report({'diversity': diversity_results})
        path = generator.save_report(content, str(tmp_path / 'report'), 'html')
        assert path

    def test_quick_analysis_report(self, diversity_results, tmp_path):
        """
        Pass an explicit filename: the default writes vegz_report.html into the
        current working directory, so a bare call litters the repo root.
        """
        path = quick_analysis_report({'diversity': diversity_results},
                                     filename=str(tmp_path / 'report'))
        assert path
        assert not (pathlib.Path.cwd() / 'vegz_report.html').exists()


# ---------------------------------------------------------------------------
# Remote sensing
# ---------------------------------------------------------------------------

class TestRemoteSensing:

    def test_unsupported_platform_raises(self):
        with pytest.raises(ValueError, match='Unsupported platform'):
            rs.RemoteSensingAPI().get_vegetation_indices(
                [(40.0, -80.0)], ('2020-01-01', '2020-12-31'), platform='sputnik')

    @pytest.mark.parametrize('api_class', [rs.LandsatAPI, rs.MODISAPI,
                                           rs.SentinelAPI])
    def test_apis_say_plainly_that_earth_engine_is_required(self, api_class):
        with mock.patch.object(rs, 'EE_AVAILABLE', False):
            with pytest.raises(ImportError, match='Earth Engine'):
                api_class().extract_indices([(40.0, -80.0)],
                                            ('2020-01-01', '2020-12-31'),
                                            ['NDVI'])

    def test_index_registry_is_complete(self):
        calculator = rs.VegetationIndexCalculator
        for name in calculator.available_indices:
            assert hasattr(calculator, name), f'{name} advertised but not defined'

    @pytest.mark.parametrize('name,bands', [
        ('ndvi', {'red': 0.1, 'nir': 0.5}),
        ('evi', {'red': 0.1, 'nir': 0.5, 'blue': 0.05}),
        ('evi2', {'red': 0.1, 'nir': 0.5}),
        ('savi', {'red': 0.1, 'nir': 0.5}),
        ('msavi', {'red': 0.1, 'nir': 0.5}),
        ('ndwi', {'green': 0.2, 'nir': 0.5}),
        ('nbr', {'nir': 0.5, 'swir': 0.2}),
        ('gndvi', {'green': 0.2, 'nir': 0.5}),
        ('ndre', {'red_edge': 0.3, 'nir': 0.5}),
        ('arvi', {'red': 0.1, 'nir': 0.5, 'blue': 0.05}),
    ])
    def test_every_index_computes_and_stays_in_range(self, name, bands):
        value = rs.VegetationIndexCalculator.compute(name, **bands)
        assert np.all(np.isfinite(value))
        assert np.all(np.abs(value) <= 1.0 + 1e-9)

    def test_ndvi_matches_its_definition(self):
        red = np.array([0.1, 0.2, 0.3])
        nir = np.array([0.5, 0.4, 0.3])
        expected = (nir - red) / (nir + red)
        np.testing.assert_allclose(
            rs.VegetationIndexCalculator.ndvi(red, nir), expected)

    def test_zero_denominator_yields_the_fill_value_not_an_exception(self):
        result = rs.VegetationIndexCalculator.ndvi(np.array([0.0]),
                                                   np.array([0.0]), fill=-999.0)
        assert result[0] == -999.0

    def test_mismatched_band_shapes_raise(self):
        with pytest.raises(ValueError, match='shape'):
            rs.VegetationIndexCalculator.ndvi(np.zeros(3), np.zeros(4))

    def test_unknown_index_lists_the_valid_ones(self):
        with pytest.raises(ValueError) as excinfo:
            rs.VegetationIndexCalculator.compute('nbvi', red=0.1, nir=0.5)
        assert 'ndvi' in str(excinfo.value)

    def test_compute_rejects_unexpected_bands(self):
        with pytest.raises(TypeError):
            rs.VegetationIndexCalculator.compute('ndvi', red=0.1, nir=0.5,
                                                 microwave=0.2)

    def test_savi_reduces_to_ndvi_when_l_is_zero(self):
        red, nir = np.array([0.1]), np.array([0.5])
        np.testing.assert_allclose(
            rs.VegetationIndexCalculator.savi(red, nir, L=0.0),
            rs.VegetationIndexCalculator.ndvi(red, nir))


# ---------------------------------------------------------------------------
# Coordinate systems
# ---------------------------------------------------------------------------

class TestCoordinateTransformer:

    @pytest.fixture
    def transformer(self):
        return CoordinateTransformer()

    @pytest.fixture
    def points(self):
        return pd.DataFrame({'longitude': [-80.0, -79.5, 0.0],
                             'latitude': [40.0, 41.0, 51.5]})

    @pytest.mark.parametrize('longitude,latitude,expected', [
        (-80.0, 40.0, 'EPSG:32617'),    # Pittsburgh, zone 17N
        (0.5, 51.5, 'EPSG:32631'),      # London, zone 31N
        (-58.0, -34.0, 'EPSG:32721'),   # Buenos Aires, zone 21S
        (139.7, 35.7, 'EPSG:32654'),    # Tokyo, zone 54N
        (7.0, 60.0, 'EPSG:32632'),      # south-west Norway exception
        (15.0, 78.0, 'EPSG:32633'),     # Svalbard exception
    ])
    def test_utm_zone_determination(self, transformer, longitude, latitude,
                                    expected):
        assert transformer.determine_utm_zone(longitude, latitude) == expected

    def test_longitude_wraps_rather_than_producing_zone_61(self, transformer):
        """180 degrees east is zone 1, not the non-existent zone 61."""
        assert transformer.determine_utm_zone(180.0, 0.0) == 'EPSG:32601'
        assert transformer.determine_utm_zone(200.0, 0.0) == \
            transformer.determine_utm_zone(-160.0, 0.0)

    def test_utm_zone_rejects_an_impossible_latitude(self, transformer):
        with pytest.raises(ValueError, match='latitude'):
            transformer.determine_utm_zone(0.0, 120.0)

    def test_identity_transform_is_a_no_op(self, transformer, points):
        result = transformer.transform_coordinates(points, 'EPSG:4326',
                                                   'EPSG:4326')
        np.testing.assert_allclose(result['longitude'], points['longitude'])
        np.testing.assert_allclose(result['latitude'], points['latitude'])

    def test_missing_columns_raise(self, transformer):
        with pytest.raises(ValueError, match='not found|missing'):
            transformer.transform_coordinates(pd.DataFrame({'x': [1.0]}),
                                              'EPSG:4326', 'EPSG:3857')

    @pytest.mark.parametrize('method', ['great_circle', 'euclidean', 'geodesic'])
    def test_distance_matrices_are_symmetric_with_a_zero_diagonal(
            self, transformer, points, method):
        distances = transformer.calculate_distances(points, method=method)
        np.testing.assert_allclose(np.diag(distances), 0.0, atol=1e-9)
        np.testing.assert_allclose(distances, distances.T, atol=1e-6)

    def test_great_circle_reproduces_a_known_distance(self, transformer):
        """London to Paris is about 344 km, returned in metres."""
        frame = pd.DataFrame({'longitude': [-0.1278, 2.3522],
                              'latitude': [51.5074, 48.8566]})
        distances = transformer.calculate_distances(frame, method='great_circle')
        assert distances[0, 1] == pytest.approx(344_000, rel=0.02)

    @pytest.mark.parametrize('method', ['great_circle', 'euclidean', 'geodesic'])
    def test_all_three_methods_agree_and_return_metres(self, transformer, method):
        """
        The spherical, projected and ellipsoidal distances differ only by the
        WGS84 flattening over this baseline, so any unit slip between them
        would show up as a factor of 1000.
        """
        frame = pd.DataFrame({'longitude': [-0.1278, 2.3522],
                              'latitude': [51.5074, 48.8566]})
        distances = transformer.calculate_distances(frame, method=method)
        assert distances[0, 1] == pytest.approx(344_000, rel=0.02)

    def test_coordinates_can_be_given_as_tuples(self, transformer):
        pairs = [(-0.1278, 51.5074), (2.3522, 48.8566)]
        distances = transformer.calculate_distances(pairs, method='great_circle')
        assert distances.shape == (2, 2)

    def test_unknown_distance_method_raises(self, transformer, points):
        with pytest.raises(ValueError, match='[Uu]nknown|[Uu]nsupported'):
            transformer.calculate_distances(points, method='as-the-crow-tunnels')

    def test_spatial_grid_covers_the_bounding_box(self, transformer):
        """A 4x2 degree box at 1 degree resolution gives 8 cell centres."""
        grid = transformer.create_spatial_grid((-80.0, 40.0, -76.0, 42.0),
                                               cell_size=1.0)
        assert len(grid) == 8
        assert set(grid.columns) == {'grid_id', 'x', 'y', 'crs'}
        # Centres sit half a cell inside the bounds.
        assert grid['x'].min() == pytest.approx(-79.5)
        assert grid['x'].max() == pytest.approx(-76.5)
        assert grid['y'].min() == pytest.approx(40.5)
        assert grid['y'].max() == pytest.approx(41.5)

    def test_crs_info(self, transformer):
        info = transformer.get_crs_info('EPSG:4326')
        assert isinstance(info, dict) and info


# ---------------------------------------------------------------------------
# Temporal validation
# ---------------------------------------------------------------------------

class TestTemporalValidator:

    @pytest.fixture
    def validator(self):
        return TemporalValidator()

    @pytest.fixture
    def dated(self):
        return pd.DataFrame({
            'event_date': ['2020-05-01', '2019-03-15', 'not a date',
                           '1700-01-01', '2099-12-31', ''],
            'identification_date': ['2020-06-01', '2019-01-15', '2018-07-01',
                                    '1701-01-01', '2100-01-01', '2020-01-01'],
        })

    def test_validation_flags_each_class_of_problem(self, validator, dated):
        results = validator.validate_dates(dated, 'event_date')
        flags = results['flags']
        assert results['total_records'] == 6
        assert flags['event_date_unparseable'] >= 2   # 'not a date' and ''
        assert flags['event_date_future_dates'] >= 1  # 2099
        assert flags['event_date_very_old'] >= 1      # 1700

    def test_pre_1677_dates_are_reported_as_unparseable(self, validator):
        """
        pandas' nanosecond timestamps start at 1677-09-21, so anything earlier
        cannot be represented and must surface as unparseable rather than being
        silently dropped.
        """
        frame = pd.DataFrame({'d': ['1500-01-01']})
        results = validator.validate_dates(frame, 'd')
        assert results['flags']['d_unparseable'] == 1

    def test_missing_column_is_reported_not_raised(self, validator, dated):
        results = validator.validate_dates(dated, 'no_such_column')
        assert any('not found' in issue for issue in results['issues_found'])

    def test_cross_validation_detects_reversed_ordering(self, validator, dated):
        """identification_date before event_date is a logical inconsistency."""
        results = validator.validate_dates(dated,
                                           ['event_date', 'identification_date'])
        assert 'cross_validation' in results

    def test_temporal_consistency_flags_a_year_wide_gap(self, validator):
        """
        A collection-to-identification gap over a year is flagged, and the flag
        must land on the right row - it is built from a Series indexed only by
        the parseable rows.
        """
        frame = pd.DataFrame({
            'event_date': ['2020-01-01', '2020-01-01', 'not a date',
                           '2020-01-01'],
            'identification_date': ['2020-02-01', '2025-01-01', '2020-03-01',
                                    '2020-01-15'],
        })
        results = validator.validate_dates(frame, ['identification_date'],
                                           event_date_col='event_date')
        assert 'temporal_consistency' in results
        flags = results['temporal_consistency']['flags'][
            'identification_date_extreme_diff']
        assert flags.dtype == bool
        assert flags.tolist() == [False, True, False, False]

    def test_date_component_extraction(self, validator, dated):
        components = validator.extract_date_components(dated, 'event_date')
        for column in ('event_date_year', 'event_date_month', 'event_date_day'):
            assert column in components.columns
        assert components['event_date_year'].dropna().iloc[0] == 2020

    def test_quality_report(self, validator, dated):
        report = validator.generate_temporal_quality_report(dated, 'event_date')
        summary = report['summary_statistics']['event_date']
        assert summary['earliest_date'] == '1700-01-01'
        assert summary['latest_date'] == '2099-12-31'
        assert summary['date_range_years'] == pytest.approx(399.9, abs=0.5)
        assert report['recommendations']

    def test_quality_report_survives_a_span_beyond_the_timedelta_limit(self,
                                                                       validator):
        """
        pandas Timedelta tops out near 292 years, so subtracting two Timestamps
        further apart overflows. A 1700-2099 span must still report a range.
        """
        frame = pd.DataFrame({'d': ['1700-01-01', '2099-12-31']})
        report = validator.generate_temporal_quality_report(frame, 'd')
        assert report['summary_statistics']['d']['date_range_years'] > 390

    @pytest.mark.parametrize('value,year', [
        ('2020-05-01', 2020),
        ('01/05/2020', 2020),
        ('2020', 2020),
        ('May 2020', 2020),
    ])
    def test_common_date_formats_parse(self, validator, value, year):
        parsed, _ = validator._parse_dates_robust(pd.Series([value]))
        assert parsed.notna().iloc[0], f'{value!r} did not parse'
        assert parsed.dt.year.iloc[0] == year

    def test_all_null_column_does_not_crash(self, validator):
        frame = pd.DataFrame({'d': [None, None, None]})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            results = validator.validate_dates(frame, 'd')
        assert results['flags']['d_unparseable'] == 3
