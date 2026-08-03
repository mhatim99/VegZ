"""
Tests for the analysis and reporting modules that previously had little or no
coverage: machine learning, spatial/temporal validation, visualisation,
interactive visualisation, nestedness null models and the VegData container.
"""

import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from VegZ import VegData  # noqa: E402
from VegZ.data_quality.spatial_validation import SpatialValidator  # noqa: E402
from VegZ.data_quality.temporal_validation import TemporalValidator  # noqa: E402
from VegZ.interactive_viz import (  # noqa: E402
    InteractiveVisualizer, ReportGenerator)
from VegZ.machine_learning import (  # noqa: E402
    MachineLearningAnalyzer, PredictiveModeling, quick_ml_analysis)
from VegZ.nestedness import NestednessSignificance, NullModels  # noqa: E402
from VegZ.visualization import VegetationPlotter  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


@pytest.fixture
def community():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.integers(0, 12, (30, 8)).astype(float),
        index=[f'S{i:02d}' for i in range(30)],
        columns=[f'sp{i}' for i in range(8)])


@pytest.fixture
def ecology_table():
    rng = np.random.default_rng(1)
    n = 60
    frame = pd.DataFrame({
        'elevation': rng.uniform(0, 2000, n),
        'soil_ph': rng.uniform(4, 8, n),
        'habitat': rng.choice(['forest', 'meadow', 'scrub'], n),
    })
    frame['presence'] = (frame.elevation > 1000).astype(int)
    frame['biomass'] = 0.01 * frame.elevation + 2 * frame.soil_ph + rng.normal(0, 2, n)
    return frame


# ---------------------------------------------------------------------------
# VegData
# ---------------------------------------------------------------------------

class TestVegData:

    def test_basic_properties(self, community):
        data = VegData(community)
        assert data.n_sites == 30
        assert data.n_species == 8
        assert len(data) == 30
        assert 'sites' in repr(data)

    def test_alignment_across_all_components(self):
        rng = np.random.default_rng(0)
        sites = [f'S{i}' for i in range(6)]
        species = [f'sp{i}' for i in range(5)]
        abundance = pd.DataFrame(rng.integers(1, 9, (6, 5)).astype(float),
                                 index=sites, columns=species)
        environment = pd.DataFrame({'elev': rng.uniform(0, 100, 5)}, index=sites[:5])
        traits = pd.DataFrame({'species': species[:4], 'sla': rng.uniform(1, 9, 4)})
        phylogeny = pd.DataFrame(np.zeros((3, 3)),
                                 index=species[:3], columns=species[:3])

        with pytest.warns(UserWarning, match='dropped'):
            data = VegData(abundance, environment, traits, phylogeny)

        assert data.n_sites == 5 and data.n_species == 3
        assert list(data.environment.index) == list(data.site_ids)
        assert list(data.traits.index) == list(data.species_ids)
        assert list(data.phylogeny.index) == list(data.species_ids)

    def test_alignment_is_deterministic(self):
        """set() intersection order varies between runs; ours must not."""
        rng = np.random.default_rng(0)
        species = [f'sp{i:02d}' for i in range(10)]
        abundance = pd.DataFrame(rng.integers(1, 5, (4, 10)).astype(float),
                                 columns=species)
        traits = pd.DataFrame({'species': species, 't': rng.uniform(0, 1, 10)})

        orders = [list(VegData(abundance.copy(), traits=traits.copy()).species_ids)
                  for _ in range(5)]
        assert all(order == species for order in orders)

    def test_alignment_report_names_the_dropped_labels(self):
        abundance = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]}, index=['S1', 'S2'])
        traits = pd.DataFrame({'species': ['a'], 'x': [1.0]})
        with pytest.warns(UserWarning):
            data = VegData(abundance, traits=traits)
        assert data.alignment_report['species_dropped_from_species_matrix'] == ['b']

    @pytest.mark.parametrize('frame,message', [
        (pd.DataFrame({'a': [1.0], 'name': ['x']}), 'numeric'),
        (pd.DataFrame({'a': [1.0, -2.0]}), 'non-negative'),
        (pd.DataFrame(), 'empty'),
    ])
    def test_invalid_species_matrices_are_rejected(self, frame, message):
        with pytest.raises(ValueError, match=message):
            VegData(frame)

    def test_duplicate_labels_are_rejected(self):
        with pytest.raises(ValueError, match='Duplicate'):
            VegData(pd.DataFrame({'a': [1.0, 2.0]}, index=['x', 'x']))

    def test_missing_values_become_absences_with_a_warning(self):
        with pytest.warns(UserWarning, match='missing values'):
            data = VegData(pd.DataFrame({'a': [1.0, np.nan], 'b': [2.0, 3.0]}))
        assert data.species.iloc[1, 0] == 0

    def test_no_shared_labels_raises(self, community):
        with pytest.raises(ValueError, match='share no site labels'):
            VegData(community, pd.DataFrame({'e': [1.0]}, index=['nowhere']))

    def test_subset_by_label_and_by_mask(self, community):
        data = VegData(community)
        assert data.subset(sites=['S00', 'S01']).n_sites == 2
        assert data.subset(sites=[True, False] * 15).n_sites == 15

    def test_subset_rejects_unknown_labels(self, community):
        with pytest.raises(KeyError, match='Unknown sites'):
            VegData(community).subset(sites=['nowhere'])

    def test_subset_rejects_a_wrong_length_mask(self, community):
        with pytest.raises(ValueError, match='length'):
            VegData(community).subset(sites=[True, False])

    def test_drop_empty_removes_rare_species(self):
        frame = pd.DataFrame({'common': [1.0, 1.0, 1.0], 'rare': [1.0, 0.0, 0.0]})
        assert VegData(frame).drop_empty(min_occurrences=2).n_species == 1

    def test_transform_returns_a_new_object(self, community):
        data = VegData(community)
        transformed = data.transform('hellinger')
        assert transformed is not data
        np.testing.assert_allclose(
            (transformed.species.values ** 2).sum(axis=1), 1.0)

    def test_summary_reports_fill_and_richness(self, community):
        summary = VegData(community).summary()
        assert summary['n_sites'] == 30
        assert 0 <= summary['fill'] <= 1
        assert summary['min_richness'] <= summary['max_richness']

    def test_to_vegz_is_preconfigured(self, community):
        environment = pd.DataFrame({'elev': np.arange(30.0)}, index=community.index)
        analyzer = VegData(community, environment).to_vegz()
        assert analyzer.species_matrix is not None
        assert analyzer.environmental_data is not None
        # No species-column guessing, so no warning.
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            analyzer.calculate_diversity()

    def test_from_files_round_trip(self, community, tmp_path):
        path = tmp_path / 'species.csv'
        community.to_csv(path)
        data = VegData.from_files(str(path))
        assert data.n_sites == community.shape[0]
        assert data.n_species == community.shape[1]

    def test_from_files_finds_species_names_in_an_unnamed_index_column(
            self, community, tmp_path):
        """
        A traits table written by a plain `to_csv()` puts the species names in
        an unnamed leading column. Reading that back as a data column left the
        traits on a RangeIndex, and alignment then failed with "share no
        species labels" on a file the user had every reason to expect to work.
        """
        species_path = tmp_path / 'species.csv'
        traits_path = tmp_path / 'traits.csv'
        community.to_csv(species_path)
        traits = pd.DataFrame({'sla': np.linspace(5, 30, community.shape[1])},
                              index=community.columns)
        traits.to_csv(traits_path)

        data = VegData.from_files(str(species_path),
                                  traits_path=str(traits_path))
        assert data.traits.index.equals(data.species.columns)

    def test_from_files_still_prefers_a_named_species_column(self, community,
                                                             tmp_path):
        species_path = tmp_path / 'species.csv'
        traits_path = tmp_path / 'traits.csv'
        community.to_csv(species_path)
        pd.DataFrame({
            'species': list(community.columns),
            'sla': np.linspace(5, 30, community.shape[1]),
        }).to_csv(traits_path, index=False)

        data = VegData.from_files(str(species_path),
                                  traits_path=str(traits_path))
        assert data.traits.index.equals(data.species.columns)
        assert 'species' not in data.traits.columns


# ---------------------------------------------------------------------------
# Machine learning
# ---------------------------------------------------------------------------

class TestMachineLearning:

    def test_categorical_predictors_are_encoded(self, ecology_table):
        _, _, names = MachineLearningAnalyzer().prepare_data(
            ecology_table, 'presence', ['elevation', 'habitat'])
        assert any(name.startswith('habitat_') for name in names)

    def test_categorical_encoding_can_be_refused(self, ecology_table):
        with pytest.raises(ValueError, match='Non-numeric'):
            MachineLearningAnalyzer().prepare_data(
                ecology_table, 'presence', ['elevation', 'habitat'],
                encode_categorical=False)

    def test_missing_values_are_imputed(self):
        frame = pd.DataFrame({'x': [1.0, np.nan, 3.0, 4.0], 'y': [0, 1, 0, 1]})
        X, _, _ = MachineLearningAnalyzer().prepare_data(frame, 'y', ['x'])
        assert np.isfinite(X).all()

    def test_dropping_incomplete_rows(self):
        frame = pd.DataFrame({'x': [1.0, np.nan, 3.0], 'y': [0, 1, 0]})
        X, y, _ = MachineLearningAnalyzer().prepare_data(
            frame, 'y', ['x'], handle_missing='drop')
        assert len(X) == 2 and len(y) == 2

    @pytest.mark.parametrize('model_type', ['rf', 'gbm', 'logistic'])
    def test_binary_target_always_uses_a_classifier(self, ecology_table, model_type):
        result = MachineLearningAnalyzer().habitat_suitability_modeling(
            ecology_table, 'presence', ['elevation', 'soil_ph'],
            model_type=model_type)
        probabilities = result['predictions']['suitability_probability']
        assert np.all((probabilities >= 0) & (probabilities <= 1))
        assert 'accuracy' in result['performance']

    def test_continuous_target_uses_a_regressor(self, ecology_table):
        result = MachineLearningAnalyzer().habitat_suitability_modeling(
            ecology_table, 'biomass', ['elevation', 'soil_ph'], model_type='gbm')
        assert 'r2' in result['performance']

    def test_logistic_on_a_continuous_target_is_refused(self, ecology_table):
        with pytest.raises(ValueError, match='binary presence/absence'):
            MachineLearningAnalyzer().habitat_suitability_modeling(
                ecology_table, 'biomass', ['elevation'], model_type='logistic')

    def test_cross_validation_runs(self, ecology_table):
        """cross_val_score has no random_state parameter; passing one raised."""
        result = MachineLearningAnalyzer().habitat_suitability_modeling(
            ecology_table, 'presence', ['elevation'], cross_validation=True)
        assert result['cv_scores'] is not None

    def test_class_weighting_is_accepted(self, ecology_table):
        imbalanced = ecology_table.copy()
        imbalanced['rare'] = 0
        imbalanced.loc[imbalanced.index[:4], 'rare'] = 1
        result = MachineLearningAnalyzer().habitat_suitability_modeling(
            imbalanced, 'rare', ['elevation'], class_weight='balanced')
        assert 'accuracy' in result['performance']

    def test_unknown_model_type_raises(self, ecology_table):
        with pytest.raises(ValueError, match='Unknown model type'):
            MachineLearningAnalyzer().habitat_suitability_modeling(
                ecology_table, 'presence', ['elevation'], model_type='crystal-ball')

    def test_species_identification(self, ecology_table):
        result = MachineLearningAnalyzer().species_identification(
            ecology_table, ['elevation', 'soil_ph'], 'habitat', model_types=['rf'])
        assert 0 <= result['performance']['rf']['accuracy'] <= 1
        assert result['best_model'] == 'rf'

    def test_singleton_class_gives_an_actionable_error(self, ecology_table):
        frame = pd.concat([
            ecology_table.assign(group='a').head(25),
            ecology_table.assign(group='b').head(24),
            ecology_table.assign(group='c').head(1),
        ])
        with pytest.raises(ValueError, match='single'):
            MachineLearningAnalyzer().species_identification(
                frame, ['elevation'], 'group', model_types=['rf'])

    def test_biomass_prediction(self, ecology_table):
        result = MachineLearningAnalyzer().biomass_prediction(
            ecology_table, 'biomass', ['elevation', 'soil_ph'],
            optimize_hyperparameters=False)
        assert set(result['performance']) >= {'mse', 'rmse', 'mae', 'r2'}

    def test_anomaly_detection_flags_an_extreme_point(self, ecology_table):
        frame = ecology_table.copy()
        frame.loc[frame.index[0], 'elevation'] = 1e6
        result = MachineLearningAnalyzer().ecological_anomaly_detection(
            frame, ['elevation', 'soil_ph'])
        assert result['n_anomalies'] >= 1
        assert result['is_anomaly'][0]

    def test_community_classification_with_and_without_k(self, community):
        analyzer = MachineLearningAnalyzer()
        columns = list(community.columns)
        assert analyzer.community_classification(
            community, columns, n_communities=3)['n_communities'] == 3
        # Automatic k must not raise IndexError on small ranges.
        assert analyzer.community_classification(
            community.head(6), columns)['n_communities'] >= 1

    @pytest.mark.parametrize('method', ['pca', 'tsne'])
    def test_dimensionality_reduction(self, community, method):
        result = MachineLearningAnalyzer().dimensionality_reduction(
            community, list(community.columns), method=method)
        assert result['reduced_df'].shape == (len(community), 2)

    def test_predictive_modeling_sdm(self):
        rng = np.random.default_rng(2)
        n = 60
        presence = pd.DataFrame({
            'longitude': rng.uniform(-80, -70, n),
            'latitude': rng.uniform(40, 45, n)})
        presence['presence'] = (presence.latitude > 42.5).astype(int)
        environment = presence[['longitude', 'latitude']].copy()
        environment['elevation'] = rng.uniform(0, 2000, n)

        result = PredictiveModeling().species_distribution_modeling(
            presence, environment, 'presence', ['longitude', 'latitude'],
            ['elevation'])
        assert result['best_model'] in result['models']

    def test_quick_ml_analysis(self, ecology_table):
        result = quick_ml_analysis(
            ecology_table[['elevation', 'soil_ph', 'habitat']],
            'habitat', analysis_type='classification')
        assert 'performance' in result

    def test_plotting_helpers(self, ecology_table):
        analyzer = MachineLearningAnalyzer()
        regression = analyzer.biomass_prediction(
            ecology_table, 'biomass', ['elevation'], optimize_hyperparameters=False)
        assert analyzer.plot_model_performance(regression, 'regression') is not None

        classification = analyzer.species_identification(
            ecology_table, ['elevation'], 'habitat', model_types=['rf'])
        assert analyzer.plot_model_performance(classification, 'classification') is not None

        with pytest.raises(ValueError, match='Unknown model type'):
            analyzer.plot_model_performance(regression, 'clairvoyance')


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------

class TestSpatialValidator:

    @pytest.fixture
    def validator(self):
        return SpatialValidator()

    @pytest.fixture
    def coordinates(self):
        return pd.DataFrame({
            'latitude': [40.1234, 200.0, np.nan, 0.0, 51.4829],
            'longitude': [-80.5678, 10.0, 5.0, 0.0, -0.2945],
            'country': ['United States', 'X', 'Y', 'Z', 'United Kingdom'],
        })

    def test_flag_categories(self, validator, coordinates):
        result = validator.validate_coordinates(coordinates)
        assert result['total_records'] == 5
        assert result['flags']['missing_coordinates'] == 1
        assert result['flags']['invalid_ranges'] == 1
        assert result['flags']['zero_coordinates'] == 1

    def test_institution_coordinates_are_flagged(self, validator, coordinates):
        """Kew Gardens is in the reference list of institution coordinates."""
        assert validator.validate_coordinates(coordinates)['flags']['near_institutions'] >= 1

    def test_missing_columns_are_reported_not_raised(self, validator):
        result = validator.validate_coordinates(pd.DataFrame({'a': [1]}))
        assert result['issues_found']

    @pytest.mark.parametrize('method', ['iqr', 'zscore', 'isolation_forest'])
    def test_outlier_methods_all_run(self, validator, coordinates, method):
        """The z-score branch used to raise NameError: scipy stats was unimported."""
        flags = validator.detect_geographic_outliers(coordinates, method=method)
        assert len(flags) == len(coordinates)
        assert flags.dtype == bool

    def test_unknown_outlier_method_raises(self, validator, coordinates):
        with pytest.raises(ValueError, match='Unknown method'):
            validator.detect_geographic_outliers(coordinates, method='vibes')

    def test_country_derivation_does_not_need_geopandas(self, validator, coordinates):
        result = validator.derive_country_from_coordinates(coordinates)
        assert result.loc[0, 'derived_country'] == 'United States'

    def test_country_derivation_without_coordinate_columns(self, validator):
        with pytest.warns(UserWarning, match='not found'):
            result = validator.derive_country_from_coordinates(pd.DataFrame({'a': [1]}))
        assert 'derived_country' in result.columns

    def test_quality_report(self, validator, coordinates):
        report = validator.generate_spatial_quality_report(coordinates,
                                                           country_col='country')
        assert report['dataset_summary']['total_records'] == 5
        assert isinstance(report['recommendations'], list)

    def test_quality_report_without_coordinates_does_not_raise(self, validator):
        report = validator.generate_spatial_quality_report(pd.DataFrame({'a': [1]}))
        assert report['geographic_outliers']['count'] == 0
        assert report['recommendations']

    def test_precision_assessment(self, validator):
        frame = pd.DataFrame({'latitude': [40.0, 40.12345],
                              'longitude': [-80.0, -80.12345]})
        summary = validator.validate_coordinates(frame)['precision_assessment']
        assert summary['mean_decimal_places'] >= 0


class TestTemporalValidator:

    @pytest.fixture
    def validator(self):
        return TemporalValidator()

    @pytest.fixture
    def dates(self):
        return pd.DataFrame({
            'date': ['2020-05-04', '1900-01-01', 'not a date', '2035-01-01',
                     '1750-03-02'],
        })

    def test_validation_flags(self, validator, dates):
        result = validator.validate_dates(dates, 'date')
        assert result['total_records'] == 5
        assert any('unparseable' in key for key in result['flags'])
        assert any('future' in key for key in result['flags'])

    def test_missing_column_is_reported(self, validator, dates):
        result = validator.validate_dates(dates, 'nonexistent')
        assert result['issues_found']

    def test_robust_parsing_handles_several_formats(self, validator):
        series = pd.Series(['2020-05-04', '05/04/2020', '2020/05/04', 'rubbish'])
        parsed, success = validator._parse_dates_robust(series)
        assert success.sum() >= 3
        assert not success.iloc[3]

    def test_future_dates_detected(self, validator):
        parsed, _ = validator._parse_dates_robust(pd.Series(['2099-01-01']))
        assert validator._detect_future_dates(parsed).iloc[0]

    def test_very_old_dates_detected(self, validator):
        parsed, _ = validator._parse_dates_robust(pd.Series(['1700-01-01']))
        assert validator._detect_very_old_dates(parsed).iloc[0]

    def test_pre_1677_dates_are_unparseable_not_silently_dropped(self, validator):
        """datetime64[ns] cannot represent them; they must show as unparseable."""
        parsed, success = validator._parse_dates_robust(pd.Series(['1500-01-01']))
        assert not success.iloc[0]

    def test_quality_report(self, validator, dates):
        report = validator.generate_temporal_quality_report(dates, ['date'])
        assert isinstance(report, dict)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

class TestVegetationPlotter:

    @pytest.fixture
    def plotter(self):
        return VegetationPlotter()

    @pytest.fixture
    def diversity(self, community):
        from VegZ.diversity import DiversityAnalyzer
        return DiversityAnalyzer().calculate_all_indices(community)

    @pytest.fixture
    def ordination(self, community):
        from VegZ.multivariate import MultivariateAnalyzer
        return MultivariateAnalyzer().pca_analysis(community)

    def test_construction_does_not_change_global_style(self):
        """Building a plotter must not restyle the caller's other figures."""
        before = dict(plt.rcParams)
        VegetationPlotter()
        assert plt.rcParams['figure.figsize'] == before['figure.figsize']

    def test_unknown_style_warns_and_falls_back(self):
        with pytest.warns(UserWarning, match='not available'):
            VegetationPlotter(style='not-a-real-style')

    def test_accepts_the_analyzer_dataframe_directly(self, plotter, diversity):
        """The plotting layer must consume the analysis layer's own output."""
        assert plotter.plot_diversity_indices(diversity, ['shannon', 'simpson'])

    def test_accepts_the_legacy_dict_form(self, plotter, diversity):
        payload = {'diversity_indices': diversity.T.to_dict()}
        assert plotter.plot_diversity_indices(payload, ['shannon'])

    def test_uninterpretable_input_raises(self, plotter):
        with pytest.raises(ValueError):
            plotter.plot_diversity_indices(['not', 'a', 'frame'])

    def test_unknown_index_raises(self, plotter, diversity):
        with pytest.raises(ValueError, match='None of the specified'):
            plotter.plot_diversity_indices(diversity, ['not_an_index'])

    def test_ordination_plot_with_species_arrows(self, plotter, ordination):
        assert plotter.plot_ordination(ordination, show_species=True)

    def test_ordination_plot_requires_site_scores(self, plotter):
        with pytest.raises(ValueError, match='Site scores'):
            plotter.plot_ordination({'nothing': 1})

    def test_rank_abundance_for_a_named_site(self, plotter, community):
        assert plotter.plot_rank_abundance(community, site='S00')

    def test_rank_abundance_rejects_an_unknown_site(self, plotter, community):
        with pytest.raises(ValueError, match='not found'):
            plotter.plot_rank_abundance(community, site='nowhere')

    def test_environmental_vectors_consume_envfit_output(self, plotter, community,
                                                         ordination):
        from VegZ.multivariate import MultivariateAnalyzer

        rng = np.random.default_rng(0)
        environment = pd.DataFrame({'elev': rng.uniform(0, 100, len(community))},
                                   index=community.index)
        fit = MultivariateAnalyzer().environmental_fitting(
            ordination['site_scores'].iloc[:, :2], environment,
            permutations=49, random_state=0)

        # Both the whole result and just its 'vectors' entry must work.
        assert plotter.plot_environmental_vectors(ordination, fit)
        assert plotter.plot_environmental_vectors(ordination, fit['vectors'])

    def test_dendrogram_requires_a_linkage_matrix(self, plotter):
        with pytest.raises(ValueError, match='Linkage matrix'):
            plotter.plot_dendrogram({})

    def test_clusters_on_ordination(self, plotter, ordination, community):
        labels = ([0] * 10 + [1] * 10 + [2] * 10)
        assert plotter.plot_clusters_on_ordination(ordination, labels)

    def test_correlation_matrix(self, plotter, community):
        assert plotter.plot_correlation_matrix(community.corr())

    def test_functional_space_needs_two_traits(self, plotter):
        traits = pd.DataFrame({'a': [1.0, 2.0, 3.0]}, index=list('xyz'))
        with pytest.raises(ValueError, match='At least 2'):
            plotter.plot_functional_space(traits)


class TestInteractiveVisualizer:

    @pytest.fixture
    def diversity_payload(self, community):
        from VegZ.diversity import DiversityAnalyzer
        return {'diversity_indices':
                DiversityAnalyzer().calculate_all_indices(community).T.to_dict()}

    def test_diversity_dashboard(self, diversity_payload):
        assert InteractiveVisualizer().create_diversity_dashboard(diversity_payload)

    def test_static_fallback_when_plotly_is_absent(self, monkeypatch,
                                                   diversity_payload):
        import VegZ.interactive_viz as module
        monkeypatch.setattr(module, 'PLOTLY_AVAILABLE', False)
        with pytest.warns(UserWarning, match='Plotly not available'):
            plots = InteractiveVisualizer().create_diversity_dashboard(diversity_payload)
        assert 'diversity_comparison' in plots

    def test_clustering_dashboard_handles_vegz_result_keys(self, monkeypatch,
                                                           community):
        from scipy.cluster.hierarchy import linkage

        import VegZ.interactive_viz as module
        monkeypatch.setattr(module, 'PLOTLY_AVAILABLE', False)

        results = {
            'cluster_labels': pd.Series([0, 1] * 15, index=community.index),
            'linkage_matrix': linkage(community.values),
            'silhouette_scores': pd.Series(np.linspace(-0.2, 0.8, 30),
                                           index=community.index),
            'site_labels': list(community.index),
        }
        plots = InteractiveVisualizer().create_clustering_dashboard(results)
        assert {'dendrogram', 'cluster_sizes', 'silhouette_hist'} <= set(plots)

    def test_clustering_dashboard_warns_when_nothing_is_plottable(self, monkeypatch):
        # The matplotlib fallback reports an empty dashboard through the same
        # _check_dashboard message as the diversity, ordination and trait
        # panels; this used to assert a clustering-only wording, which could
        # not hold at the same time as the equivalent assertion in
        # test_support_modules.py once plotly was absent.
        import VegZ.interactive_viz as module
        monkeypatch.setattr(module, 'PLOTLY_AVAILABLE', False)
        with pytest.warns(UserWarning, match='No plottable content.*clustering'):
            plots = InteractiveVisualizer().create_clustering_dashboard({'x': 1})
        assert plots == {}

    def test_report_generation(self, diversity_payload):
        report = ReportGenerator().generate_analysis_report(
            {'diversity': diversity_payload})
        assert isinstance(report, str) and report

    def test_report_without_jinja2(self, monkeypatch, diversity_payload):
        import VegZ.interactive_viz as module
        monkeypatch.setattr(module, 'JINJA2_AVAILABLE', False)
        report = ReportGenerator().generate_analysis_report(
            {'diversity': diversity_payload})
        assert isinstance(report, str) and report

    def test_report_can_be_saved(self, diversity_payload, tmp_path):
        generator = ReportGenerator()
        report = generator.generate_analysis_report({'diversity': diversity_payload})
        path = generator.save_report(report, str(tmp_path / 'report.html'))
        assert path


# ---------------------------------------------------------------------------
# Null models
# ---------------------------------------------------------------------------

class TestNullModels:

    @pytest.fixture
    def incidence(self, community):
        return (community > 0).astype(int)

    @pytest.mark.parametrize('model', ['equiprobable', 'proportional',
                                       'fixed_fixed', 'sequential_swap'])
    def test_every_model_produces_matrices_of_the_right_shape(self, incidence, model):
        matrices = NullModels(random_state=0).generate_null_matrices(
            incidence, n_iterations=3, model_type=model)
        assert len(matrices) == 3
        assert all(m.shape == incidence.shape for m in matrices)

    @pytest.mark.parametrize('model', ['fixed_fixed', 'sequential_swap'])
    def test_swap_models_preserve_both_marginals(self, incidence, model):
        null = NullModels(random_state=0).generate_null_matrices(
            incidence, n_iterations=1, model_type=model)[0]
        np.testing.assert_array_equal(null.sum(axis=1).values,
                                      incidence.sum(axis=1).values)
        np.testing.assert_array_equal(null.sum(axis=0).values,
                                      incidence.sum(axis=0).values)

    def test_row_marginals_preserved_when_requested(self, incidence):
        null = NullModels(random_state=0).generate_null_matrices(
            incidence, 1, 'equiprobable', fixed_marginals='rows')[0]
        np.testing.assert_array_equal(null.sum(axis=1).values,
                                      incidence.sum(axis=1).values)

    def test_unknown_model_raises(self, incidence):
        with pytest.raises(ValueError, match='Unknown null model'):
            NullModels().generate_null_matrices(incidence, 1, 'wishful-thinking')

    def test_generation_is_reproducible(self, incidence):
        first = NullModels(random_state=7).generate_null_matrices(
            incidence, 2, 'sequential_swap')
        second = NullModels(random_state=7).generate_null_matrices(
            incidence, 2, 'sequential_swap')
        for a, b in zip(first, second):
            pd.testing.assert_frame_equal(a, b)

    def test_significance_p_values_are_valid_probabilities(self, community):
        result = NestednessSignificance(random_state=0).test_nestedness_significance(
            community, ['nodf'], n_iterations=19)
        test = result['significance_tests']['nodf']
        for key in ('p_greater', 'p_lesser', 'p_two_tailed'):
            assert 0 < test[key] <= 1, key
