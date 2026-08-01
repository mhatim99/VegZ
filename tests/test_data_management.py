"""
Tests for the data-management layer: parsers, standardisation, transformations,
Darwin Core and coordinate systems.
"""

import numpy as np
import pandas as pd
import pytest

from VegZ.data_management.coordinate_systems import (
    PYPROJ_AVAILABLE, CoordinateTransformer)
from VegZ.data_management.darwin_core import DarwinCoreHandler
from VegZ.data_management.parsers import (
    AgencyDataParser, TurbovegParser, VegetationDataParser)
from VegZ.data_management.remote_sensing import VegetationIndexCalculator
from VegZ.data_management.standardization import (
    FUZZYWUZZY_AVAILABLE, CoordinateStandardizer, DataStandardizer,
    SpeciesNameStandardizer, _DifflibFuzz)
from VegZ.data_management.transformations import DataTransformer


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

class TestVegetationDataParser:

    @pytest.fixture
    def parser(self):
        return VegetationDataParser()

    def test_reads_a_comma_separated_file(self, parser, tmp_path):
        path = tmp_path / 'data.csv'
        path.write_text('site_id,Species_A,Species_B\nS1,5,3\nS2,2,7\n', encoding='utf-8')

        frame = parser.parse(path)
        assert frame.shape == (2, 3)
        # Column names are lower-cased, and `site_id` is normalised to the
        # parser's canonical plot identifier.
        assert list(frame.columns) == ['plot_id', 'species_a', 'species_b']

    def test_identifier_aliases_collapse_to_plot_id(self, parser, tmp_path):
        for alias in ('plot', 'site', 'site_id', 'releve', 'quadrat'):
            path = tmp_path / f'{alias}.csv'
            path.write_text(f'{alias},cover\nS1,5\n', encoding='utf-8')
            assert 'plot_id' in parser.parse(path).columns, alias

    @pytest.mark.parametrize('separator', [',', ';', '\t', '|'])
    def test_delimiter_is_sniffed(self, parser, tmp_path, separator):
        path = tmp_path / 'data.csv'
        path.write_text(
            separator.join(['site', 'a', 'b']) + '\n'
            + separator.join(['S1', '1', '2']) + '\n', encoding='utf-8')

        frame = parser.parse(path)
        assert frame.shape == (1, 3), f'failed for {separator!r}'

    def test_explicit_separator_is_not_duplicated(self, parser, tmp_path):
        """Passing sep through kwargs used to raise 'multiple values for sep'."""
        path = tmp_path / 'data.tsv'
        path.write_text('a\tb\n1\t2\n', encoding='utf-8')
        assert parser.parse(path, format_type='txt', sep='\t').shape == (1, 2)

    def test_explicit_encoding_is_not_duplicated(self, parser, tmp_path):
        path = tmp_path / 'data.csv'
        path.write_text('a,b\n1,2\n', encoding='utf-8')
        assert parser.parse(path, encoding='utf-8').shape == (1, 2)

    def test_latin1_fallback(self, parser, tmp_path):
        path = tmp_path / 'data.csv'
        path.write_bytes('species,cover\nCr\xe8me,5\n'.encode('latin-1'))
        with pytest.warns(UserWarning, match='encoding'):
            frame = parser.parse(path)
        assert len(frame) == 1

    def test_missing_file_raises(self, parser):
        with pytest.raises(FileNotFoundError):
            parser.parse('no_such_file.csv')

    def test_unsupported_extension_raises(self, parser, tmp_path):
        path = tmp_path / 'data.parquet'
        path.write_bytes(b'x')
        with pytest.raises(ValueError, match='Unsupported format'):
            parser.parse(path)

    def test_column_aliases_are_standardised(self, parser, tmp_path):
        path = tmp_path / 'data.csv'
        path.write_text('taxon,cover,lat,lon\nQuercus,5,40.1,-80.2\n', encoding='utf-8')
        frame = parser.parse(path)
        for expected in ('species', 'abundance', 'latitude', 'longitude'):
            assert expected in frame.columns


class TestTurbovegParser:

    def test_positional_rename_does_not_mutate_the_index_array(self, tmp_path):
        """Assigning into df.columns.values is undefined behaviour in pandas."""
        path = tmp_path / 'species.txt'
        path.write_text('c1\tc2\tc3\tc4\n1\tAbies alba\tL.\tPinaceae\n', encoding='utf-8')

        frame = TurbovegParser()._parse_species_list(path)
        assert list(frame.columns) == ['species_nr', 'species_name', 'author', 'family']

    def test_missing_directory_raises(self):
        with pytest.raises(ValueError, match='Directory not found'):
            TurbovegParser().parse_turboveg_export('no_such_directory')

    def test_export_directory_is_scanned(self, tmp_path):
        (tmp_path / 'species.txt').write_text('a\tb\n1\tAbies\n', encoding='utf-8')
        results = TurbovegParser().parse_turboveg_export(tmp_path)
        assert 'species_list' in results


class TestAgencyDataParser:

    def test_unknown_agency_raises(self, tmp_path):
        path = tmp_path / 'd.csv'
        path.write_text('a,b\n1,2\n', encoding='utf-8')
        with pytest.raises(ValueError, match='Unsupported agency'):
            AgencyDataParser().parse_agency_data(path, agency='atlantis')

    @pytest.mark.parametrize('agency', ['usfs', 'nps', 'epa', 'fia'])
    def test_each_agency_parser_runs(self, tmp_path, agency):
        path = tmp_path / 'd.csv'
        path.write_text('plot,spcd,lat,lon\n1,202,40.0,-80.0\n', encoding='utf-8')
        assert len(AgencyDataParser().parse_agency_data(path, agency=agency)) == 1


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------

class TestDataTransformer:

    @pytest.fixture
    def frame(self):
        rng = np.random.default_rng(0)
        return pd.DataFrame(rng.integers(0, 20, (6, 5)).astype(float),
                            columns=list('abcde'))

    def test_unknown_method_raises(self, frame):
        with pytest.raises(ValueError, match='Unknown transformation'):
            DataTransformer().transform(frame, 'not-a-transform')

    def test_transformations_preserve_shape(self, frame):
        transformer = DataTransformer()
        for method in transformer.transformation_methods:
            result = transformer.transform(frame, method)
            assert np.asarray(result).shape == frame.shape, method

    def test_negative_values_are_floored_for_count_transforms(self):
        frame = pd.DataFrame({'a': [-5.0, 3.0], 'b': [2.0, 4.0]})
        result = DataTransformer().sqrt_transform(frame)
        assert (np.asarray(result) >= 0).all()

    def test_empty_rows_do_not_divide_by_zero(self):
        frame = pd.DataFrame({'a': [0.0, 5.0], 'b': [0.0, 5.0]})
        for method in ('hellinger', 'chord', 'wisconsin'):
            result = np.asarray(DataTransformer().transform(frame, method))
            assert np.isfinite(result).all(), method

    def test_log_bases(self, frame):
        transformer = DataTransformer()
        natural = np.asarray(transformer.log_transform(frame, base='natural'))
        base10 = np.asarray(transformer.log_transform(frame, base='log10'))
        np.testing.assert_allclose(natural / np.log(10), base10)

        with pytest.raises(ValueError, match='Unknown logarithm base'):
            transformer.log_transform(frame, base='base-e-ish')

    def test_log_inverse_round_trips(self, frame):
        transformer = DataTransformer()
        transformed = transformer.log_transform(frame)
        restored = transformer.inverse_transform(transformed, 'log')
        np.testing.assert_allclose(np.asarray(restored), frame.values, atol=1e-9)

    def test_sqrt_inverse_round_trips(self, frame):
        transformer = DataTransformer()
        restored = transformer.inverse_transform(
            transformer.sqrt_transform(frame), 'sqrt')
        np.testing.assert_allclose(np.asarray(restored), frame.values, atol=1e-9)

    def test_arcsine_clips_to_valid_proportions(self):
        frame = pd.DataFrame({'a': [-0.5, 0.5, 1.5]})
        result = np.asarray(DataTransformer().arcsine_transform(frame))
        assert np.isfinite(result).all()
        assert (result >= 0).all() and (result <= np.pi / 2).all()

    def test_robust_standardisation(self, frame):
        result = np.asarray(
            DataTransformer().standardize_transform(frame, method='robust'))
        assert np.isfinite(result).all()

    def test_unknown_standardisation_raises(self, frame):
        with pytest.raises(ValueError, match='Unknown standardization'):
            DataTransformer().standardize_transform(frame, method='nope')

    def test_minmax_normalisation_spans_zero_to_one(self, frame):
        result = DataTransformer().normalize_transform(frame, method='minmax')
        values = np.asarray(result)
        assert np.isclose(values.min(), 0.0)
        assert np.isclose(values.max(), 1.0)


# ---------------------------------------------------------------------------
# Species name standardisation
# ---------------------------------------------------------------------------

class TestSpeciesNameStandardizer:

    @pytest.fixture
    def standardizer(self):
        return SpeciesNameStandardizer()

    @pytest.mark.parametrize('raw,expected', [
        ('quercus alba', 'Quercus alba'),
        ('QUERCUS ALBA', 'Quercus alba'),
        ('  Quercus   alba  ', 'Quercus alba'),
    ])
    def test_cleaning_normalises_binomials(self, standardizer, raw, expected):
        assert standardizer.clean_species_name(raw) == expected

    def test_cleaning_handles_null(self, standardizer):
        assert standardizer.clean_species_name(None) == ''
        assert standardizer.clean_species_name(np.nan) == ''

    def test_valid_name_passes(self, standardizer):
        assert standardizer.validate_species_name('Quercus alba')['is_valid']

    @pytest.mark.parametrize('name,category', [
        ('Quercus', 'incomplete_binomial'),
        ('quercus alba', 'capitalization_errors'),
        ('Quercus Alba', 'capitalization_errors'),
        ('Quercus sp.', 'placeholder_names'),
        ('Quercus alba 12345', 'invalid_characters'),
    ])
    def test_error_categories_are_detected(self, standardizer, name, category):
        result = standardizer.validate_species_name(name)
        assert not result['is_valid']
        assert category in result['errors'], result['errors']

    def test_batch_validation_covers_every_name(self, standardizer):
        names = ['Quercus alba', 'quercus sp.', 'Pinus strobus']
        result = standardizer.batch_validate_names(names)
        assert len(result) == len(names)

    def test_error_report_percentages_are_sane(self, standardizer):
        frame = pd.DataFrame({'species': ['Quercus alba', 'quercus sp.',
                                          'Pinus strobus', 'Betula']})
        report = standardizer.generate_error_report(frame, species_column='species')
        percentage = report['summary']['validity_percentage']
        assert 0 <= percentage <= 100


class TestFuzzyFallback:

    def test_flag_is_boolean(self):
        assert isinstance(FUZZYWUZZY_AVAILABLE, bool)

    def test_identical_strings_score_100(self):
        assert _DifflibFuzz.ratio('Quercus alba', 'Quercus alba') == 100

    def test_different_strings_score_less(self):
        assert _DifflibFuzz.ratio('Quercus alba', 'Pinus strobus') < 100

    def test_token_sort_ignores_word_order(self):
        assert _DifflibFuzz.token_sort_ratio('alba Quercus', 'Quercus alba') == 100

    def test_partial_ratio_finds_a_substring(self):
        assert _DifflibFuzz.partial_ratio('alba', 'Quercus alba') == 100

    def test_extract_one_picks_the_best_candidate(self):
        from VegZ.data_management.standardization import _DifflibProcess
        best, score = _DifflibProcess.extractOne(
            'Quercus alba', ['Pinus strobus', 'Quercus alba', 'Acer rubrum'])
        assert best == 'Quercus alba'
        assert score == 100


class TestDataStandardizer:

    def test_column_aliases_are_mapped(self):
        frame = pd.DataFrame({'lat': [40.1], 'lon': [-80.1],
                              'taxon': ['Quercus alba'], 'cover': [10]})
        result = DataStandardizer().standardize_dataset(frame)
        for expected in ('latitude', 'longitude', 'species', 'abundance'):
            assert expected in result.columns

    def test_dates_are_expanded_into_parts(self):
        frame = pd.DataFrame({'sampling_date': ['2020-05-04', '2021-06-15']})
        result = DataStandardizer().standardize_dataset(frame)
        for part in ('year', 'month', 'day', 'day_of_year'):
            assert part in result.columns
        assert result['year'].tolist() == [2020, 2021]

    def test_integration_does_not_mutate_the_inputs(self):
        """Tagging the caller's frames with source_dataset was a side effect."""
        left = pd.DataFrame({'plot_id': [1, 2], 'a': [1, 2]})
        right = pd.DataFrame({'plot_id': [1, 2], 'b': [3, 4]})
        original_columns = list(left.columns)

        DataStandardizer().integrate_datasets([left, right], ['plot_id'])
        assert list(left.columns) == original_columns

    def test_integration_merges_on_shared_keys(self):
        left = pd.DataFrame({'plot_id': [1, 2], 'a': [1, 2]})
        right = pd.DataFrame({'plot_id': [1, 2], 'b': [3, 4]})
        merged = DataStandardizer().integrate_datasets([left, right], ['plot_id'])
        assert 'a' in merged.columns and 'b' in merged.columns

    def test_integration_without_shared_keys_concatenates(self):
        left = pd.DataFrame({'x': [1]})
        right = pd.DataFrame({'y': [2]})
        with pytest.warns(UserWarning, match='No common columns'):
            merged = DataStandardizer().integrate_datasets([left, right], ['plot_id'])
        assert len(merged) == 2

    def test_no_datasets_raises(self):
        with pytest.raises(ValueError, match='No datasets'):
            DataStandardizer().integrate_datasets([], ['plot_id'])


class TestCoordinateStandardizer:

    def test_decimal_coordinates_pass_through(self):
        frame = pd.DataFrame({'latitude': [40.1], 'longitude': [-80.2]})
        result = CoordinateStandardizer().standardize_coordinates(frame)
        assert np.isclose(result['latitude'].iloc[0], 40.1)


# ---------------------------------------------------------------------------
# Darwin Core
# ---------------------------------------------------------------------------

class TestDarwinCoreHandler:

    @pytest.fixture
    def handler(self):
        return DarwinCoreHandler()

    @pytest.fixture
    def occurrence(self):
        return pd.DataFrame({
            'species': ['Quercus alba', 'Pinus strobus'],
            'latitude': [40.0, 41.0],
            'longitude': [-80.0, -81.0],
            'date': ['2020-01-01', '2020-06-15'],
        })

    def test_conversion_produces_dwc_terms(self, handler, occurrence):
        result = handler.convert_to_dwc(occurrence)
        assert len(result) == len(occurrence)
        assert any(term in result.columns
                   for term in ('scientificName', 'decimalLatitude'))

    def test_conversion_adds_required_identifiers(self, handler, occurrence):
        result = handler.convert_to_dwc(occurrence)
        assert 'occurrenceID' in result.columns
        assert result['occurrenceID'].is_unique

    def test_validation_returns_a_report(self, handler, occurrence):
        report = handler.validate_dwc_data(handler.convert_to_dwc(occurrence))
        assert isinstance(report, dict)

    def test_invalid_coordinates_are_counted(self, handler):
        frame = pd.DataFrame({'decimalLatitude': [95.0, 40.0],
                              'decimalLongitude': [-80.0, -200.0]})
        assert handler._validate_coordinates(frame) >= 1

    def test_archive_export_writes_files(self, handler, occurrence, tmp_path):
        handler.export_dwc_archive(handler.convert_to_dwc(occurrence),
                                   str(tmp_path / 'archive'))
        assert any(tmp_path.rglob('*'))


# ---------------------------------------------------------------------------
# Coordinate systems
# ---------------------------------------------------------------------------

class TestCoordinateTransformer:

    @pytest.fixture
    def transformer(self):
        return CoordinateTransformer()

    @pytest.mark.parametrize('longitude,expected_zone', [
        (-75.0, 18), (-123.0, 10), (0.5, 31), (179.0, 60),
    ])
    def test_utm_zone_numbers(self, transformer, longitude, expected_zone):
        code = transformer.determine_utm_zone(longitude, 40.0)
        assert code == f'EPSG:{32600 + expected_zone}'

    def test_southern_hemisphere_uses_the_327xx_band(self, transformer):
        assert transformer.determine_utm_zone(-60.0, -30.0).startswith('EPSG:327')

    def test_longitude_180_does_not_produce_zone_61(self, transformer):
        """int((180+180)/6)+1 is 61, which is not a real UTM zone."""
        code = transformer.determine_utm_zone(180.0, 0.0)
        zone = int(code.split(':')[1]) - 32600
        assert 1 <= zone <= 60

    def test_out_of_range_latitude_raises(self, transformer):
        with pytest.raises(ValueError, match='latitude'):
            transformer.determine_utm_zone(0.0, 120.0)

    def test_norway_exception_zone(self, transformer):
        """Zone 32 is widened over southern Norway."""
        assert transformer.determine_utm_zone(5.0, 60.0) == 'EPSG:32632'

    def test_named_crs_resolution(self, transformer):
        assert transformer._resolve_crs('WGS84') == 'EPSG:4326'
        assert transformer._resolve_crs('EPSG:3857') == 'EPSG:3857'
        assert transformer._resolve_crs('4326') == 'EPSG:4326'

    def test_unknown_crs_raises_with_guidance(self, transformer):
        with pytest.raises(ValueError, match='Unknown CRS'):
            transformer._resolve_crs('MIDDLE_EARTH_GRID')

    def test_great_circle_distance_against_known_value(self, transformer):
        """London to Paris is about 344 km."""
        distance = transformer._great_circle_distance((-0.1278, 51.5074),
                                                      (2.3522, 48.8566))
        assert 340_000 < distance < 350_000

    def test_great_circle_distance_is_zero_for_identical_points(self, transformer):
        assert transformer._great_circle_distance((0.0, 0.0), (0.0, 0.0)) == 0.0

    def test_distance_matrix_is_symmetric(self, transformer):
        coords = [(-0.13, 51.51), (2.35, 48.86), (13.4, 52.52)]
        matrix = transformer.calculate_distances(coords, method='great_circle')
        np.testing.assert_allclose(matrix, matrix.T)
        np.testing.assert_allclose(np.diag(matrix), 0)

    def test_unknown_distance_method_raises(self, transformer):
        with pytest.raises(ValueError, match='Unknown distance method'):
            transformer.calculate_distances([(0.0, 0.0), (1.0, 1.0)], method='taxicab')

    @pytest.mark.skipif(not PYPROJ_AVAILABLE, reason='pyproj not installed')
    def test_round_trip_projection_returns_the_original(self, transformer):
        original = [(-80.0, 40.0), (-75.0, 42.0)]
        projected = transformer.transform_coordinates(original, 'EPSG:4326', 'EPSG:3857')
        restored = transformer.transform_coordinates(projected, 'EPSG:3857', 'EPSG:4326')
        np.testing.assert_allclose(np.array(restored), np.array(original), atol=1e-6)

    @pytest.mark.skipif(PYPROJ_AVAILABLE, reason='pyproj is installed')
    def test_transform_without_pyproj_raises_clearly(self, transformer):
        with pytest.raises(ImportError, match='PyProj'):
            transformer.transform_coordinates([(0.0, 0.0)], 'EPSG:4326', 'EPSG:3857')


# ---------------------------------------------------------------------------
# Vegetation indices
# ---------------------------------------------------------------------------

class TestVegetationIndexCalculator:

    def test_ndvi_known_value(self):
        assert VegetationIndexCalculator.ndvi(0.1, 0.5) == pytest.approx(0.4 / 0.6)

    def test_ndvi_is_zero_when_bands_are_equal(self):
        assert VegetationIndexCalculator.ndvi(0.3, 0.3) == pytest.approx(0.0)

    @pytest.mark.parametrize('index_name', ['ndvi', 'ndwi', 'nbr', 'gndvi', 'ndre', 'arvi'])
    def test_normalised_indices_are_bounded(self, index_name):
        rng = np.random.default_rng(0)
        bands = {name: rng.uniform(0.01, 0.9, 50)
                 for name in VegetationIndexCalculator.available_indices[index_name]}
        values = VegetationIndexCalculator.compute(index_name, **bands)
        assert np.nanmin(values) >= -1.0 and np.nanmax(values) <= 1.0

    def test_savi_with_zero_soil_factor_equals_ndvi(self):
        assert VegetationIndexCalculator.savi(0.1, 0.5, L=0.0) == pytest.approx(
            VegetationIndexCalculator.ndvi(0.1, 0.5))

    @pytest.mark.parametrize('inputs', [
        (0.1, 0.5),                       # scalars
        ([0.1, 0.2], [0.5, 0.6]),         # lists
        (np.array([1, 2]), np.array([5, 6])),   # integer arrays
    ])
    def test_scalars_lists_and_arrays_all_accepted(self, inputs):
        result = VegetationIndexCalculator.ndvi(*inputs)
        assert np.all(np.isfinite(np.asarray(result, dtype=float)))

    def test_mismatched_band_shapes_raise(self):
        """Silent broadcasting would produce a plausible but wrong raster."""
        with pytest.raises(ValueError, match='same shape'):
            VegetationIndexCalculator.ndvi(np.array([0.1, 0.2]), np.array([0.5]))

    def test_zero_denominator_returns_the_fill_value(self):
        assert np.isnan(VegetationIndexCalculator.ndvi(np.array([0.0]),
                                                       np.array([0.0]))[0])
        assert VegetationIndexCalculator.ndvi(np.array([0.0]), np.array([0.0]),
                                              fill=0.0)[0] == 0.0

    def test_two_dimensional_rasters_keep_their_shape(self):
        rng = np.random.default_rng(0)
        red, nir = rng.uniform(0.02, 0.2, (4, 5)), rng.uniform(0.3, 0.6, (4, 5))
        assert VegetationIndexCalculator.ndvi(red, nir).shape == (4, 5)

    def test_compute_dispatcher_rejects_unknown_names(self):
        with pytest.raises(ValueError, match='Unknown index'):
            VegetationIndexCalculator.compute('greenness', red=0.1, nir=0.5)

    def test_compute_reports_missing_bands(self):
        with pytest.raises(ValueError, match='requires the band'):
            VegetationIndexCalculator.compute('evi', red=0.1, nir=0.5)

    def test_msavi_is_defined_for_all_valid_reflectance(self):
        """The discriminant is (2*nir - 1)^2 >= 0 whenever red >= 0."""
        rng = np.random.default_rng(1)
        red, nir = rng.uniform(0, 1, 500), rng.uniform(0, 1, 500)
        assert np.isfinite(VegetationIndexCalculator.msavi(red, nir)).all()
