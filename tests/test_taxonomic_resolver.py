"""
Tests for the online taxonomic name resolver.

Every HTTP call is mocked. These tests are about VegZ's own logic - source
selection, fallback, caching, score thresholds, error handling and the shape of
the output - not about whether the remote databases are up. A test suite that
needs the network is a test suite that gets skipped.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import requests

from VegZ.data_management.taxonomic_resolver import (
    TaxonomicResolver, resolve_species_names)


def make_response(payload, status: int = 200):
    """A stand-in for `requests.Response` carrying a JSON body."""
    response = MagicMock()
    response.status_code = status
    response.json.return_value = payload
    response.text = json.dumps(payload)
    response.raise_for_status.side_effect = (
        None if status < 400 else requests.HTTPError(f'HTTP {status}'))
    return response


GBIF_MATCH = {
    'matchType': 'EXACT',
    'confidence': 97,
    'status': 'ACCEPTED',
    'species': 'Quercus robur',
    'canonicalName': 'Quercus robur',
    'family': 'Fagaceae',
    'genus': 'Quercus',
    'kingdom': 'Plantae',
    'phylum': 'Tracheophyta',
    'class': 'Magnoliopsida',
    'order': 'Fagales',
    'usageKey': 2879737,
}

GBIF_NO_MATCH = {'matchType': 'NONE'}


@pytest.fixture
def resolver():
    """A GBIF-only resolver with no inter-request delay."""
    return TaxonomicResolver(sources='gbif', request_delay=0.0)


class TestConstruction:

    def test_default_source_is_wfo(self, capsys):
        r = TaxonomicResolver(request_delay=0.0)
        assert r.sources == ['wfo']

    def test_single_source_as_string(self):
        assert TaxonomicResolver(sources='gbif', request_delay=0.0).sources == ['gbif']

    def test_multiple_sources_preserve_order(self):
        r = TaxonomicResolver(sources=['gbif', 'wfo', 'itis'], request_delay=0.0)
        assert r.sources == ['gbif', 'wfo', 'itis']

    def test_case_insensitive_source_names(self):
        assert TaxonomicResolver(sources='GBIF', request_delay=0.0).sources == ['gbif']

    @pytest.mark.parametrize('bad', ['notadatabase', ['gbif', 'nope']])
    def test_unsupported_source_rejected(self, bad):
        with pytest.raises(ValueError, match='Unsupported source'):
            TaxonomicResolver(sources=bad, request_delay=0.0)

    def test_every_supported_source_has_an_endpoint(self):
        r = TaxonomicResolver(request_delay=0.0)
        for source in r.SUPPORTED_SOURCES:
            assert source in r.API_ENDPOINTS, f'{source} has no endpoint'


class TestResolution:

    def test_successful_resolution_shape(self, resolver):
        with patch.object(resolver._session, 'get',
                          return_value=make_response(GBIF_MATCH)):
            results = resolver.resolve_names(['Quercus robur'], verbose=False,
                                             include_synonyms=False)

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 1
        row = results.iloc[0]
        assert row['original_name'] == 'Quercus robur'
        assert row['accepted_name'] == 'Quercus robur'
        assert row['family'] == 'Fagaceae'
        assert row['genus'] == 'Quercus'
        assert row['source'] == 'GBIF'
        assert row['match_score'] == 97

    def test_expected_columns_present(self, resolver):
        with patch.object(resolver._session, 'get',
                          return_value=make_response(GBIF_MATCH)):
            results = resolver.resolve_names(['Quercus robur'], verbose=False,
                                             include_synonyms=False)

        for column in ('original_name', 'accepted_name', 'match_score',
                       'match_type', 'taxonomic_status', 'family', 'genus',
                       'source'):
            assert column in results.columns

    def test_unmatched_name_is_reported_not_dropped(self, resolver):
        """An unresolvable name must still appear, so counts stay consistent."""
        with patch.object(resolver._session, 'get',
                          return_value=make_response(GBIF_NO_MATCH)):
            results = resolver.resolve_names(['Notaplant fakensis'], verbose=False)

        assert len(results) == 1
        assert results.iloc[0]['original_name'] == 'Notaplant fakensis'

    def test_accepts_a_series_as_input(self, resolver):
        with patch.object(resolver._session, 'get',
                          return_value=make_response(GBIF_MATCH)):
            results = resolver.resolve_names(
                pd.Series(['Quercus robur', 'Quercus robur']), verbose=False,
                include_synonyms=False)
        assert len(results) == 2

    def test_every_input_name_yields_a_row(self, resolver):
        names = ['Quercus robur', 'Pinus sylvestris', 'Betula pendula']
        with patch.object(resolver._session, 'get',
                          return_value=make_response(GBIF_MATCH)):
            results = resolver.resolve_names(names, verbose=False,
                                             include_synonyms=False)
        assert list(results['original_name']) == names


class TestNetworkFailureHandling:

    def test_connection_error_does_not_propagate(self, resolver):
        """A database being unreachable must degrade, not crash the analysis."""
        with patch.object(resolver._session, 'get',
                          side_effect=requests.ConnectionError('no route to host')):
            with pytest.warns(UserWarning):
                results = resolver.resolve_names(['Quercus robur'], verbose=False)

        assert len(results) == 1

    def test_http_error_does_not_propagate(self, resolver):
        with patch.object(resolver._session, 'get',
                          return_value=make_response({}, status=500)):
            with pytest.warns(UserWarning):
                results = resolver.resolve_names(['Quercus robur'], verbose=False)
        assert len(results) == 1

    def test_timeout_does_not_propagate(self, resolver):
        with patch.object(resolver._session, 'get',
                          side_effect=requests.Timeout('timed out')):
            with pytest.warns(UserWarning):
                results = resolver.resolve_names(['Quercus robur'], verbose=False)
        assert len(results) == 1

    def test_malformed_json_does_not_propagate(self, resolver):
        broken = MagicMock()
        broken.status_code = 200
        broken.raise_for_status.return_value = None
        broken.json.side_effect = ValueError('not json')

        with patch.object(resolver._session, 'get', return_value=broken):
            results = resolver.resolve_names(['Quercus robur'], verbose=False)
        assert len(results) == 1


class TestFallbackAndCaching:

    def test_fallback_moves_to_the_next_source(self):
        """With use_fallback, an empty first source must not end the search."""
        resolver = TaxonomicResolver(sources=['gbif', 'wfo'], use_fallback=True,
                                     request_delay=0.0)
        calls = []

        def fake_query(source, name, include_synonyms):
            calls.append(source)
            if source == 'gbif':
                return None
            return {'accepted_name': 'Quercus robur', 'match_score': 90,
                    'source': 'WFO', 'family': 'Fagaceae'}

        with patch.object(resolver, '_query_source', side_effect=fake_query):
            resolver.resolve_names(['Quercus robur'], verbose=False)

        assert calls == ['gbif', 'wfo']

    def test_without_fallback_only_the_first_source_is_tried(self):
        resolver = TaxonomicResolver(sources=['gbif', 'wfo'], use_fallback=False,
                                     request_delay=0.0)
        calls = []

        with patch.object(resolver, '_query_source',
                          side_effect=lambda s, n, i: calls.append(s)):
            resolver.resolve_names(['Quercus robur'], verbose=False)

        assert calls == ['gbif']

    def test_repeated_names_hit_the_cache(self, resolver):
        """The same name twice must cost one request, not two."""
        with patch.object(resolver._session, 'get',
                          return_value=make_response(GBIF_MATCH)) as mock_get:
            resolver.resolve_names(['Quercus robur'] * 4, verbose=False,
                                   include_synonyms=False)
        assert mock_get.call_count == 1

    def test_cache_can_be_disabled(self):
        resolver = TaxonomicResolver(sources='gbif', cache_results=False,
                                     request_delay=0.0)
        with patch.object(resolver._session, 'get',
                          return_value=make_response(GBIF_MATCH)) as mock_get:
            resolver.resolve_names(['Quercus robur'] * 3, verbose=False,
                                   include_synonyms=False)
        assert mock_get.call_count == 3

    def test_clear_cache_forces_a_refetch(self, resolver):
        with patch.object(resolver._session, 'get',
                          return_value=make_response(GBIF_MATCH)) as mock_get:
            resolver.resolve_names(['Quercus robur'], verbose=False,
                                   include_synonyms=False)
            resolver.clear_cache()
            resolver.resolve_names(['Quercus robur'], verbose=False,
                                   include_synonyms=False)
        assert mock_get.call_count == 2


class TestNameCleaning:

    @pytest.mark.parametrize('raw,expected', [
        ('  Quercus   robur  ', 'Quercus robur'),   # collapse internal spaces
        ('Quercus robur L.', 'Quercus robur'),      # strip a trailing authority
        ('Quercus robur 1753', 'Quercus robur'),    # strip a trailing year
        ('Quercus robur', 'Quercus robur'),         # already clean
    ])
    def test_cleaning_strips_whitespace_and_authorities(self, resolver, raw, expected):
        assert resolver._clean_name(raw) == expected

    def test_cleaning_leaves_case_alone(self, resolver):
        """Case is not normalised; the remote APIs match case-insensitively."""
        assert resolver._clean_name('QUERCUS ROBUR') == 'QUERCUS ROBUR'

    def test_cleaning_handles_empty_and_null(self, resolver):
        for value in ('', '   ', None, np.nan):
            assert resolver._clean_name(value) == '' or isinstance(
                resolver._clean_name(value), str)


class TestDataFrameIntegration:

    def test_species_column_auto_detection(self, resolver):
        for column in ('species', 'scientific_name', 'taxon'):
            frame = pd.DataFrame({column: ['Quercus robur'], 'cover': [10]})
            assert resolver._auto_detect_species_column(frame) == column

    def test_auto_detection_returns_none_when_absent(self, resolver):
        frame = pd.DataFrame({'cover': [10], 'elevation': [100]})
        assert resolver._auto_detect_species_column(frame) is None

    def test_resolve_dataframe_preserves_rows_and_adds_columns(self, resolver):
        frame = pd.DataFrame({'species': ['Quercus robur', 'Quercus robur'],
                              'cover': [10, 20]})
        with patch.object(resolver._session, 'get',
                          return_value=make_response(GBIF_MATCH)):
            updated = resolver.resolve_dataframe(frame, species_column='species')

        assert len(updated) == len(frame)
        assert 'cover' in updated.columns
        assert list(updated['cover']) == [10, 20]


class TestFileIO:

    def test_resolve_from_file_reads_csv(self, resolver, tmp_path):
        path = tmp_path / 'species.csv'
        pd.DataFrame({'species': ['Quercus robur']}).to_csv(path, index=False)

        with patch.object(resolver._session, 'get',
                          return_value=make_response(GBIF_MATCH)):
            results = resolver.resolve_from_file(str(path))

        assert len(results) == 1

    def test_missing_file_raises(self, resolver):
        with pytest.raises((FileNotFoundError, ValueError)):
            resolver.resolve_from_file('does_not_exist_anywhere.csv')

    def test_export_round_trips_to_csv(self, resolver, tmp_path):
        with patch.object(resolver._session, 'get',
                          return_value=make_response(GBIF_MATCH)):
            results = resolver.resolve_names(['Quercus robur'], verbose=False,
                                             include_synonyms=False)

        out = tmp_path / 'resolved.csv'
        resolver.export_results(results, str(out))
        assert out.exists()
        assert len(pd.read_csv(out)) == 1


class TestSummary:

    def test_summary_counts_are_consistent(self, resolver):
        with patch.object(resolver._session, 'get',
                          return_value=make_response(GBIF_MATCH)):
            results = resolver.resolve_names(
                ['Quercus robur', 'Pinus sylvestris'], verbose=False,
                include_synonyms=False)

        summary = resolver.get_summary(results)
        assert summary['total_names'] == 2
        assert 0 <= summary.get('resolved', 0) <= 2

    def test_print_summary_runs(self, resolver, capsys):
        with patch.object(resolver._session, 'get',
                          return_value=make_response(GBIF_MATCH)):
            results = resolver.resolve_names(['Quercus robur'], verbose=False,
                                             include_synonyms=False)
        resolver.print_summary(results)
        assert capsys.readouterr().out


class TestConvenienceFunction:

    def test_resolve_species_names_wraps_the_class(self):
        with patch.object(requests.Session, 'get',
                          return_value=make_response(GBIF_MATCH)):
            results = resolve_species_names(['Quercus robur'], sources='gbif')
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 1
