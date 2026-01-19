"""
Taxonomic name resolution module for vegetation data.

Resolves and validates plant species names against online taxonomic databases.

Copyright (c) 2025 Mohamed Z. Hatim
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import requests
import time
import warnings
from functools import lru_cache
import json
import re


class TaxonomicResolver:
    """
    Resolve and validate plant species names against online taxonomic databases.

    Supported sources (in default fallback order):
    - WFO (World Flora Online) - default
    - POWO (Plants of the World Online) - Kew Gardens
    - IPNI (International Plant Names Index)
    - ITIS (Integrated Taxonomic Information System)
    - GBIF (Global Biodiversity Information Facility / Catalogue of Life backbone)

    Parameters:
    -----------
    sources : str or list, optional
        Source(s) to use for name resolution. If None, uses 'wfo' (World Flora Online).
        Can be a single source string or a list for fallback order.
        Supported: 'wfo', 'powo', 'ipni', 'itis', 'gbif'
    use_fallback : bool
        If True and multiple sources provided, try next source if first fails.
        Default is True.
    cache_results : bool
        If True, cache API results to avoid repeated calls. Default is True.
    request_delay : float
        Delay between API requests in seconds to avoid rate limiting. Default is 0.1.
    timeout : int
        Request timeout in seconds. Default is 30.

    Example:
    --------
    >>> from VegZ.data_management import TaxonomicResolver
    >>>
    >>> # Using default (WFO)
    >>> resolver = TaxonomicResolver()
    >>> results = resolver.resolve_names(['Quercus robur', 'Pinus sylvestris'])
    >>>
    >>> # Using multiple sources with fallback
    >>> resolver = TaxonomicResolver(sources=['wfo', 'powo', 'gbif'], use_fallback=True)
    >>> results = resolver.resolve_names(['Quercus robur', 'Unknown species'])
    >>>
    >>> # Export results
    >>> resolver.export_results(results, 'resolved_names.csv')
    >>> resolver.export_results(results, 'resolved_names.xlsx')
    """

    SUPPORTED_SOURCES = ['wfo', 'powo', 'ipni', 'itis', 'gbif']
    DEFAULT_SOURCE = 'wfo'

# Copyright (c) 2025 Mohamed Z. Hatim
    API_ENDPOINTS = {
        'wfo': 'https://list.worldfloraonline.org/matching_rest.php',
        'powo': 'https://powo.science.kew.org/api/2/search',
        'ipni': 'https://www.ipni.org/api/1/search',
        'itis': 'https://www.itis.gov/ITISWebService/jsonservice',
        'gbif': 'https://api.gbif.org/v1/species/match'
    }

    def __init__(self,
                 sources: Optional[Union[str, List[str]]] = None,
                 use_fallback: bool = True,
                 cache_results: bool = True,
                 request_delay: float = 0.1,
                 timeout: int = 30):
        """Initialize the taxonomic resolver."""

# Copyright (c) 2025 Mohamed Z. Hatim
        if sources is None:
            self.sources = [self.DEFAULT_SOURCE]
            print(f"No source specified. Using default: {self.DEFAULT_SOURCE.upper()} (World Flora Online)")
        elif isinstance(sources, str):
            sources_lower = sources.lower()
            if sources_lower not in self.SUPPORTED_SOURCES:
                raise ValueError(f"Unsupported source: {sources}. Supported: {self.SUPPORTED_SOURCES}")
            self.sources = [sources_lower]
        else:
            self.sources = []
            for s in sources:
                s_lower = s.lower()
                if s_lower not in self.SUPPORTED_SOURCES:
                    raise ValueError(f"Unsupported source: {s}. Supported: {self.SUPPORTED_SOURCES}")
                self.sources.append(s_lower)

        self.use_fallback = use_fallback
        self.cache_results = cache_results
        self.request_delay = request_delay
        self.timeout = timeout

# Copyright (c) 2025 Mohamed Z. Hatim
        self._cache = {}

# Copyright (c) 2025 Mohamed Z. Hatim
        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': 'VegZ-Python-Package/1.3.0 (vegetation data analysis)',
            'Accept': 'application/json'
        })

    def resolve_names(self,
                     names: Union[List[str], pd.Series],
                     include_synonyms: bool = True,
                     verbose: bool = True) -> pd.DataFrame:
        """
        Resolve a list of plant species names against taxonomic databases.

        Parameters:
        -----------
        names : list or pd.Series
            List of species names to resolve
        include_synonyms : bool
            If True, include synonym information in results. Default is True.
        verbose : bool
            If True, print progress information. Default is True.

        Returns:
        --------
        pd.DataFrame
            DataFrame with columns:
            - original_name: The input species name
            - accepted_name: The resolved accepted name
            - accepted_author: Author of the accepted name
            - match_score: Confidence score (0-100)
            - match_type: Type of match (exact, fuzzy, synonym, etc.)
            - taxonomic_status: Status (accepted, synonym, unresolved, etc.)
            - synonyms: List of known synonyms (if include_synonyms=True)
            - family: Taxonomic family
            - source: Database source used for resolution
            - source_id: ID in the source database
            - source_url: URL to the record in source database
        """
        if isinstance(names, pd.Series):
            names = names.dropna().astype(str).tolist()

        results = []
        total = len(names)

        if verbose:
            print(f"Resolving {total} species names using: {', '.join([s.upper() for s in self.sources])}")
            if self.use_fallback and len(self.sources) > 1:
                print(f"Fallback enabled: will try sources in order if no match found")

        for i, name in enumerate(names):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{total} names processed...")

            result = self._resolve_single_name(name, include_synonyms)
            results.append(result)

# Copyright (c) 2025 Mohamed Z. Hatim
            time.sleep(self.request_delay)

        if verbose:
            print(f"Completed: {total} names processed")

# Copyright (c) 2025 Mohamed Z. Hatim
        df = pd.DataFrame(results)

# Copyright (c) 2025 Mohamed Z. Hatim
        column_order = [
            'original_name', 'accepted_name', 'accepted_author', 'match_score',
            'match_type', 'taxonomic_status', 'synonyms', 'family', 'genus',
            'source', 'source_id', 'source_url'
        ]
        existing_cols = [c for c in column_order if c in df.columns]
        other_cols = [c for c in df.columns if c not in column_order]
        df = df[existing_cols + other_cols]

        return df

    def _resolve_single_name(self, name: str, include_synonyms: bool = True) -> Dict[str, Any]:
        """Resolve a single species name."""

# Copyright (c) 2025 Mohamed Z. Hatim
        name_clean = self._clean_name(name)

# Copyright (c) 2025 Mohamed Z. Hatim
        cache_key = f"{name_clean}_{include_synonyms}_{'-'.join(self.sources)}"
        if self.cache_results and cache_key in self._cache:
            return self._cache[cache_key]

# Copyright (c) 2025 Mohamed Z. Hatim
        result = {
            'original_name': name,
            'accepted_name': None,
            'accepted_author': None,
            'match_score': 0,
            'match_type': 'no_match',
            'taxonomic_status': 'unresolved',
            'synonyms': [],
            'family': None,
            'genus': None,
            'source': None,
            'source_id': None,
            'source_url': None
        }

# Copyright (c) 2025 Mohamed Z. Hatim
        for source in self.sources:
            try:
                source_result = self._query_source(source, name_clean, include_synonyms)

                if source_result and source_result.get('match_score', 0) > 0:
                    result.update(source_result)
                    result['source'] = source.upper()
                    break

            except Exception as e:
                warnings.warn(f"Error querying {source.upper()} for '{name}': {str(e)}")

                if not self.use_fallback:
                    break
                continue

# Copyright (c) 2025 Mohamed Z. Hatim
        if self.cache_results:
            self._cache[cache_key] = result

        return result

    def _clean_name(self, name: str) -> str:
        """Clean and normalize a species name for querying."""
        if not name or not isinstance(name, str):
            return ''

# Copyright (c) 2025 Mohamed Z. Hatim
        name = name.strip()
        name = re.sub(r'\s+', ' ', name)

# Copyright (c) 2025 Mohamed Z. Hatim
        author_patterns = [
            r'\s+L\.$',
            r'\s+\([^)]+\)\s*[A-Z][a-z]+.*$',
            r'\s+[A-Z][a-z]+\s+\d{4}$',
            r'\s+\d{4}$',
        ]
        for pattern in author_patterns:
            name = re.sub(pattern, '', name)

        return name.strip()

    def _query_source(self, source: str, name: str, include_synonyms: bool) -> Optional[Dict[str, Any]]:
        """Query a specific taxonomic source."""

        if source == 'wfo':
            return self._query_wfo(name, include_synonyms)
        elif source == 'powo':
            return self._query_powo(name, include_synonyms)
        elif source == 'ipni':
            return self._query_ipni(name, include_synonyms)
        elif source == 'itis':
            return self._query_itis(name, include_synonyms)
        elif source == 'gbif':
            return self._query_gbif(name, include_synonyms)
        else:
            return None

# Copyright (c) 2025 Mohamed Z. Hatim
    def _query_wfo(self, name: str, include_synonyms: bool) -> Optional[Dict[str, Any]]:
        """Query World Flora Online API."""

        try:
            params = {
                'input_string': name,
                'match_mode': 'normal'
            }

            response = self._session.get(
                self.API_ENDPOINTS['wfo'],
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()

            if data.get('error'):
                return None

# Copyright (c) 2025 Mohamed Z. Hatim
            match_data = data.get('match')
            candidates = data.get('candidates', [])

            if not match_data and not candidates:
                return None

# Copyright (c) 2025 Mohamed Z. Hatim
            if match_data:
                record = match_data
                match_type = 'exact'
                match_score = 100
            elif candidates:
                record = candidates[0]
                match_type = 'candidate'
                match_score = 85 if len(candidates) == 1 else 75

# Copyright (c) 2025 Mohamed Z. Hatim
            full_name = record.get('full_name_plain', '')
            name_parts = full_name.rsplit(' ', 1) if ' ' in full_name else [full_name, '']
            accepted_name = name_parts[0] if len(name_parts) > 1 and name_parts[-1] and name_parts[-1][0].isupper() else full_name
            author = name_parts[-1] if len(name_parts) > 1 and name_parts[-1] and name_parts[-1][0].isupper() else None

# Copyright (c) 2025 Mohamed Z. Hatim
            placement = record.get('placement', '')
            family = None
            if placement:
                for part in placement.split('/'):
                    if part.endswith('aceae'):
                        family = part
                        break

# Copyright (c) 2025 Mohamed Z. Hatim
            genus = None
            if accepted_name and ' ' in accepted_name:
                genus = accepted_name.split()[0]

            wfo_id = record.get('wfo_id')

            result = {
                'accepted_name': accepted_name,
                'accepted_author': author,
                'match_score': match_score,
                'match_type': match_type,
                'taxonomic_status': 'accepted' if match_data else 'candidate',
                'family': family,
                'genus': genus or (accepted_name.split()[0] if accepted_name and ' ' in accepted_name else None),
                'source_id': wfo_id,
                'source_url': f"https://wfoplantlist.org/plant-list/taxon/{wfo_id}" if wfo_id else None
            }

# Copyright (c) 2025 Mohamed Z. Hatim
            result['synonyms'] = []

            return result

        except requests.RequestException as e:
            warnings.warn(f"WFO API request failed: {str(e)}")
            return None

# Copyright (c) 2025 Mohamed Z. Hatim
    def _query_powo(self, name: str, include_synonyms: bool) -> Optional[Dict[str, Any]]:
        """Query Plants of the World Online (Kew) API."""

        try:
            params = {
                'q': name,
                'f': 'accepted_names,synonyms',
                'page.size': 1
            }

            response = self._session.get(
                self.API_ENDPOINTS['powo'],
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()

            if not data.get('results'):
                return None

            record = data['results'][0]

            accepted_name = record.get('name')
            genus = record.get('genus')
            if not genus and accepted_name and ' ' in accepted_name:
                genus = accepted_name.split()[0]

            result = {
                'accepted_name': accepted_name,
                'accepted_author': record.get('authors'),
                'match_score': self._calculate_powo_score(record, name),
                'match_type': 'exact' if record.get('name', '').lower() == name.lower() else 'fuzzy',
                'taxonomic_status': 'accepted' if record.get('accepted') else 'synonym',
                'family': record.get('family'),
                'genus': genus,
                'source_id': record.get('fqId'),
                'source_url': f"https://powo.science.kew.org/taxon/{record.get('fqId')}" if record.get('fqId') else None
            }

# Copyright (c) 2025 Mohamed Z. Hatim
            if not record.get('accepted') and record.get('synonymOf'):
                result['accepted_name'] = record['synonymOf'].get('name')
                result['accepted_author'] = record['synonymOf'].get('authors')
                result['source_id'] = record['synonymOf'].get('fqId')
                result['source_url'] = f"https://powo.science.kew.org/taxon/{record['synonymOf'].get('fqId')}" if record['synonymOf'].get('fqId') else None

# Copyright (c) 2025 Mohamed Z. Hatim
            if include_synonyms and record.get('fqId'):
                result['synonyms'] = self._fetch_powo_synonyms(record.get('fqId'))

            return result

        except requests.RequestException as e:
            warnings.warn(f"POWO API request failed: {str(e)}")
            return None

    def _fetch_powo_synonyms(self, fq_id: str) -> List[str]:
        """Fetch synonyms for a POWO record."""
        try:
            response = self._session.get(
                f"https://powo.science.kew.org/api/2/taxon/{fq_id}",
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            synonyms = []
            for syn in data.get('synonyms', []):
                synonyms.append(syn.get('name', ''))

            return synonyms[:20]

        except Exception:
            return []

    def _calculate_powo_score(self, record: Dict, query: str) -> int:
        """Calculate match confidence score for POWO results."""
        name = record.get('name', '').lower()
        query = query.lower()

        if name == query:
            return 100
        elif query in name or name in query:
            return 85
        else:
            return 70

# Copyright (c) 2025 Mohamed Z. Hatim
    def _query_ipni(self, name: str, include_synonyms: bool) -> Optional[Dict[str, Any]]:
        """Query International Plant Names Index API."""

        try:
            params = {
                'q': name,
                'f': 'f_species',
                'page.size': 1
            }

            response = self._session.get(
                self.API_ENDPOINTS['ipni'],
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()

            if not data.get('results'):
                return None

            record = data['results'][0]

            result = {
                'accepted_name': record.get('name'),
                'accepted_author': record.get('authors'),
                'match_score': self._calculate_ipni_score(record, name),
                'match_type': 'exact' if record.get('name', '').lower() == name.lower() else 'fuzzy',
                'taxonomic_status': 'published',
                'family': record.get('family'),
                'genus': record.get('genus'),
                'source_id': record.get('id'),
                'source_url': f"https://www.ipni.org/n/{record.get('id')}" if record.get('id') else None,
                'publication': record.get('reference'),
                'publication_year': record.get('publicationYear')
            }

# Copyright (c) 2025 Mohamed Z. Hatim
            result['synonyms'] = []

            return result

        except requests.RequestException as e:
            warnings.warn(f"IPNI API request failed: {str(e)}")
            return None

    def _calculate_ipni_score(self, record: Dict, query: str) -> int:
        """Calculate match confidence score for IPNI results."""
        name = record.get('name', '').lower()
        query = query.lower()

        if name == query:
            return 100
        elif query in name or name in query:
            return 80
        else:
            return 65

# Copyright (c) 2025 Mohamed Z. Hatim
    def _query_itis(self, name: str, include_synonyms: bool) -> Optional[Dict[str, Any]]:
        """Query Integrated Taxonomic Information System API."""

        try:
# Copyright (c) 2025 Mohamed Z. Hatim
            search_url = f"{self.API_ENDPOINTS['itis']}/searchByScientificName"
            params = {'srchKey': name}

            response = self._session.get(search_url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()

            if not data.get('scientificNames'):
                return None

            names_list = data['scientificNames']
            if not names_list:
                return None

# Copyright (c) 2025 Mohamed Z. Hatim
            best_match = None
            for item in names_list:
                if item.get('combinedName', '').lower() == name.lower():
                    best_match = item
                    break

            if not best_match:
                best_match = names_list[0]

            tsn = best_match.get('tsn')

# Copyright (c) 2025 Mohamed Z. Hatim
            full_record = self._fetch_itis_full_record(tsn) if tsn else {}

            result = {
                'accepted_name': full_record.get('accepted_name') or best_match.get('combinedName'),
                'accepted_author': full_record.get('author'),
                'match_score': 100 if best_match.get('combinedName', '').lower() == name.lower() else 75,
                'match_type': 'exact' if best_match.get('combinedName', '').lower() == name.lower() else 'fuzzy',
                'taxonomic_status': full_record.get('usage', 'unknown'),
                'family': full_record.get('family'),
                'genus': full_record.get('genus'),
                'source_id': str(tsn) if tsn else None,
                'source_url': f"https://www.itis.gov/servlet/SingleRpt/SingleRpt?search_topic=TSN&search_value={tsn}" if tsn else None
            }

# Copyright (c) 2025 Mohamed Z. Hatim
            if include_synonyms and tsn:
                result['synonyms'] = self._fetch_itis_synonyms(tsn)
            else:
                result['synonyms'] = []

            return result

        except requests.RequestException as e:
            warnings.warn(f"ITIS API request failed: {str(e)}")
            return None

    def _fetch_itis_full_record(self, tsn: int) -> Dict[str, Any]:
        """Fetch full record from ITIS."""
        try:
            record = {}

# Copyright (c) 2025 Mohamed Z. Hatim
            url = f"{self.API_ENDPOINTS['itis']}/getFullRecordFromTSN"
            response = self._session.get(url, params={'tsn': tsn}, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            if data.get('scientificName'):
                record['accepted_name'] = data['scientificName'].get('combinedName')

            if data.get('taxonAuthor'):
                record['author'] = data['taxonAuthor'].get('authorship')

            if data.get('usage'):
                record['usage'] = data['usage'].get('taxonUsageRating', 'unknown')

# Copyright (c) 2025 Mohamed Z. Hatim
            if data.get('hierarchyUp'):
                for taxon in data.get('hierarchyUp', []):
                    rank = taxon.get('rankName', '').lower()
                    if rank == 'family':
                        record['family'] = taxon.get('taxonName')
                    elif rank == 'genus':
                        record['genus'] = taxon.get('taxonName')

            return record

        except Exception:
            return {}

    def _fetch_itis_synonyms(self, tsn: int) -> List[str]:
        """Fetch synonyms from ITIS."""
        try:
            url = f"{self.API_ENDPOINTS['itis']}/getSynonymNameFromTSN"
            response = self._session.get(url, params={'tsn': tsn}, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            synonyms = []
            for syn in data.get('synonyms', []):
                if syn.get('sciName'):
                    synonyms.append(syn['sciName'])

            return synonyms[:20]

        except Exception:
            return []

# Copyright (c) 2025 Mohamed Z. Hatim
    def _query_gbif(self, name: str, include_synonyms: bool) -> Optional[Dict[str, Any]]:
        """Query GBIF Species API (Catalogue of Life backbone)."""

        try:
            params = {
                'name': name,
                'kingdom': 'Plantae',
                'strict': False
            }

            response = self._session.get(
                self.API_ENDPOINTS['gbif'],
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()

            if data.get('matchType') == 'NONE':
                return None

            result = {
                'accepted_name': data.get('species') or data.get('canonicalName'),
                'accepted_author': None,
                'match_score': data.get('confidence', 0),
                'match_type': data.get('matchType', 'unknown').lower(),
                'taxonomic_status': data.get('status', 'unknown').lower(),
                'family': data.get('family'),
                'genus': data.get('genus'),
                'source_id': str(data.get('usageKey')) if data.get('usageKey') else None,
                'source_url': f"https://www.gbif.org/species/{data.get('usageKey')}" if data.get('usageKey') else None,
                'kingdom': data.get('kingdom'),
                'phylum': data.get('phylum'),
                'class': data.get('class'),
                'order': data.get('order')
            }

# Copyright (c) 2025 Mohamed Z. Hatim
            if data.get('status') == 'SYNONYM' and data.get('acceptedUsageKey'):
                accepted = self._fetch_gbif_accepted(data['acceptedUsageKey'])
                if accepted:
                    result['accepted_name'] = accepted.get('canonicalName') or accepted.get('species')
                    result['source_url'] = f"https://www.gbif.org/species/{data['acceptedUsageKey']}"

# Copyright (c) 2025 Mohamed Z. Hatim
            if include_synonyms and data.get('usageKey'):
                result['synonyms'] = self._fetch_gbif_synonyms(data.get('usageKey'))
            else:
                result['synonyms'] = []

            return result

        except requests.RequestException as e:
            warnings.warn(f"GBIF API request failed: {str(e)}")
            return None

    def _fetch_gbif_accepted(self, usage_key: int) -> Optional[Dict]:
        """Fetch accepted name from GBIF."""
        try:
            response = self._session.get(
                f"https://api.gbif.org/v1/species/{usage_key}",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

    def _fetch_gbif_synonyms(self, usage_key: int) -> List[str]:
        """Fetch synonyms from GBIF."""
        try:
            response = self._session.get(
                f"https://api.gbif.org/v1/species/{usage_key}/synonyms",
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            synonyms = []
            for result in data.get('results', [])[:20]:
                name = result.get('canonicalName') or result.get('species')
                if name:
                    synonyms.append(name)

            return synonyms

        except Exception:
            return []

    def resolve_from_file(self,
                         filepath: str,
                         species_column: Optional[str] = None,
                         sheet_name: Optional[str] = None,
                         include_synonyms: bool = True,
                         verbose: bool = True) -> pd.DataFrame:
        """
        Resolve species names directly from a file.

        Parameters:
        -----------
        filepath : str
            Path to input file. Supported formats: .csv, .xlsx, .xls, .tsv, .txt
        species_column : str, optional
            Name of the column containing species names. If None, will auto-detect
            columns named 'species', 'scientific_name', 'taxon', 'species_name', etc.
        sheet_name : str, optional
            For Excel files, the sheet name to read. Default is first sheet.
        include_synonyms : bool
            If True, include synonym information. Default is True.
        verbose : bool
            If True, print progress information. Default is True.

        Returns:
        --------
        pd.DataFrame
            Resolution results with all taxonomic information

        Example:
        --------
        >>> resolver = TaxonomicResolver(sources='gbif')
        >>> results = resolver.resolve_from_file('my_species_list.csv')
        >>> results = resolver.resolve_from_file('data.xlsx', species_column='ScientificName')
        """
        filepath = str(filepath)

        if verbose:
            print(f"Reading species names from: {filepath}")

# Copyright (c) 2025 Mohamed Z. Hatim
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filepath, sheet_name=sheet_name or 0)
        elif filepath.endswith('.tsv') or filepath.endswith('.txt'):
            df = pd.read_csv(filepath, sep='\t')
        else:
            raise ValueError(f"Unsupported file format. Supported: .csv, .xlsx, .xls, .tsv, .txt")

# Copyright (c) 2025 Mohamed Z. Hatim
        if species_column is None:
            species_column = self._auto_detect_species_column(df)
            if species_column is None:
                raise ValueError(
                    "Could not auto-detect species column. Please specify 'species_column' parameter. "
                    f"Available columns: {list(df.columns)}"
                )
            if verbose:
                print(f"Auto-detected species column: '{species_column}'")

        if species_column not in df.columns:
            raise ValueError(f"Column '{species_column}' not found. Available: {list(df.columns)}")

# Copyright (c) 2025 Mohamed Z. Hatim
        names = df[species_column].dropna().astype(str).unique().tolist()

        if verbose:
            print(f"Found {len(names)} unique species names to resolve")

        return self.resolve_names(names, include_synonyms=include_synonyms, verbose=verbose)

    def resolve_dataframe(self,
                         df: pd.DataFrame,
                         species_column: Optional[str] = None,
                         update_names: bool = True,
                         add_taxonomy_columns: bool = True,
                         min_score_threshold: int = 70,
                         include_synonyms: bool = False,
                         verbose: bool = True) -> pd.DataFrame:
        """
        Resolve and optionally update species names in an existing DataFrame.

        This method is designed for integration with data analysis workflows,
        allowing you to validate and standardize species names in your vegetation data.

        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame containing species data
        species_column : str, optional
            Name of the column containing species names. Auto-detects if None.
        update_names : bool
            If True, update the species column with accepted names. Default is True.
        add_taxonomy_columns : bool
            If True, add columns for family, genus, match_score, etc. Default is True.
        min_score_threshold : int
            Minimum match score to update a name (0-100). Default is 70.
        include_synonyms : bool
            If True, add synonyms column. Default is False (to keep DataFrame clean).
        verbose : bool
            If True, print progress. Default is True.

        Returns:
        --------
        pd.DataFrame
            DataFrame with resolved/updated species names and optional taxonomy columns

        Example:
        --------
        >>> from VegZ import VegZ, TaxonomicResolver
        >>>
        >>> # Load your vegetation data
        >>> veg = VegZ()
        >>> veg.load_data('vegetation_survey.csv')
        >>>
        >>> # Resolve species names in the data
        >>> resolver = TaxonomicResolver(sources=['gbif', 'wfo'])
        >>> updated_data = resolver.resolve_dataframe(veg.species_matrix)
        >>>
        >>> # Or resolve names during data preparation
        >>> df = pd.read_csv('raw_data.csv')
        >>> df_resolved = resolver.resolve_dataframe(df, species_column='species')
        """
        df_result = df.copy()

# Copyright (c) 2025 Mohamed Z. Hatim
        if species_column is None:
            species_column = self._auto_detect_species_column(df_result)
            if species_column is None:
                raise ValueError(
                    "Could not auto-detect species column. Please specify 'species_column' parameter. "
                    f"Available columns: {list(df_result.columns)}"
                )
            if verbose:
                print(f"Auto-detected species column: '{species_column}'")

        if species_column not in df_result.columns:
            raise ValueError(f"Column '{species_column}' not found. Available: {list(df_result.columns)}")

# Copyright (c) 2025 Mohamed Z. Hatim
        unique_names = df_result[species_column].dropna().astype(str).unique().tolist()

        if verbose:
            print(f"Resolving {len(unique_names)} unique species names...")

# Copyright (c) 2025 Mohamed Z. Hatim
        resolution_results = self.resolve_names(unique_names, include_synonyms=include_synonyms, verbose=verbose)

# Copyright (c) 2025 Mohamed Z. Hatim
        resolution_map = {}
        for _, row in resolution_results.iterrows():
            resolution_map[row['original_name']] = row.to_dict()

# Copyright (c) 2025 Mohamed Z. Hatim
        if update_names:
            df_result[f'{species_column}_original'] = df_result[species_column]

            def get_updated_name(name):
                if pd.isna(name):
                    return name
                name_str = str(name)
                if name_str in resolution_map:
                    result = resolution_map[name_str]
                    if result['match_score'] >= min_score_threshold and result['accepted_name']:
                        return result['accepted_name']
                return name_str

            df_result[species_column] = df_result[species_column].apply(get_updated_name)

# Copyright (c) 2025 Mohamed Z. Hatim
        if add_taxonomy_columns:
            def get_resolution_value(name, key):
                if pd.isna(name):
                    return None
                name_str = str(name)
                if name_str in resolution_map:
                    return resolution_map[name_str].get(key)
                return None

            original_col = f'{species_column}_original' if update_names else species_column

            df_result['taxon_family'] = df_result[original_col].apply(lambda x: get_resolution_value(x, 'family'))
            df_result['taxon_genus'] = df_result[original_col].apply(lambda x: get_resolution_value(x, 'genus'))
            df_result['taxon_match_score'] = df_result[original_col].apply(lambda x: get_resolution_value(x, 'match_score'))
            df_result['taxon_status'] = df_result[original_col].apply(lambda x: get_resolution_value(x, 'taxonomic_status'))
            df_result['taxon_source'] = df_result[original_col].apply(lambda x: get_resolution_value(x, 'source'))

            if include_synonyms:
                df_result['taxon_synonyms'] = df_result[original_col].apply(
                    lambda x: '; '.join(get_resolution_value(x, 'synonyms') or [])
                )

        if verbose:
            resolved = resolution_results[resolution_results['match_score'] >= min_score_threshold]
            print(f"\nResolution complete:")
            print(f"  - {len(resolved)}/{len(unique_names)} names resolved (>={min_score_threshold}% confidence)")
            if update_names:
                print(f"  - Original names preserved in '{species_column}_original'")
            if add_taxonomy_columns:
                print(f"  - Added taxonomy columns: taxon_family, taxon_genus, taxon_match_score, taxon_status, taxon_source")

        return df_result

    def _auto_detect_species_column(self, df: pd.DataFrame) -> Optional[str]:
        """Auto-detect the species name column in a DataFrame."""
        potential_names = [
            'species', 'species_name', 'scientific_name', 'scientificname',
            'taxon', 'taxa', 'binomial', 'name', 'plant_name', 'plantname',
            'species_col', 'sp', 'organism', 'accepted_name', 'canonical_name'
        ]

        columns_lower = {col.lower(): col for col in df.columns}

        for potential in potential_names:
            if potential in columns_lower:
                return columns_lower[potential]

# Copyright (c) 2025 Mohamed Z. Hatim
        for col in df.columns:
            if 'species' in col.lower() or 'taxon' in col.lower() or 'name' in col.lower():
                return col

        return None

    def export_results(self,
                      results: pd.DataFrame,
                      filepath: str,
                      **kwargs) -> None:
        """
        Export resolution results to a file.

        Parameters:
        -----------
        results : pd.DataFrame
            Results DataFrame from resolve_names()
        filepath : str
            Output file path. Format determined by extension.
            Supported: .csv, .xlsx, .json, .tsv, .parquet, .html
        **kwargs
            Additional arguments passed to the pandas export function
        """
        filepath = str(filepath)

# Copyright (c) 2025 Mohamed Z. Hatim
        export_df = results.copy()
        if 'synonyms' in export_df.columns:
            export_df['synonyms'] = export_df['synonyms'].apply(
                lambda x: '; '.join(x) if isinstance(x, list) else str(x) if x else ''
            )

        if filepath.endswith('.csv'):
            export_df.to_csv(filepath, index=False, **kwargs)
        elif filepath.endswith('.xlsx'):
            export_df.to_excel(filepath, index=False, **kwargs)
        elif filepath.endswith('.json'):
            results.to_json(filepath, orient='records', indent=2, **kwargs)
        elif filepath.endswith('.tsv'):
            export_df.to_csv(filepath, index=False, sep='\t', **kwargs)
        elif filepath.endswith('.parquet'):
            results.to_parquet(filepath, index=False, **kwargs)
        elif filepath.endswith('.html'):
            export_df.to_html(filepath, index=False, **kwargs)
        else:
            raise ValueError(f"Unsupported file format. Supported: .csv, .xlsx, .json, .tsv, .parquet, .html")

        print(f"Results exported to: {filepath}")

    def get_summary(self, results: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a summary of resolution results.

        Parameters:
        -----------
        results : pd.DataFrame
            Results DataFrame from resolve_names()

        Returns:
        --------
        dict
            Summary statistics of the resolution
        """
        total = len(results)

        resolved = results[results['match_score'] > 0]
        unresolved = results[results['match_score'] == 0]

        high_confidence = results[results['match_score'] >= 90]
        medium_confidence = results[(results['match_score'] >= 70) & (results['match_score'] < 90)]
        low_confidence = results[(results['match_score'] > 0) & (results['match_score'] < 70)]

        summary = {
            'total_names': total,
            'resolved': len(resolved),
            'unresolved': len(unresolved),
            'resolution_rate': round(len(resolved) / total * 100, 2) if total > 0 else 0,
            'high_confidence_matches': len(high_confidence),
            'medium_confidence_matches': len(medium_confidence),
            'low_confidence_matches': len(low_confidence),
            'average_match_score': round(results['match_score'].mean(), 2),
            'sources_used': results['source'].dropna().unique().tolist(),
            'match_types': results['match_type'].value_counts().to_dict(),
            'taxonomic_statuses': results['taxonomic_status'].value_counts().to_dict(),
            'families_found': results['family'].dropna().nunique()
        }

        return summary

    def print_summary(self, results: pd.DataFrame) -> None:
        """Print a formatted summary of resolution results."""
        summary = self.get_summary(results)

        print("\n" + "=" * 60)
        print("TAXONOMIC RESOLUTION SUMMARY")
        print("=" * 60)
        print(f"Total names processed:     {summary['total_names']}")
        print(f"Successfully resolved:     {summary['resolved']} ({summary['resolution_rate']}%)")
        print(f"Unresolved:                {summary['unresolved']}")
        print("-" * 60)
        print(f"High confidence (>=90):    {summary['high_confidence_matches']}")
        print(f"Medium confidence (70-89): {summary['medium_confidence_matches']}")
        print(f"Low confidence (<70):      {summary['low_confidence_matches']}")
        print(f"Average match score:       {summary['average_match_score']}")
        print("-" * 60)
        print(f"Sources used:              {', '.join(summary['sources_used']) if summary['sources_used'] else 'None'}")
        print(f"Unique families found:     {summary['families_found']}")
        print("-" * 60)
        print("Match types:")
        for match_type, count in summary['match_types'].items():
            print(f"  - {match_type}: {count}")
        print("-" * 60)
        print("Taxonomic statuses:")
        for status, count in summary['taxonomic_statuses'].items():
            print(f"  - {status}: {count}")
        print("=" * 60 + "\n")

    def clear_cache(self) -> None:
        """Clear the results cache."""
        self._cache = {}
        print("Cache cleared")

    def __repr__(self) -> str:
        return f"TaxonomicResolver(sources={self.sources}, use_fallback={self.use_fallback})"


def resolve_species_names(names: Union[List[str], pd.Series],
                         sources: Optional[Union[str, List[str]]] = None,
                         use_fallback: bool = True,
                         include_synonyms: bool = True,
                         verbose: bool = True) -> pd.DataFrame:
    """
    Convenience function to resolve plant species names.

    Parameters:
    -----------
    names : list or pd.Series
        List of species names to resolve
    sources : str or list, optional
        Source(s) to use. Default is 'wfo' (World Flora Online).
        Options: 'wfo', 'powo', 'ipni', 'itis', 'gbif'
    use_fallback : bool
        If True, try alternative sources if first fails. Default is True.
    include_synonyms : bool
        If True, include synonym information. Default is True.
    verbose : bool
        If True, print progress. Default is True.

    Returns:
    --------
    pd.DataFrame
        Resolution results with original names, accepted names, scores, and metadata

    Example:
    --------
    >>> from VegZ.data_management import resolve_species_names
    >>>
    >>> names = ['Quercus robur', 'Pinus sylvestris', 'Unknown plant']
    >>> results = resolve_species_names(names)
    >>> results.to_csv('resolved.csv')
    """
    resolver = TaxonomicResolver(sources=sources, use_fallback=use_fallback)
    return resolver.resolve_names(names, include_synonyms=include_synonyms, verbose=verbose)


def resolve_species_from_file(filepath: str,
                              species_column: Optional[str] = None,
                              sources: Optional[Union[str, List[str]]] = None,
                              use_fallback: bool = True,
                              output_file: Optional[str] = None,
                              verbose: bool = True) -> pd.DataFrame:
    """
    Convenience function to resolve species names from a file.

    Parameters:
    -----------
    filepath : str
        Path to input file (.csv, .xlsx, .tsv, .txt)
    species_column : str, optional
        Column containing species names. Auto-detects if None.
    sources : str or list, optional
        Source(s) to use. Default is 'wfo' (World Flora Online).
    use_fallback : bool
        If True, try alternative sources if first fails. Default is True.
    output_file : str, optional
        If provided, export results to this file.
    verbose : bool
        If True, print progress. Default is True.

    Returns:
    --------
    pd.DataFrame
        Resolution results

    Example:
    --------
    >>> from VegZ.data_management import resolve_species_from_file
    >>>
    >>> # Resolve from CSV and export to Excel
    >>> results = resolve_species_from_file(
    ...     'my_species.csv',
    ...     output_file='resolved_species.xlsx'
    ... )
    """
    resolver = TaxonomicResolver(sources=sources, use_fallback=use_fallback)
    results = resolver.resolve_from_file(filepath, species_column=species_column, verbose=verbose)

    if output_file:
        resolver.export_results(results, output_file)

    return results


def update_species_in_dataframe(df: pd.DataFrame,
                                species_column: Optional[str] = None,
                                sources: Optional[Union[str, List[str]]] = None,
                                use_fallback: bool = True,
                                min_score: int = 70,
                                verbose: bool = True) -> pd.DataFrame:
    """
    Convenience function to resolve and update species names in a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with species data
    species_column : str, optional
        Column containing species names. Auto-detects if None.
    sources : str or list, optional
        Source(s) to use. Default is 'wfo' (World Flora Online).
    use_fallback : bool
        If True, try alternative sources if first fails. Default is True.
    min_score : int
        Minimum match score to update names (0-100). Default is 70.
    verbose : bool
        If True, print progress. Default is True.

    Returns:
    --------
    pd.DataFrame
        DataFrame with updated species names and taxonomy columns

    Example:
    --------
    >>> import pandas as pd
    >>> from VegZ.data_management import update_species_in_dataframe
    >>>
    >>> # Load and update species names
    >>> df = pd.read_csv('vegetation_data.csv')
    >>> df_updated = update_species_in_dataframe(df, sources='gbif')
    >>> df_updated.to_csv('vegetation_data_resolved.csv')
    """
    resolver = TaxonomicResolver(sources=sources, use_fallback=use_fallback)
    return resolver.resolve_dataframe(
        df,
        species_column=species_column,
        update_names=True,
        add_taxonomy_columns=True,
        min_score_threshold=min_score,
        verbose=verbose
    )
