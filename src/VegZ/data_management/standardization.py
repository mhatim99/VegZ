"""
Data standardization and integration utilities.

Copyright (c) 2025 Mohamed Z. Hatim
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import warnings

# Copyright (c) 2025 Mohamed Z. Hatim
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*Using slow pure-python SequenceMatcher.*")
    from fuzzywuzzy import fuzz, process
import re


class DataStandardizer:
    """Main class for standardizing and integrating heterogeneous datasets."""
    
    def __init__(self):
        self.standard_columns = {
            'spatial': ['latitude', 'longitude', 'coordinate_precision', 'coordinate_system'],
            'taxonomic': ['species', 'genus', 'family', 'order', 'class', 'kingdom'],
            'temporal': ['date', 'year', 'month', 'day'],
            'ecological': ['abundance', 'cover', 'frequency', 'biomass'],
            'environmental': ['temperature', 'precipitation', 'elevation', 'slope', 'aspect'],
            'identification': ['plot_id', 'site_id', 'sample_id', 'observer']
        }
        
        self.species_standardizer = SpeciesNameStandardizer()
        self.coordinate_standardizer = CoordinateStandardizer()
        
    def standardize_dataset(self, 
                           df: pd.DataFrame,
                           dataset_type: str = 'vegetation',
                           column_mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Standardize a dataset to common format.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset
        dataset_type : str
            Type of dataset ('vegetation', 'environmental', 'spatial')
        column_mapping : dict, optional
            Custom column name mappings
            
        Returns:
        --------
        pd.DataFrame
            Standardized dataset
        """
        df_std = df.copy()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if column_mapping:
            df_std = df_std.rename(columns=column_mapping)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        df_std = self._auto_standardize_columns(df_std)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if 'species' in df_std.columns:
            df_std = self.species_standardizer.standardize_dataframe(df_std)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if 'latitude' in df_std.columns and 'longitude' in df_std.columns:
            df_std = self.coordinate_standardizer.standardize_coordinates(df_std)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if 'date' in df_std.columns:
            df_std = self._standardize_dates(df_std)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        df_std = self._handle_missing_values(df_std)
        
        return df_std
    
    def integrate_datasets(self, 
                          datasets: List[pd.DataFrame],
                          join_columns: List[str],
                          dataset_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Integrate multiple standardized datasets.
        
        Parameters:
        -----------
        datasets : list of pd.DataFrame
            List of standardized datasets
        join_columns : list
            Columns to use for joining datasets
        dataset_names : list, optional
            Names for source datasets
            
        Returns:
        --------
        pd.DataFrame
            Integrated dataset
        """
        if not datasets:
            raise ValueError("No datasets provided")
        
        if dataset_names is None:
            dataset_names = [f"dataset_{i}" for i in range(len(datasets))]
        
# Copyright (c) 2025 Mohamed Z. Hatim
        for i, df in enumerate(datasets):
            df['source_dataset'] = dataset_names[i]
        
# Copyright (c) 2025 Mohamed Z. Hatim
        result = datasets[0].copy()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        for i, df in enumerate(datasets[1:], 1):
# Copyright (c) 2025 Mohamed Z. Hatim
            common_cols = [col for col in join_columns if col in result.columns and col in df.columns]
            
            if not common_cols:
                warnings.warn(f"No common columns found for dataset {i}. Concatenating instead.")
                result = pd.concat([result, df], ignore_index=True, sort=False)
            else:
                result = pd.merge(result, df, on=common_cols, how='outer', suffixes=('', f'_{i}'))
        
        return result
    
    def _auto_standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Auto-detect and standardize column names."""
        column_mappings = {
# Copyright (c) 2025 Mohamed Z. Hatim
            'lat': 'latitude',
            'y': 'latitude',
            'coord_y': 'latitude',
            'lat_dd': 'latitude',
            'decimal_latitude': 'latitude',
            
            'lon': 'longitude',
            'lng': 'longitude',
            'long': 'longitude',
            'x': 'longitude',
            'coord_x': 'longitude',
            'lon_dd': 'longitude',
            'decimal_longitude': 'longitude',
            
# Copyright (c) 2025 Mohamed Z. Hatim
            'species_name': 'species',
            'scientific_name': 'species',
            'taxon': 'species',
            'taxa': 'species',
            'binomial': 'species',
            
            'genus_species': 'species',
            'tax_genus': 'genus',
            'tax_family': 'family',
            
# Copyright (c) 2025 Mohamed Z. Hatim
            'cover': 'abundance',
            'coverage': 'abundance',
            'percent_cover': 'abundance',
            'pct_cover': 'abundance',
            'abundance_value': 'abundance',
            
# Copyright (c) 2025 Mohamed Z. Hatim
            'plot': 'plot_id',
            'site': 'site_id',
            'releve': 'plot_id',
            'quadrat': 'plot_id',
            'sample': 'sample_id',
            
# Copyright (c) 2025 Mohamed Z. Hatim
            'sampling_date': 'date',
            'survey_date': 'date',
            'collection_date': 'date',
            'obs_date': 'date',
            'eventdate': 'date'
        }
        
# Copyright (c) 2025 Mohamed Z. Hatim
        for old_name, new_name in column_mappings.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        return df
    
    def _standardize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize date columns."""
        if 'date' in df.columns:
# Copyright (c) 2025 Mohamed Z. Hatim
            df['date'] = pd.to_datetime(df['date'], errors='coerce', infer_datetime_format=True)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['day_of_year'] = df['date'].dt.dayofyear
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to column type."""
        for column in df.columns:
            if column in self.standard_columns['spatial']:
# Copyright (c) 2025 Mohamed Z. Hatim
                continue
            elif column in self.standard_columns['taxonomic']:
# Copyright (c) 2025 Mohamed Z. Hatim
                continue
            elif column in self.standard_columns['ecological']:
# Copyright (c) 2025 Mohamed Z. Hatim
                if df[column].dtype in ['int64', 'float64']:
                    df[column] = df[column].fillna(0)
            elif column in self.standard_columns['environmental']:
# Copyright (c) 2025 Mohamed Z. Hatim
                if df[column].dtype in ['int64', 'float64']:
                    df[column] = df[column].fillna(df[column].median())
        
        return df


class SpeciesNameStandardizer:
    """Standardize and clean species names with comprehensive error detection and classification."""

    def __init__(self):
# Copyright (c) 2025 Mohamed Z. Hatim
        self.author_patterns = [
            r'\s+L\.(?:\s|$)',  # Linnaeus
            r'\s+\([^)]+\)\s*[A-Z][a-z]+',  # (Author) SecondAuthor
            r'\s+[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+\d{4})?$',  # Two or more author names with optional year
            r'\s+[A-Z][a-z]+\s+\d{4}(?:\s|$)',  # Author with year
            r'\s+\d{4}(?:\s|$)',  # Years only
            r'\s+ex\s+[A-Z][a-z]+',  # ex Author
            r'\s+sensu\s+[A-Z][a-z]+',  # sensu Author
            r'\s+in\s+[A-Z][a-z]+',  # in Author
            r'\s+emend\.?\s+[A-Z][a-z]+',  # emend. Author
            r'\s+nom\.\s+nud\.',  # nom. nud.
            r'\s+nom\.\s+inval\.',  # nom. inval.
            r'\s+et\s+al\.',  # et al.
            r'\s+&\s+[A-Z][a-z]+',  # & Author
        ]

# Copyright (c) 2025 Mohamed Z. Hatim
        self.infraspecific_markers = {
            'var.': 'variety',
            'variety': 'variety',
            'subsp.': 'subspecies',
            'ssp.': 'subspecies',
            'subspecies': 'subspecies',
            'f.': 'form',
            'forma': 'form',
            'form': 'form',
            'cv.': 'cultivar',
            'cultivar': 'cultivar',
            'subvar.': 'subvariety',
            'subf.': 'subform'
        }

# Copyright (c) 2025 Mohamed Z. Hatim
        self.placeholder_patterns = {
            r'\bsp\.?(?:\s|$)': 'species_placeholder',
            r'\bcf\.?(?:\s|$)': 'confer_placeholder',
            r'\baff\.?(?:\s|$)': 'affinis_placeholder',
            r'\bindet\.?(?:\s|$)': 'indeterminate_placeholder',
            r'\bunknown(?:\s|$)': 'unknown_placeholder',
            r'\bunidentified(?:\s|$)': 'unidentified_placeholder',
            r'\bundet\.?(?:\s|$)': 'undetermined_placeholder',
            r'\bspec\.?(?:\s|$)': 'species_placeholder',
            r'\bspecies(?:\s|$)': 'species_placeholder',
            r'\bgen\.?(?:\s|$)': 'genus_placeholder',
            r'\bgenus(?:\s|$)': 'genus_placeholder'
        }

# Copyright (c) 2025 Mohamed Z. Hatim
        self.hybrid_patterns = {
            '×': 'multiplication_hybrid',
            'x ': 'letter_x_hybrid',
            ' x ': 'letter_x_hybrid',
            'hybrid': 'text_hybrid',
            'hybr.': 'abbreviated_hybrid'
        }

# Copyright (c) 2025 Mohamed Z. Hatim
        self.invalid_patterns = {
            r'\d+': 'contains_numbers',
            r'[!@#$%^&*()+={}\[\]|\\:;"`~<>?/]': 'invalid_symbols',
            r'[\u2000-\u206F\u2E00-\u2E7F\\\'"]': 'unusual_unicode',
            r'\s{2,}': 'multiple_spaces',
            r'^\s+|\s+$': 'leading_trailing_spaces'
        }

# Copyright (c) 2025 Mohamed Z. Hatim
        self.error_categories = {
            'incomplete_binomial': [],
            'formatting_issues': [],
            'author_citations': [],
            'hybrid_markers': [],
            'infraspecific_issues': [],
            'placeholder_names': [],
            'invalid_characters': [],
            'missing_components': [],
            'capitalization_errors': [],
            'spacing_errors': []
        }
        
    def clean_species_name(self, name: str) -> str:
        """
        Clean a single species name.
        
        Parameters:
        -----------
        name : str
            Raw species name
            
        Returns:
        --------
        str
            Cleaned species name
        """
        if pd.isna(name) or not isinstance(name, str):
            return ''
        
# Copyright (c) 2025 Mohamed Z. Hatim
        name = name.strip()
        name = re.sub(r'\s+', ' ', name)  # Multiple spaces to single
        
# Copyright (c) 2025 Mohamed Z. Hatim
        for pattern in self.author_patterns:
            name = re.sub(pattern, '', name)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        for marker in self.infraspecific_markers.keys():
            if marker in name:
                parts = name.split(marker)
                if len(parts) >= 2:
                    genus_species = parts[0].strip()
                    infraspecific = parts[1].strip()
                    name = f"{genus_species} {marker} {infraspecific}"
                break
        
# Copyright (c) 2025 Mohamed Z. Hatim
        words = name.split()
        if len(words) >= 2:
# Copyright (c) 2025 Mohamed Z. Hatim
            words[0] = words[0].capitalize()
            words[1] = words[1].lower()
            
# Copyright (c) 2025 Mohamed Z. Hatim
            for i, word in enumerate(words[2:], 2):
                if word in self.infraspecific_markers.keys():
                    if i + 1 < len(words):
                        words[i + 1] = words[i + 1].lower()
        
        return ' '.join(words).strip()

    def detect_errors(self, name: str) -> Dict[str, Any]:
        """
        Comprehensive error detection and classification for species names.

        Parameters:
        -----------
        name : str
            Species name to analyze

        Returns:
        --------
        dict
            Dictionary with error classifications and details
        """
        if pd.isna(name) or not isinstance(name, str):
            return {
                'is_valid': False,
                'errors': {'missing_components': ['empty_or_null_name']},
                'error_count': 1,
                'severity': 'critical',
                'original_name': name,
                'cleaned_name': '',
                'suggestions': ['Provide a valid species name']
            }

        errors = {category: [] for category in self.error_categories.keys()}
        error_count = 0
        suggestions = []

# Copyright (c) 2025 Mohamed Z. Hatim
        name_clean = name.strip()

# Copyright (c) 2025 Mohamed Z. Hatim
        incomplete_errors = self._detect_incomplete_binomial(name_clean)
        errors['incomplete_binomial'].extend(incomplete_errors)
        error_count += len(incomplete_errors)

# Copyright (c) 2025 Mohamed Z. Hatim
        format_errors = self._detect_formatting_issues(name_clean)
        errors['formatting_issues'].extend(format_errors.get('general', []))
        errors['capitalization_errors'].extend(format_errors.get('capitalization', []))
        errors['spacing_errors'].extend(format_errors.get('spacing', []))
        error_count += len(format_errors.get('general', []))
        error_count += len(format_errors.get('capitalization', []))
        error_count += len(format_errors.get('spacing', []))

# Copyright (c) 2025 Mohamed Z. Hatim
        author_errors = self._detect_author_citations(name_clean)
        errors['author_citations'].extend(author_errors)
        error_count += len(author_errors)

# Copyright (c) 2025 Mohamed Z. Hatim
        hybrid_errors = self._detect_hybrid_markers(name_clean)
        errors['hybrid_markers'].extend(hybrid_errors)
        error_count += len(hybrid_errors)

# Copyright (c) 2025 Mohamed Z. Hatim
        infraspecific_errors = self._detect_infraspecific_issues(name_clean)
        errors['infraspecific_issues'].extend(infraspecific_errors)
        error_count += len(infraspecific_errors)

# Copyright (c) 2025 Mohamed Z. Hatim
        placeholder_errors = self._detect_placeholder_names(name_clean)
        errors['placeholder_names'].extend(placeholder_errors)
        error_count += len(placeholder_errors)

# Copyright (c) 2025 Mohamed Z. Hatim
        invalid_char_errors = self._detect_invalid_characters(name_clean)
        errors['invalid_characters'].extend(invalid_char_errors)
        error_count += len(invalid_char_errors)

# Copyright (c) 2025 Mohamed Z. Hatim
        missing_errors = self._detect_missing_components(name_clean)
        errors['missing_components'].extend(missing_errors)
        error_count += len(missing_errors)

# Copyright (c) 2025 Mohamed Z. Hatim
        suggestions = self._generate_suggestions(name_clean, errors)

# Copyright (c) 2025 Mohamed Z. Hatim
        severity = self._determine_severity(errors, error_count)

# Copyright (c) 2025 Mohamed Z. Hatim
        filtered_errors = {k: v for k, v in errors.items() if v}

        return {
            'is_valid': error_count == 0,
            'errors': filtered_errors,
            'error_count': error_count,
            'severity': severity,
            'original_name': name,
            'cleaned_name': self.clean_species_name(name),
            'suggestions': suggestions
        }

    def _detect_incomplete_binomial(self, name: str) -> List[str]:
        """Detect incomplete binomial names (genus only, species only)."""
        errors = []
        words = name.strip().split()

        if len(words) == 0:
            errors.append('empty_name')
        elif len(words) == 1:
            word = words[0].lower()
            if word in [marker.lower().rstrip('.') for marker in self.placeholder_patterns.keys()]:
                errors.append('single_placeholder_word')
            elif word[0].isupper():
                errors.append('genus_only')
            elif word[0].islower():
                errors.append('species_epithet_only')
            else:
                errors.append('single_word_unknown_type')
        elif len(words) > 1:
# Copyright (c) 2025 Mohamed Z. Hatim
            if not words[0][0].isupper():
                errors.append('genus_not_capitalized')
# Copyright (c) 2025 Mohamed Z. Hatim
            if words[1][0].isupper():
                errors.append('species_epithet_capitalized')

        return errors

    def _detect_formatting_issues(self, name: str) -> Dict[str, List[str]]:
        """Detect various formatting issues."""
        errors = {'general': [], 'capitalization': [], 'spacing': []}

# Copyright (c) 2025 Mohamed Z. Hatim
        if re.search(r'\s{2,}', name):
            errors['spacing'].append('multiple_consecutive_spaces')
        if name.startswith(' ') or name.endswith(' '):
            errors['spacing'].append('leading_trailing_spaces')
        if re.search(r'\s+[.,;:]', name):
            errors['spacing'].append('space_before_punctuation')

# Copyright (c) 2025 Mohamed Z. Hatim
        words = name.split()
        if len(words) >= 2:
            if not words[0][0].isupper():
                errors['capitalization'].append('genus_not_capitalized')
            if words[0] != words[0].capitalize():
                errors['capitalization'].append('genus_improper_case')
            if words[1][0].isupper():
                errors['capitalization'].append('species_epithet_capitalized')
            if words[1] != words[1].lower():
                errors['capitalization'].append('species_epithet_improper_case')

# Copyright (c) 2025 Mohamed Z. Hatim
        for word in words:
            if any(c.isupper() for c in word[1:]) and any(c.islower() for c in word[1:]):
                errors['capitalization'].append('mixed_case_within_word')
                break

        return errors

    def _detect_author_citations(self, name: str) -> List[str]:
        """Detect author citations that should be flagged or removed."""
        errors = []

        for pattern in self.author_patterns:
            if re.search(pattern, name, re.IGNORECASE):
                errors.append(f'author_citation_detected')
                break

# Copyright (c) 2025 Mohamed Z. Hatim
        if re.search(r'\b[A-Z][a-z]+\s+\d{4}\b', name):
            errors.append('author_with_year')
        if re.search(r'\([^)]+\)', name):
            errors.append('parenthetical_author')
        if re.search(r'\b[A-Z]\.$', name):
            errors.append('abbreviated_author')

        return errors

    def _detect_hybrid_markers(self, name: str) -> List[str]:
        """Detect hybrid markers and handle them correctly."""
        errors = []

        for marker, marker_type in self.hybrid_patterns.items():
            if marker in name:
                errors.append(f'hybrid_marker_{marker_type}')

# Copyright (c) 2025 Mohamed Z. Hatim
        if '\u00d7' in name and not re.search(r'\u00d7\s*[A-Z][a-z]+', name):
            errors.append('malformed_hybrid_multiplication')
        if ' x ' in name.lower() and not re.search(r'\s+x\s+[A-Z][a-z]+', name, re.IGNORECASE):
            errors.append('malformed_hybrid_letter_x')

        return errors

    def _detect_infraspecific_issues(self, name: str) -> List[str]:
        """Detect and validate infraspecific rank formatting."""
        errors = []

        for marker in self.infraspecific_markers.keys():
            if marker in name:
# Copyright (c) 2025 Mohamed Z. Hatim
                parts = name.split(marker)
                if len(parts) != 2:
                    errors.append(f'malformed_infraspecific_{marker.replace(".", "")}')
                else:
                    before = parts[0].strip()
                    after = parts[1].strip()

# Copyright (c) 2025 Mohamed Z. Hatim
                    before_words = before.split()
                    if len(before_words) < 2:
                        errors.append('incomplete_binomial_before_infraspecific')

# Copyright (c) 2025 Mohamed Z. Hatim
                    if not after or not after.split():
                        errors.append('missing_infraspecific_epithet')

# Copyright (c) 2025 Mohamed Z. Hatim
                    after_words = after.split()
                    if after_words and after_words[0][0].isupper():
                        errors.append('infraspecific_epithet_capitalized')

        return errors

    def _detect_placeholder_names(self, name: str) -> List[str]:
        """Detect anonymous/placeholder names."""
        errors = []

        name_lower = name.lower()
        for pattern, placeholder_type in self.placeholder_patterns.items():
            if re.search(pattern, name_lower):
                errors.append(placeholder_type)

# Copyright (c) 2025 Mohamed Z. Hatim
        if re.search(r'\b\d+\b', name_lower):  # Numbers suggesting specimen numbers
            errors.append('specimen_number_in_name')

        return errors

    def _detect_invalid_characters(self, name: str) -> List[str]:
        """Detect invalid characters and symbols."""
        errors = []

        for pattern, error_type in self.invalid_patterns.items():
            if re.search(pattern, name):
                if error_type not in errors:  # Avoid duplicates
                    errors.append(error_type)

# Copyright (c) 2025 Mohamed Z. Hatim
        if re.search(r'[^\w\s\u00d7\.\-]', name):
            errors.append('non_standard_characters')

        return errors

    def _detect_missing_components(self, name: str) -> List[str]:
        """Detect missing genus or species epithets."""
        errors = []
        words = name.strip().split()

        if len(words) == 0:
            errors.append('completely_empty')
        elif len(words) == 1:
# Copyright (c) 2025 Mohamed Z. Hatim
            if not any(re.search(pattern, name.lower()) for pattern in self.placeholder_patterns.keys()):
                if words[0][0].isupper():
                    errors.append('missing_species_epithet')
                else:
                    errors.append('missing_genus')

        return errors

    def _generate_suggestions(self, name: str, errors: Dict[str, List[str]]) -> List[str]:
        """Generate suggestions for fixing detected errors."""
        suggestions = []

# Copyright (c) 2025 Mohamed Z. Hatim
        if errors['incomplete_binomial']:
            if 'genus_only' in errors['incomplete_binomial']:
                suggestions.append('Add species epithet after genus name')
            if 'species_epithet_only' in errors['incomplete_binomial']:
                suggestions.append('Add genus name before species epithet')

        if errors['formatting_issues'] or errors['capitalization_errors']:
            suggestions.append('Use proper scientific name formatting (Genus species)')

        if errors['author_citations']:
            suggestions.append('Remove author citations from species name')

        if errors['placeholder_names']:
            suggestions.append('Replace placeholder terms with actual species identification')

        if errors['invalid_characters']:
            suggestions.append('Remove invalid characters and symbols')

        if errors['hybrid_markers']:
            suggestions.append('Use proper hybrid notation (\u00d7 for hybrids)')

        if errors['infraspecific_issues']:
            suggestions.append('Check infraspecific rank formatting (var., subsp., etc.)')

# Copyright (c) 2025 Mohamed Z. Hatim
        if not suggestions:
            suggestions.append('Use standardized scientific nomenclature')

        return suggestions

    def _determine_severity(self, errors: Dict[str, List[str]], error_count: int) -> str:
        """Determine the severity level of detected errors."""
        if error_count == 0:
            return 'none'

# Copyright (c) 2025 Mohamed Z. Hatim
        critical_categories = ['missing_components', 'incomplete_binomial', 'placeholder_names']
        if any(errors[cat] for cat in critical_categories):
            return 'critical'

# Copyright (c) 2025 Mohamed Z. Hatim
        high_severity_categories = ['invalid_characters']
        if any(errors[cat] for cat in high_severity_categories):
            return 'high'

# Copyright (c) 2025 Mohamed Z. Hatim
        medium_severity_categories = ['author_citations', 'infraspecific_issues', 'hybrid_markers']
        if any(errors[cat] for cat in medium_severity_categories):
            return 'medium'

# Copyright (c) 2025 Mohamed Z. Hatim
        return 'low'

    def validate_species_name(self, name: str) -> Dict[str, Any]:
        """
        Comprehensive validation of a species name with detailed reporting.

        Parameters:
        -----------
        name : str
            Species name to validate

        Returns:
        --------
        dict
            Complete validation report
        """
        return self.detect_errors(name)

    def classify_name_type(self, name: str) -> str:
        """
        Classify the type of taxonomic name.

        Parameters:
        -----------
        name : str
            Species name to classify

        Returns:
        --------
        str
            Classification of name type
        """
        if pd.isna(name) or not isinstance(name, str):
            return 'invalid'

        name_clean = name.strip()
        words = name_clean.split()

# Copyright (c) 2025 Mohamed Z. Hatim
        for pattern in self.placeholder_patterns.keys():
            if re.search(pattern, name_clean.lower()):
                return 'placeholder'

# Copyright (c) 2025 Mohamed Z. Hatim
        for marker in self.hybrid_patterns.keys():
            if marker in name_clean:
                return 'hybrid'

# Copyright (c) 2025 Mohamed Z. Hatim
        for marker in self.infraspecific_markers.keys():
            if marker in name_clean:
                return 'infraspecific'

# Copyright (c) 2025 Mohamed Z. Hatim
        if len(words) == 0:
            return 'empty'
        elif len(words) == 1:
            if words[0][0].isupper():
                return 'genus_only'
            else:
                return 'epithet_only'
        elif len(words) == 2:
            if words[0][0].isupper() and words[1][0].islower():
                return 'binomial'
            else:
                return 'malformed_binomial'
        elif len(words) > 2:
# Copyright (c) 2025 Mohamed Z. Hatim
            if any(re.search(pattern, name_clean) for pattern in self.author_patterns):
                return 'binomial_with_author'
            else:
                return 'trinomial_or_complex'

        return 'unknown'

    def batch_validate_names(self, names: List[str]) -> pd.DataFrame:
        """
        Validate a batch of species names and return detailed results.

        Parameters:
        -----------
        names : list
            List of species names to validate

        Returns:
        --------
        pd.DataFrame
            DataFrame with validation results for each name
        """
        results = []

        for i, name in enumerate(names):
            validation = self.validate_species_name(name)

            result = {
                'index': i,
                'original_name': name,
                'cleaned_name': validation['cleaned_name'],
                'name_type': self.classify_name_type(name),
                'is_valid': validation['is_valid'],
                'error_count': validation['error_count'],
                'severity': validation['severity'],
                'errors_summary': '; '.join([f"{k}: {len(v)}" for k, v in validation['errors'].items() if v]),
                'suggestions': '; '.join(validation['suggestions'])
            }

# Copyright (c) 2025 Mohamed Z. Hatim
            for category in self.error_categories.keys():
                result[f'has_{category}'] = category in validation['errors'] and len(validation['errors'][category]) > 0

            results.append(result)

        return pd.DataFrame(results)

    def standardize_dataframe(self, df: pd.DataFrame,
                            species_column: str = 'species',
                            include_error_detection: bool = True,
                            error_columns_prefix: str = 'name_') -> pd.DataFrame:
        """
        Standardize species names in a dataframe with comprehensive error detection.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        species_column : str
            Name of the species column
        include_error_detection : bool
            Whether to include error detection columns
        error_columns_prefix : str
            Prefix for error detection columns

        Returns:
        --------
        pd.DataFrame
            Dataframe with standardized species names and error detection
        """
        if species_column not in df.columns:
            return df

        df_clean = df.copy()

# Copyright (c) 2025 Mohamed Z. Hatim
        df_clean[f'{species_column}_original'] = df_clean[species_column]

# Copyright (c) 2025 Mohamed Z. Hatim
        df_clean[species_column] = df_clean[species_column].apply(self.clean_species_name)

# Copyright (c) 2025 Mohamed Z. Hatim
        df_clean['genus'] = df_clean[species_column].apply(
            lambda x: x.split()[0] if x and len(x.split()) > 0 else ''
        )

        df_clean['specific_epithet'] = df_clean[species_column].apply(
            lambda x: x.split()[1] if x and len(x.split()) > 1 else ''
        )

# Copyright (c) 2025 Mohamed Z. Hatim
        if include_error_detection:
# Copyright (c) 2025 Mohamed Z. Hatim
            validation_results = []
            for name in df_clean[f'{species_column}_original']:
                validation = self.validate_species_name(name)
                validation_results.append(validation)

# Copyright (c) 2025 Mohamed Z. Hatim
            df_clean[f'{error_columns_prefix}is_valid'] = [r['is_valid'] for r in validation_results]
            df_clean[f'{error_columns_prefix}error_count'] = [r['error_count'] for r in validation_results]
            df_clean[f'{error_columns_prefix}severity'] = [r['severity'] for r in validation_results]
            df_clean[f'{error_columns_prefix}type'] = df_clean[f'{species_column}_original'].apply(self.classify_name_type)

# Copyright (c) 2025 Mohamed Z. Hatim
            for category in self.error_categories.keys():
                df_clean[f'{error_columns_prefix}has_{category}'] = [
                    category in r['errors'] and len(r['errors'][category]) > 0
                    for r in validation_results
                ]

# Copyright (c) 2025 Mohamed Z. Hatim
            df_clean[f'{error_columns_prefix}errors_summary'] = [
                '; '.join([f"{k}: {len(v)}" for k, v in r['errors'].items() if v])
                for r in validation_results
            ]

            df_clean[f'{error_columns_prefix}suggestions'] = [
                '; '.join(r['suggestions']) for r in validation_results
            ]

        return df_clean

    def generate_error_report(self, df: pd.DataFrame,
                            species_column: str = 'species') -> Dict[str, Any]:
        """
        Generate a comprehensive error report for species names in a DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        species_column : str
            Name of the species column

        Returns:
        --------
        dict
            Comprehensive error report with statistics and examples
        """
        if species_column not in df.columns:
            return {'error': f'Column {species_column} not found in dataframe'}

# Copyright (c) 2025 Mohamed Z. Hatim
        names = df[species_column].dropna().astype(str).tolist()
        validation_df = self.batch_validate_names(names)

# Copyright (c) 2025 Mohamed Z. Hatim
        total_names = len(names)
        valid_names = validation_df['is_valid'].sum()
        invalid_names = total_names - valid_names

# Copyright (c) 2025 Mohamed Z. Hatim
        error_stats = {}
        for category in self.error_categories.keys():
            error_count = validation_df[f'has_{category}'].sum()
            error_stats[category] = {
                'count': int(error_count),
                'percentage': round((error_count / total_names) * 100, 2)
            }

# Copyright (c) 2025 Mohamed Z. Hatim
        severity_counts = validation_df['severity'].value_counts().to_dict()
        severity_stats = {k: {'count': int(v), 'percentage': round((v / total_names) * 100, 2)}
                         for k, v in severity_counts.items()}

# Copyright (c) 2025 Mohamed Z. Hatim
        name_type_counts = validation_df['name_type'].value_counts().to_dict()
        name_type_stats = {k: {'count': int(v), 'percentage': round((v / total_names) * 100, 2)}
                          for k, v in name_type_counts.items()}

# Copyright (c) 2025 Mohamed Z. Hatim
        problematic_examples = {}
        for category in self.error_categories.keys():
            if error_stats[category]['count'] > 0:
                examples = validation_df[validation_df[f'has_{category}']]['original_name'].head(5).tolist()
                problematic_examples[category] = examples

# Copyright (c) 2025 Mohamed Z. Hatim
        error_combinations = validation_df[validation_df['error_count'] > 0]['errors_summary'].value_counts().head(10).to_dict()

        report = {
            'summary': {
                'total_names': total_names,
                'valid_names': valid_names,
                'invalid_names': invalid_names,
                'validity_percentage': round((valid_names / total_names) * 100, 2)
            },
            'error_statistics': error_stats,
            'severity_distribution': severity_stats,
            'name_type_distribution': name_type_stats,
            'problematic_examples': problematic_examples,
            'common_error_combinations': error_combinations,
            'recommendations': self._generate_report_recommendations(error_stats, severity_stats)
        }

        return report

    def _generate_report_recommendations(self, error_stats: Dict, severity_stats: Dict) -> List[str]:
        """Generate recommendations based on error statistics."""
        recommendations = []

# Copyright (c) 2025 Mohamed Z. Hatim
        if severity_stats.get('critical', {}).get('count', 0) > 0:
            recommendations.append("CRITICAL: Address missing or incomplete species names immediately")

# Copyright (c) 2025 Mohamed Z. Hatim
        if error_stats.get('placeholder_names', {}).get('percentage', 0) > 10:
            recommendations.append("High priority: Replace placeholder names (sp., cf., etc.) with proper identifications")

        if error_stats.get('incomplete_binomial', {}).get('percentage', 0) > 5:
            recommendations.append("High priority: Complete incomplete binomial names")

# Copyright (c) 2025 Mohamed Z. Hatim
        if error_stats.get('author_citations', {}).get('percentage', 0) > 15:
            recommendations.append("Medium priority: Remove author citations from species names")

        if error_stats.get('formatting_issues', {}).get('percentage', 0) > 20:
            recommendations.append("Medium priority: Standardize formatting (capitalization, spacing)")

# Copyright (c) 2025 Mohamed Z. Hatim
        if error_stats.get('infraspecific_issues', {}).get('percentage', 0) > 5:
            recommendations.append("Low priority: Review infraspecific rank formatting")

        if not recommendations:
            recommendations.append("Data quality is good - only minor formatting improvements needed")

        return recommendations
    
    def fuzzy_match_species(self, 
                           query_species: List[str],
                           reference_species: List[str],
                           threshold: int = 80) -> Dict[str, str]:
        """
        Fuzzy match species names against a reference list.
        
        Parameters:
        -----------
        query_species : list
            Species names to match
        reference_species : list
            Reference species list
        threshold : int
            Minimum match score (0-100)
            
        Returns:
        --------
        dict
            Mapping of query species to best matches
        """
        matches = {}
        
        for query in query_species:
            if not query or query == '':
                continue
                
            best_match, score = process.extractOne(query, reference_species)
            
            if score >= threshold:
                matches[query] = best_match
            else:
                matches[query] = query  # Keep original if no good match
        
        return matches


class CoordinateStandardizer:
    """Standardize coordinate data and handle different formats."""
    
    def __init__(self):
        self.coordinate_patterns = {
            'decimal': r'^-?\d+\.?\d*$',
            'dms': r'(\d+)[°d]\s*(\d+)[\'m]\s*([\d.]+)[\"s]?\s*([NSEW])?',
            'dm': r'(\d+)[°d]\s*([\d.]+)[\'m]\s*([NSEW])?'
        }
    
    def standardize_coordinates(self, df: pd.DataFrame,
                              lat_col: str = 'latitude',
                              lon_col: str = 'longitude') -> pd.DataFrame:
        """
        Standardize coordinate formats to decimal degrees.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        lat_col : str
            Latitude column name
        lon_col : str
            Longitude column name
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with standardized coordinates
        """
        df_std = df.copy()
        
        if lat_col in df.columns:
            df_std[lat_col] = df_std[lat_col].apply(self._convert_to_decimal)
            df_std[lat_col] = pd.to_numeric(df_std[lat_col], errors='coerce')
        
        if lon_col in df.columns:
            df_std[lon_col] = df_std[lon_col].apply(self._convert_to_decimal)
            df_std[lon_col] = pd.to_numeric(df_std[lon_col], errors='coerce')
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if lat_col in df_std.columns:
            invalid_lat = (df_std[lat_col] < -90) | (df_std[lat_col] > 90)
            if invalid_lat.any():
                warnings.warn(f"Found {invalid_lat.sum()} invalid latitude values")
                df_std.loc[invalid_lat, lat_col] = np.nan
        
        if lon_col in df_std.columns:
            invalid_lon = (df_std[lon_col] < -180) | (df_std[lon_col] > 180)
            if invalid_lon.any():
                warnings.warn(f"Found {invalid_lon.sum()} invalid longitude values")
                df_std.loc[invalid_lon, lon_col] = np.nan
        
        return df_std
    
    def _convert_to_decimal(self, coord_str) -> float:
        """Convert coordinate string to decimal degrees."""
        if pd.isna(coord_str):
            return np.nan
        
        coord_str = str(coord_str).strip()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if re.match(self.coordinate_patterns['decimal'], coord_str):
            return float(coord_str)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        dms_match = re.match(self.coordinate_patterns['dms'], coord_str, re.IGNORECASE)
        if dms_match:
            degrees, minutes, seconds, hemisphere = dms_match.groups()
            decimal = float(degrees) + float(minutes)/60 + float(seconds)/3600
            
            if hemisphere and hemisphere.upper() in ['S', 'W']:
                decimal = -decimal
            
            return decimal
        
# Copyright (c) 2025 Mohamed Z. Hatim
        dm_match = re.match(self.coordinate_patterns['dm'], coord_str, re.IGNORECASE)
        if dm_match:
            degrees, minutes, hemisphere = dm_match.groups()
            decimal = float(degrees) + float(minutes)/60
            
            if hemisphere and hemisphere.upper() in ['S', 'W']:
                decimal = -decimal
            
            return decimal
        
# Copyright (c) 2025 Mohamed Z. Hatim
        numeric = re.findall(r'-?\d+\.?\d*', coord_str)
        if numeric:
            return float(numeric[0])
        
        return np.nan