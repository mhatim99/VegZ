"""
Data parsing utilities for vegetation survey data from various formats.

Copyright (c) 2025 Mohamed Z. Hatim
"""

import pandas as pd
from typing import Dict, List, Optional, Union
from pathlib import Path
import warnings

from .._compat import require

# openpyxl / xlrd are imported lazily inside _parse_excel. Importing them at
# module scope made them hard requirements of `import VegZ`, even for users who
# only ever read CSV files.


class VegetationDataParser:
    """Main parser for vegetation survey data from multiple formats."""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.txt', '.tab']
        self.parsers = {
            '.csv': self._parse_csv,
            '.xlsx': self._parse_excel,
            '.xls': self._parse_excel,
            '.txt': self._parse_text,
            '.tab': self._parse_text
        }
    
    def parse(self, filepath: Union[str, Path], 
              format_type: Optional[str] = None,
              **kwargs) -> pd.DataFrame:
        """
        Parse vegetation data from file.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to the data file
        format_type : str, optional
            Force specific format parsing
        **kwargs : dict
            Additional arguments for specific parsers
            
        Returns:
        --------
        pd.DataFrame
            Parsed vegetation data
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if format_type:
            extension = f".{format_type.lower()}"
        else:
            extension = filepath.suffix.lower()
        
        if extension not in self.supported_formats:
            raise ValueError(f"Unsupported format: {extension}")
        
        parser_func = self.parsers[extension]
        return parser_func(filepath, **kwargs)
    
    def _parse_csv(self, filepath: Path, **kwargs) -> pd.DataFrame:
        """Parse CSV files, sniffing the delimiter when it is not supplied."""
        # Pop rather than get: leaving these in kwargs and also passing them
        # explicitly to read_csv raises "got multiple values for keyword".
        encoding = kwargs.pop('encoding', 'utf-8')
        separator = kwargs.pop('sep', None)

        if separator is None:
            separator = self._sniff_separator(filepath, encoding)

        try:
            df = pd.read_csv(filepath, sep=separator, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            for enc in ['utf-8-sig', 'latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    df = pd.read_csv(filepath, sep=separator, encoding=enc, **kwargs)
                    warnings.warn(f"Used encoding {enc} instead of {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not decode file with any common encoding")

        return self._standardize_columns(df)

    @staticmethod
    def _sniff_separator(filepath: Path, encoding: str) -> str:
        """Guess the delimiter from the header line."""
        try:
            with open(filepath, 'r', encoding=encoding, errors='replace') as f:
                first_line = f.readline()
        except OSError:  # pragma: no cover - re-raised by the caller anyway
            return ','

        # Pick whichever candidate appears most often, not merely first.
        candidates = ['\t', ';', ',', '|']
        counts = {sep: first_line.count(sep) for sep in candidates}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else ','

    def _parse_excel(self, filepath: Path, **kwargs) -> pd.DataFrame:
        """Parse Excel files (requires openpyxl for .xlsx, xlrd for legacy .xls)."""
        sheet_name = kwargs.pop('sheet_name', 0)

        suffix = filepath.suffix.lower()
        if suffix == '.xls':
            require('xlrd', 'Reading legacy .xls files', extra='excel')
        else:
            require('openpyxl', 'Reading .xlsx files', extra='excel')

        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
        except Exception as e:
            raise ValueError(f"Error reading Excel file: {e}") from e

        if isinstance(df, dict):
            raise ValueError(
                "sheet_name selected multiple sheets; pass a single sheet name "
                "or index to parse()."
            )

        return self._standardize_columns(df)

    def _parse_text(self, filepath: Path, **kwargs) -> pd.DataFrame:
        """Parse text/tab-delimited files."""
        kwargs.setdefault('sep', '\t' if filepath.suffix.lower() == '.tab' else None)
        return self._parse_csv(filepath, **kwargs)
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and common data issues."""
# Copyright (c) 2025 Mohamed Z. Hatim
        df.columns = df.columns.str.strip().str.lower()
        df.columns = df.columns.str.replace(' ', '_')
        df.columns = df.columns.str.replace('[^a-zA-Z0-9_]', '', regex=True)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        column_mappings = {
            'species': ['species_name', 'taxon', 'scientific_name'],
            'abundance': ['cover', 'coverage', 'abundance_value', 'percent_cover'],
            'plot_id': ['plot', 'site', 'site_id', 'releve', 'quadrat'],
            'latitude': ['lat', 'y', 'coord_y'],
            'longitude': ['lon', 'lng', 'long', 'x', 'coord_x'],
            'date': ['sampling_date', 'survey_date', 'collection_date']
        }
        
        for standard, variants in column_mappings.items():
            for variant in variants:
                if variant in df.columns and standard not in df.columns:
                    df = df.rename(columns={variant: standard})
                    break
        
        return df


class TurbovegParser(VegetationDataParser):
    """Specialized parser for Turboveg export files."""
    
    def __init__(self):
        super().__init__()
        self.turboveg_formats = {
            'species_list': self._parse_species_list,
            'releves': self._parse_releves,
            'header_data': self._parse_header_data
        }
    
    def parse_turboveg_export(self, export_dir: Union[str, Path]) -> Dict[str, pd.DataFrame]:
        """
        Parse complete Turboveg export directory.
        
        Parameters:
        -----------
        export_dir : str or Path
            Directory containing Turboveg export files
            
        Returns:
        --------
        dict
            Dictionary with parsed data tables
        """
        export_dir = Path(export_dir)
        
        if not export_dir.is_dir():
            raise ValueError(f"Directory not found: {export_dir}")
        
        results = {}
        
# Copyright (c) 2025 Mohamed Z. Hatim
        file_patterns = {
            'species_list': ['species.txt', 'taxa.txt', 'florlist.txt'],
            'releves': ['releves.txt', 'vegetation.txt', 'plots.txt'],
            'header_data': ['header.txt', 'sites.txt', 'metadata.txt']
        }
        
        for data_type, patterns in file_patterns.items():
            for pattern in patterns:
                filepath = export_dir / pattern
                if filepath.exists():
                    parser_func = self.turboveg_formats[data_type]
                    results[data_type] = parser_func(filepath)
                    break
        
        if not results:
# Copyright (c) 2025 Mohamed Z. Hatim
            for filepath in export_dir.glob('*.txt'):
                try:
                    results[filepath.stem] = self.parse(filepath)
                except Exception:
                    continue
        
        return results
    
    @staticmethod
    def _rename_positional(df: pd.DataFrame, expected_cols: List[str]) -> pd.DataFrame:
        """
        Rename the leading columns by position.

        Assigning into ``df.columns.values`` mutates the Index's backing array
        in place, which pandas treats as undefined behaviour (an Index is meant
        to be immutable, and the write may not propagate).
        """
        columns = list(df.columns)
        for i, name in enumerate(expected_cols):
            if i < len(columns):
                columns[i] = name
        df = df.copy()
        df.columns = columns
        return df

    def _parse_species_list(self, filepath: Path) -> pd.DataFrame:
        """Parse Turboveg species list file."""
        df = self._parse_text(filepath, sep='\t')
        
        return self._rename_positional(
            df, ['species_nr', 'species_name', 'author', 'family'])
    
    def _parse_releves(self, filepath: Path) -> pd.DataFrame:
        """Parse Turboveg releves/vegetation data."""
        df = self._parse_text(filepath, sep='\t')
        
        return self._rename_positional(
            df, ['releve_nr', 'species_nr', 'layer', 'cover_code', 'abundance'])
    
    def _parse_header_data(self, filepath: Path) -> pd.DataFrame:
        """Parse Turboveg header/site data."""
        df = self._parse_text(filepath, sep='\t')
        
        return self._rename_positional(
            df, ['releve_nr', 'date', 'author', 'latitude', 'longitude', 'altitude'])


class AgencyDataParser(VegetationDataParser):
    """Parser for agency-specific vegetation data formats."""
    
    def __init__(self):
        super().__init__()
        self.agency_parsers = {
            'usfs': self._parse_usfs,
            'nps': self._parse_nps,
            'epa': self._parse_epa,
            'fia': self._parse_fia
        }
    
    def parse_agency_data(self, filepath: Union[str, Path], 
                         agency: str, **kwargs) -> pd.DataFrame:
        """
        Parse agency-specific vegetation data.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to the data file
        agency : str
            Agency code ('usfs', 'nps', 'epa', 'fia')
        **kwargs : dict
            Additional parsing arguments
            
        Returns:
        --------
        pd.DataFrame
            Parsed and standardized data
        """
        agency = agency.lower()
        
        if agency not in self.agency_parsers:
            raise ValueError(f"Unsupported agency: {agency}")
        
        parser_func = self.agency_parsers[agency]
        return parser_func(filepath, **kwargs)
    
    def _parse_usfs(self, filepath: Path, **kwargs) -> pd.DataFrame:
        """Parse USFS Forest Inventory data."""
        df = self.parse(filepath, **kwargs)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        usfs_mappings = {
            'plot': 'plot_id',
            'spcd': 'species_code',
            'lat': 'latitude',
            'lon': 'longitude'
        }
        
        for old, new in usfs_mappings.items():
            if old in df.columns:
                df = df.rename(columns={old: new})
        
        return df
    
    def _parse_nps(self, filepath: Path, **kwargs) -> pd.DataFrame:
        """Parse National Park Service data."""
        df = self.parse(filepath, **kwargs)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        nps_mappings = {
            'plot_code': 'plot_id',
            'taxa_code': 'species_code',
            'pct_cover': 'abundance'
        }
        
        for old, new in nps_mappings.items():
            if old in df.columns:
                df = df.rename(columns={old: new})
        
        return df
    
    def _parse_epa(self, filepath: Path, **kwargs) -> pd.DataFrame:
        """Parse EPA environmental data."""
        df = self.parse(filepath, **kwargs)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        epa_mappings = {
            'site_id': 'plot_id',
            'taxon': 'species',
            'percent_cover': 'abundance'
        }
        
        for old, new in epa_mappings.items():
            if old in df.columns:
                df = df.rename(columns={old: new})
        
        return df
    
    def _parse_fia(self, filepath: Path, **kwargs) -> pd.DataFrame:
        """Parse Forest Inventory and Analysis data."""
        df = self.parse(filepath, **kwargs)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        fia_mappings = {
            'plt_cn': 'plot_id',
            'spcd': 'species_code',
            'dia': 'diameter',
            'ht': 'height'
        }
        
        for old, new in fia_mappings.items():
            if old in df.columns:
                df = df.rename(columns={old: new})
        
        return df