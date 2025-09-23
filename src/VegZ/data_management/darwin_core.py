"""
Darwin Core standards implementation for biodiversity data.

Copyright (c) 2025 Mohamed Z. Hatim
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import warnings
import uuid


class DarwinCoreHandler:
    """Handler for Darwin Core standard biodiversity data."""
    
    def __init__(self):
        self.dwc_terms = self._load_dwc_terms()
        self.required_terms = [
            'basisOfRecord', 'occurrenceID', 'catalogNumber',
            'scientificName', 'kingdom', 'decimalLatitude', 'decimalLongitude'
        ]
        
    def _load_dwc_terms(self) -> Dict[str, Dict[str, Any]]:
        """Load Darwin Core term definitions."""
        return {
# Copyright (c) 2025 Mohamed Z. Hatim
            'type': {'category': 'record', 'datatype': 'string'},
            'modified': {'category': 'record', 'datatype': 'datetime'},
            'language': {'category': 'record', 'datatype': 'string'},
            'license': {'category': 'record', 'datatype': 'string'},
            'rightsHolder': {'category': 'record', 'datatype': 'string'},
            'accessRights': {'category': 'record', 'datatype': 'string'},
            'bibliographicCitation': {'category': 'record', 'datatype': 'string'},
            'references': {'category': 'record', 'datatype': 'string'},
            'institutionID': {'category': 'record', 'datatype': 'string'},
            'collectionID': {'category': 'record', 'datatype': 'string'},
            'datasetID': {'category': 'record', 'datatype': 'string'},
            'institutionCode': {'category': 'record', 'datatype': 'string'},
            'collectionCode': {'category': 'record', 'datatype': 'string'},
            'datasetName': {'category': 'record', 'datatype': 'string'},
            'ownerInstitutionCode': {'category': 'record', 'datatype': 'string'},
            'basisOfRecord': {'category': 'record', 'datatype': 'string'},
            'informationWithheld': {'category': 'record', 'datatype': 'string'},
            'dataGeneralizations': {'category': 'record', 'datatype': 'string'},
            'dynamicProperties': {'category': 'record', 'datatype': 'string'},
            
# Copyright (c) 2025 Mohamed Z. Hatim
            'occurrenceID': {'category': 'occurrence', 'datatype': 'string'},
            'catalogNumber': {'category': 'occurrence', 'datatype': 'string'},
            'recordNumber': {'category': 'occurrence', 'datatype': 'string'},
            'recordedBy': {'category': 'occurrence', 'datatype': 'string'},
            'individualCount': {'category': 'occurrence', 'datatype': 'integer'},
            'organismQuantity': {'category': 'occurrence', 'datatype': 'string'},
            'organismQuantityType': {'category': 'occurrence', 'datatype': 'string'},
            'sex': {'category': 'occurrence', 'datatype': 'string'},
            'lifeStage': {'category': 'occurrence', 'datatype': 'string'},
            'reproductiveCondition': {'category': 'occurrence', 'datatype': 'string'},
            'behavior': {'category': 'occurrence', 'datatype': 'string'},
            'establishmentMeans': {'category': 'occurrence', 'datatype': 'string'},
            'occurrenceStatus': {'category': 'occurrence', 'datatype': 'string'},
            'preparations': {'category': 'occurrence', 'datatype': 'string'},
            'disposition': {'category': 'occurrence', 'datatype': 'string'},
            'associatedReferences': {'category': 'occurrence', 'datatype': 'string'},
            'associatedSequences': {'category': 'occurrence', 'datatype': 'string'},
            'associatedTaxa': {'category': 'occurrence', 'datatype': 'string'},
            'otherCatalogNumbers': {'category': 'occurrence', 'datatype': 'string'},
            'occurrenceRemarks': {'category': 'occurrence', 'datatype': 'string'},
            
# Copyright (c) 2025 Mohamed Z. Hatim
            'eventID': {'category': 'event', 'datatype': 'string'},
            'parentEventID': {'category': 'event', 'datatype': 'string'},
            'fieldNumber': {'category': 'event', 'datatype': 'string'},
            'eventDate': {'category': 'event', 'datatype': 'datetime'},
            'eventTime': {'category': 'event', 'datatype': 'time'},
            'startDayOfYear': {'category': 'event', 'datatype': 'integer'},
            'endDayOfYear': {'category': 'event', 'datatype': 'integer'},
            'year': {'category': 'event', 'datatype': 'integer'},
            'month': {'category': 'event', 'datatype': 'integer'},
            'day': {'category': 'event', 'datatype': 'integer'},
            'verbatimEventDate': {'category': 'event', 'datatype': 'string'},
            'habitat': {'category': 'event', 'datatype': 'string'},
            'samplingProtocol': {'category': 'event', 'datatype': 'string'},
            'samplingEffort': {'category': 'event', 'datatype': 'string'},
            'sampleSizeValue': {'category': 'event', 'datatype': 'string'},
            'sampleSizeUnit': {'category': 'event', 'datatype': 'string'},
            'fieldNotes': {'category': 'event', 'datatype': 'string'},
            'eventRemarks': {'category': 'event', 'datatype': 'string'},
            
# Copyright (c) 2025 Mohamed Z. Hatim
            'locationID': {'category': 'location', 'datatype': 'string'},
            'higherGeographyID': {'category': 'location', 'datatype': 'string'},
            'higherGeography': {'category': 'location', 'datatype': 'string'},
            'continent': {'category': 'location', 'datatype': 'string'},
            'waterBody': {'category': 'location', 'datatype': 'string'},
            'islandGroup': {'category': 'location', 'datatype': 'string'},
            'island': {'category': 'location', 'datatype': 'string'},
            'country': {'category': 'location', 'datatype': 'string'},
            'countryCode': {'category': 'location', 'datatype': 'string'},
            'stateProvince': {'category': 'location', 'datatype': 'string'},
            'county': {'category': 'location', 'datatype': 'string'},
            'municipality': {'category': 'location', 'datatype': 'string'},
            'locality': {'category': 'location', 'datatype': 'string'},
            'verbatimLocality': {'category': 'location', 'datatype': 'string'},
            'minimumElevationInMeters': {'category': 'location', 'datatype': 'float'},
            'maximumElevationInMeters': {'category': 'location', 'datatype': 'float'},
            'verbatimElevation': {'category': 'location', 'datatype': 'string'},
            'minimumDepthInMeters': {'category': 'location', 'datatype': 'float'},
            'maximumDepthInMeters': {'category': 'location', 'datatype': 'float'},
            'verbatimDepth': {'category': 'location', 'datatype': 'string'},
            'minimumDistanceAboveSurfaceInMeters': {'category': 'location', 'datatype': 'float'},
            'maximumDistanceAboveSurfaceInMeters': {'category': 'location', 'datatype': 'float'},
            'locationAccordingTo': {'category': 'location', 'datatype': 'string'},
            'locationRemarks': {'category': 'location', 'datatype': 'string'},
            'decimalLatitude': {'category': 'location', 'datatype': 'float'},
            'decimalLongitude': {'category': 'location', 'datatype': 'float'},
            'geodeticDatum': {'category': 'location', 'datatype': 'string'},
            'coordinateUncertaintyInMeters': {'category': 'location', 'datatype': 'float'},
            'coordinatePrecision': {'category': 'location', 'datatype': 'float'},
            'pointRadiusSpatialFit': {'category': 'location', 'datatype': 'float'},
            'verbatimCoordinates': {'category': 'location', 'datatype': 'string'},
            'verbatimLatitude': {'category': 'location', 'datatype': 'string'},
            'verbatimLongitude': {'category': 'location', 'datatype': 'string'},
            'verbatimCoordinateSystem': {'category': 'location', 'datatype': 'string'},
            'verbatimSRS': {'category': 'location', 'datatype': 'string'},
            'footprintWKT': {'category': 'location', 'datatype': 'string'},
            'footprintSRS': {'category': 'location', 'datatype': 'string'},
            'footprintSpatialFit': {'category': 'location', 'datatype': 'float'},
            'georeferencedBy': {'category': 'location', 'datatype': 'string'},
            'georeferencedDate': {'category': 'location', 'datatype': 'datetime'},
            'georeferenceProtocol': {'category': 'location', 'datatype': 'string'},
            'georeferenceSources': {'category': 'location', 'datatype': 'string'},
            'georeferenceVerificationStatus': {'category': 'location', 'datatype': 'string'},
            'georeferenceRemarks': {'category': 'location', 'datatype': 'string'},
            
# Copyright (c) 2025 Mohamed Z. Hatim
            'taxonID': {'category': 'taxon', 'datatype': 'string'},
            'scientificNameID': {'category': 'taxon', 'datatype': 'string'},
            'acceptedNameUsageID': {'category': 'taxon', 'datatype': 'string'},
            'parentNameUsageID': {'category': 'taxon', 'datatype': 'string'},
            'originalNameUsageID': {'category': 'taxon', 'datatype': 'string'},
            'nameAccordingToID': {'category': 'taxon', 'datatype': 'string'},
            'namePublishedInID': {'category': 'taxon', 'datatype': 'string'},
            'taxonConceptID': {'category': 'taxon', 'datatype': 'string'},
            'scientificName': {'category': 'taxon', 'datatype': 'string'},
            'acceptedNameUsage': {'category': 'taxon', 'datatype': 'string'},
            'parentNameUsage': {'category': 'taxon', 'datatype': 'string'},
            'originalNameUsage': {'category': 'taxon', 'datatype': 'string'},
            'nameAccordingTo': {'category': 'taxon', 'datatype': 'string'},
            'namePublishedIn': {'category': 'taxon', 'datatype': 'string'},
            'namePublishedInYear': {'category': 'taxon', 'datatype': 'integer'},
            'higherClassification': {'category': 'taxon', 'datatype': 'string'},
            'kingdom': {'category': 'taxon', 'datatype': 'string'},
            'phylum': {'category': 'taxon', 'datatype': 'string'},
            'class': {'category': 'taxon', 'datatype': 'string'},
            'order': {'category': 'taxon', 'datatype': 'string'},
            'family': {'category': 'taxon', 'datatype': 'string'},
            'genus': {'category': 'taxon', 'datatype': 'string'},
            'subgenus': {'category': 'taxon', 'datatype': 'string'},
            'specificEpithet': {'category': 'taxon', 'datatype': 'string'},
            'infraspecificEpithet': {'category': 'taxon', 'datatype': 'string'},
            'taxonRank': {'category': 'taxon', 'datatype': 'string'},
            'verbatimTaxonRank': {'category': 'taxon', 'datatype': 'string'},
            'scientificNameAuthorship': {'category': 'taxon', 'datatype': 'string'},
            'vernacularName': {'category': 'taxon', 'datatype': 'string'},
            'nomenclaturalCode': {'category': 'taxon', 'datatype': 'string'},
            'taxonomicStatus': {'category': 'taxon', 'datatype': 'string'},
            'nomenclaturalStatus': {'category': 'taxon', 'datatype': 'string'},
            'taxonRemarks': {'category': 'taxon', 'datatype': 'string'}
        }
    
    def convert_to_dwc(self, df: pd.DataFrame,
                       field_mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Convert dataset to Darwin Core format.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset
        field_mapping : dict, optional
            Mapping from source fields to DwC terms
            
        Returns:
        --------
        pd.DataFrame
            Darwin Core formatted dataset
        """
        dwc_df = df.copy()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if field_mapping:
            dwc_df = dwc_df.rename(columns=field_mapping)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        auto_mapping = self._get_auto_field_mapping(dwc_df.columns)
        dwc_df = dwc_df.rename(columns=auto_mapping)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        dwc_df = self._add_required_fields(dwc_df)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        dwc_df = self._validate_dwc_types(dwc_df)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        dwc_df = self._add_metadata(dwc_df)
        
        return dwc_df
    
    def validate_dwc_data(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Validate Darwin Core dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Darwin Core dataset
            
        Returns:
        --------
        dict
            Validation results with warnings and errors
        """
        validation_results = {
            'errors': [],
            'warnings': [],
            'info': []
        }
        
# Copyright (c) 2025 Mohamed Z. Hatim
        missing_required = [term for term in self.required_terms if term not in df.columns]
        if missing_required:
            validation_results['errors'].extend([f"Missing required term: {term}" for term in missing_required])
        
# Copyright (c) 2025 Mohamed Z. Hatim
        invalid_terms = [col for col in df.columns if col not in self.dwc_terms and not col.startswith('_')]
        if invalid_terms:
            validation_results['warnings'].extend([f"Non-standard term: {term}" for term in invalid_terms])
        
# Copyright (c) 2025 Mohamed Z. Hatim
        for column in df.columns:
            if column in self.dwc_terms:
                expected_type = self.dwc_terms[column]['datatype']
                validation_results.update(self._validate_column(df, column, expected_type))
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if 'decimalLatitude' in df.columns and 'decimalLongitude' in df.columns:
            invalid_coords = self._validate_coordinates(df)
            if invalid_coords:
                validation_results['warnings'].append(f"Found {invalid_coords} invalid coordinate pairs")
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if 'eventDate' in df.columns:
            invalid_dates = self._validate_dates(df)
            if invalid_dates:
                validation_results['warnings'].append(f"Found {invalid_dates} invalid dates")
        
        return validation_results
    
    def _get_auto_field_mapping(self, columns: List[str]) -> Dict[str, str]:
        """Generate automatic field mapping to DwC terms."""
        mapping = {}
        
# Copyright (c) 2025 Mohamed Z. Hatim
        field_mappings = {
            'species': 'scientificName',
            'species_name': 'scientificName',
            'scientific_name': 'scientificName',
            'latitude': 'decimalLatitude',
            'longitude': 'decimalLongitude',
            'lat': 'decimalLatitude',
            'lon': 'decimalLongitude',
            'lng': 'decimalLongitude',
            'date': 'eventDate',
            'sampling_date': 'eventDate',
            'collection_date': 'eventDate',
            'collector': 'recordedBy',
            'recorded_by': 'recordedBy',
            'plot_id': 'locationID',
            'site_id': 'locationID',
            'elevation': 'minimumElevationInMeters',
            'altitude': 'minimumElevationInMeters',
            'abundance': 'individualCount',
            'count': 'individualCount',
            'cover': 'organismQuantity',
            'coverage': 'organismQuantity'
        }
        
        for col in columns:
            col_lower = col.lower()
            if col_lower in field_mappings and field_mappings[col_lower] not in columns:
                mapping[col] = field_mappings[col_lower]
        
        return mapping
    
    def _add_required_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add missing required fields with default values."""
        dwc_df = df.copy()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if 'occurrenceID' not in dwc_df.columns:
            dwc_df['occurrenceID'] = [str(uuid.uuid4()) for _ in range(len(dwc_df))]
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if 'catalogNumber' not in dwc_df.columns:
            dwc_df['catalogNumber'] = dwc_df.index.astype(str)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if 'basisOfRecord' not in dwc_df.columns:
            dwc_df['basisOfRecord'] = 'HumanObservation'
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if 'kingdom' not in dwc_df.columns and 'scientificName' in dwc_df.columns:
            dwc_df['kingdom'] = 'Plantae'  # Default for vegetation data
        
        return dwc_df
    
    def _validate_dwc_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and convert data types for DwC terms."""
        dwc_df = df.copy()
        
        for column in dwc_df.columns:
            if column in self.dwc_terms:
                expected_type = self.dwc_terms[column]['datatype']
                
                if expected_type == 'integer':
                    dwc_df[column] = pd.to_numeric(dwc_df[column], errors='coerce').astype('Int64')
                elif expected_type == 'float':
                    dwc_df[column] = pd.to_numeric(dwc_df[column], errors='coerce')
                elif expected_type == 'datetime':
                    dwc_df[column] = pd.to_datetime(dwc_df[column], errors='coerce')
                elif expected_type == 'string':
                    dwc_df[column] = dwc_df[column].astype(str).replace('nan', '')
        
        return dwc_df
    
    def _validate_column(self, df: pd.DataFrame, column: str, expected_type: str) -> Dict[str, List[str]]:
        """Validate individual column."""
        results = {'errors': [], 'warnings': [], 'info': []}
        
        if column not in df.columns:
            return results
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if column in self.required_terms:
            null_count = df[column].isnull().sum()
            if null_count > 0:
                results['warnings'].append(f"Required field '{column}' has {null_count} null values")
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if expected_type == 'float':
            try:
                numeric_series = pd.to_numeric(df[column], errors='coerce')
                invalid_count = numeric_series.isnull().sum() - df[column].isnull().sum()
                if invalid_count > 0:
                    results['warnings'].append(f"Field '{column}' has {invalid_count} non-numeric values")
            except:
                pass
        
        return results
    
    def _validate_coordinates(self, df: pd.DataFrame) -> int:
        """Validate coordinate pairs."""
        lat_col = 'decimalLatitude'
        lon_col = 'decimalLongitude'
        
        if lat_col not in df.columns or lon_col not in df.columns:
            return 0
        
# Copyright (c) 2025 Mohamed Z. Hatim
        invalid_lat = (df[lat_col] < -90) | (df[lat_col] > 90)
        invalid_lon = (df[lon_col] < -180) | (df[lon_col] > 180)
        
        return (invalid_lat | invalid_lon).sum()
    
    def _validate_dates(self, df: pd.DataFrame) -> int:
        """Validate date fields."""
        date_col = 'eventDate'
        
        if date_col not in df.columns:
            return 0
        
        try:
            date_series = pd.to_datetime(df[date_col], errors='coerce')
            invalid_count = date_series.isnull().sum() - df[date_col].isnull().sum()
            return invalid_count
        except:
            return 0
    
    def _add_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add metadata fields."""
        dwc_df = df.copy()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if 'modified' not in dwc_df.columns:
            dwc_df['modified'] = datetime.now().isoformat()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if 'language' not in dwc_df.columns:
            dwc_df['language'] = 'en'
        
        return dwc_df
    
    def export_dwc_archive(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Export dataset as Darwin Core Archive.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Darwin Core dataset
        output_path : str
            Output file path (without extension)
        """
# Copyright (c) 2025 Mohamed Z. Hatim
        occurrence_file = f"{output_path}_occurrence.csv"
        df.to_csv(occurrence_file, index=False)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        meta_content = self._generate_meta_xml(df.columns)
        with open(f"{output_path}_meta.xml", 'w') as f:
            f.write(meta_content)
        
        print(f"Darwin Core Archive exported to {output_path}_occurrence.csv and {output_path}_meta.xml")
    
    def _generate_meta_xml(self, columns: List[str]) -> str:
        """Generate meta.xml for Darwin Core Archive."""
        meta_template = """<?xml version="1.0" encoding="UTF-8"?>
<archive xmlns="http://rs.tdwg.org/dwc/text/" metadata="eml.xml">
  <core encoding="UTF-8" fieldsTerminatedBy="," linesTerminatedBy="\\n" fieldsEnclosedBy='"' ignoreHeaderLines="1" rowType="http://rs.tdwg.org/dwc/terms/Occurrence">
    <files>
      <location>occurrence.csv</location>
    </files>
{fields}
  </core>
</archive>"""
        
        field_template = '    <field index="{}" term="http://rs.tdwg.org/dwc/terms/{}"/>'
        
        fields = []
        for i, col in enumerate(columns):
            if col in self.dwc_terms:
                fields.append(field_template.format(i, col))
        
        return meta_template.format(fields='\n'.join(fields))