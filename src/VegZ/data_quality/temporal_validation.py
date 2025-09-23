"""
Temporal data validation module - comprehensive date and time validation.

Copyright (c) 2025 Mohamed Z. Hatim
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import warnings
import re


class TemporalValidator:
    """Comprehensive temporal data validation for biodiversity records."""
    
    def __init__(self):
        """Initialize temporal validator."""
        self.suspicious_patterns = {
            'default_dates': [
                '1900-01-01', '2000-01-01', '1999-12-31',
                '1970-01-01', '1900-12-31'
            ],
            'suspicious_days': [1, 15, 31],  # Common default days
            'future_threshold': datetime.now() + timedelta(days=30)  # Future dates threshold
        }
        
# Copyright (c) 2025 Mohamed Z. Hatim
        self.date_patterns = [
            r'\d{4}-\d{2}-\d{2}',      # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',      # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',      # MM-DD-YYYY
            r'\d{4}/\d{2}/\d{2}',      # YYYY/MM/DD
            r'\d{1,2}/\d{1,2}/\d{4}',  # M/D/YYYY
            r'\d{4}\d{2}\d{2}',        # YYYYMMDD
        ]
    
    def validate_dates(self, df: pd.DataFrame,
                      date_cols: Union[str, List[str]],
                      event_date_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive date validation.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with date columns
        date_cols : str or list
            Date column name(s) to validate
        event_date_col : str, optional
            Main event date column for cross-validation
            
        Returns:
        --------
        dict
            Validation results and flags
        """
        if isinstance(date_cols, str):
            date_cols = [date_cols]
        
        results = {
            'total_records': len(df),
            'date_columns_analyzed': date_cols,
            'flags': {},
            'valid_dates': {},
            'issues_found': []
        }
        
# Copyright (c) 2025 Mohamed Z. Hatim
        flags_df = pd.DataFrame(index=df.index)
        
        for date_col in date_cols:
            if date_col not in df.columns:
                results['issues_found'].append(f"Date column '{date_col}' not found")
                continue
            
            col_results = self._validate_single_date_column(df, date_col)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            for flag_type, flag_data in col_results['flags'].items():
                flag_col_name = f"{date_col}_{flag_type}"
                flags_df[flag_col_name] = flag_data
                results['flags'][flag_col_name] = flag_data.sum()
            
            results['valid_dates'][date_col] = col_results['valid_dates']
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if len(date_cols) > 1:
            cross_validation = self._cross_validate_dates(df, date_cols)
            results['cross_validation'] = cross_validation
            
# Copyright (c) 2025 Mohamed Z. Hatim
            for flag_type, flag_data in cross_validation['flags'].items():
                flags_df[flag_type] = flag_data
                results['flags'][flag_type] = flag_data.sum()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if event_date_col and event_date_col in df.columns:
            consistency_results = self._check_temporal_consistency(df, event_date_col, date_cols)
            results['temporal_consistency'] = consistency_results
        
# Copyright (c) 2025 Mohamed Z. Hatim
        seasonal_validation = self._validate_seasonal_patterns(df, date_cols)
        results['seasonal_validation'] = seasonal_validation
        
        results['flags_dataframe'] = flags_df
        
        return results
    
    def _validate_single_date_column(self, df: pd.DataFrame, 
                                   date_col: str) -> Dict[str, Any]:
        """Validate a single date column."""
        results = {
            'flags': {},
            'valid_dates': 0,
            'parsed_dates': None
        }
        
# Copyright (c) 2025 Mohamed Z. Hatim
        parsed_dates, parse_success = self._parse_dates_robust(df[date_col])
        results['parsed_dates'] = parsed_dates
        
# Copyright (c) 2025 Mohamed Z. Hatim
        unparseable = ~parse_success
        results['flags']['unparseable'] = unparseable
        
# Copyright (c) 2025 Mohamed Z. Hatim
        missing = df[date_col].isna()
        results['flags']['missing'] = missing
        
# Copyright (c) 2025 Mohamed Z. Hatim
        valid_mask = parse_success & ~missing
        
        if valid_mask.any():
            valid_dates = parsed_dates[valid_mask]
            
# Copyright (c) 2025 Mohamed Z. Hatim
            suspicious_defaults = self._detect_suspicious_default_dates(parsed_dates)
            results['flags']['suspicious_defaults'] = suspicious_defaults
            
# Copyright (c) 2025 Mohamed Z. Hatim
            future_dates = self._detect_future_dates(parsed_dates)
            results['flags']['future_dates'] = future_dates
            
# Copyright (c) 2025 Mohamed Z. Hatim
            very_old = self._detect_very_old_dates(parsed_dates)
            results['flags']['very_old'] = very_old
            
# Copyright (c) 2025 Mohamed Z. Hatim
            impossible = self._detect_impossible_dates(df[date_col], parsed_dates, parse_success)
            results['flags']['impossible'] = impossible
            
# Copyright (c) 2025 Mohamed Z. Hatim
            suspicious_patterns = self._detect_suspicious_temporal_patterns(parsed_dates)
            results['flags']['suspicious_patterns'] = suspicious_patterns
            
# Copyright (c) 2025 Mohamed Z. Hatim
            all_flags = (unparseable | missing | suspicious_defaults | 
                        future_dates | very_old | impossible | suspicious_patterns)
            results['valid_dates'] = (~all_flags).sum()
        else:
# Copyright (c) 2025 Mohamed Z. Hatim
            for flag_type in ['suspicious_defaults', 'future_dates', 'very_old', 
                             'impossible', 'suspicious_patterns']:
                results['flags'][flag_type] = pd.Series(False, index=df.index)
        
        return results
    
    def _parse_dates_robust(self, date_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Robustly parse dates trying multiple formats."""
        parsed_dates = pd.Series(pd.NaT, index=date_series.index, dtype='datetime64[ns]')
        parse_success = pd.Series(False, index=date_series.index)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        try:
            default_parsed = pd.to_datetime(date_series, errors='coerce', infer_datetime_format=True)
            success_mask = ~default_parsed.isna()
            parsed_dates[success_mask] = default_parsed[success_mask]
            parse_success[success_mask] = True
        except:
            pass
        
# Copyright (c) 2025 Mohamed Z. Hatim
        remaining_mask = ~parse_success & date_series.notna()
        
        if remaining_mask.any():
            date_formats = [
                '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d',
                '%m-%d-%Y', '%d-%m-%Y', '%Y%m%d', '%m%d%Y',
                '%B %d, %Y', '%b %d, %Y', '%d %B %Y', '%d %b %Y'
            ]
            
            for date_format in date_formats:
                still_remaining = remaining_mask & ~parse_success
                if not still_remaining.any():
                    break
                
                try:
                    format_parsed = pd.to_datetime(
                        date_series[still_remaining], 
                        format=date_format, 
                        errors='coerce'
                    )
                    format_success = ~format_parsed.isna()
                    
                    if format_success.any():
                        success_indices = still_remaining[still_remaining].index[format_success]
                        parsed_dates.loc[success_indices] = format_parsed[format_success]
                        parse_success.loc[success_indices] = True
                        
                except:
                    continue
        
        return parsed_dates, parse_success
    
    def _detect_suspicious_default_dates(self, parsed_dates: pd.Series) -> pd.Series:
        """Detect suspicious default dates."""
        suspicious = pd.Series(False, index=parsed_dates.index)
        
        for default_date in self.suspicious_patterns['default_dates']:
            default_dt = pd.to_datetime(default_date)
            suspicious |= (parsed_dates == default_dt)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        suspicious_days = parsed_dates.dt.day.isin(self.suspicious_patterns['suspicious_days'])
        
# Copyright (c) 2025 Mohamed Z. Hatim
        jan_first = (parsed_dates.dt.month == 1) & (parsed_dates.dt.day == 1)
        
        return suspicious | suspicious_days | jan_first
    
    def _detect_future_dates(self, parsed_dates: pd.Series) -> pd.Series:
        """Detect dates in the future."""
        return parsed_dates > self.suspicious_patterns['future_threshold']
    
    def _detect_very_old_dates(self, parsed_dates: pd.Series, 
                              min_year: int = 1800) -> pd.Series:
        """Detect unrealistically old dates."""
        return parsed_dates < pd.to_datetime(f'{min_year}-01-01')
    
    def _detect_impossible_dates(self, original_series: pd.Series,
                                parsed_dates: pd.Series,
                                parse_success: pd.Series) -> pd.Series:
        """Detect dates that look valid but are impossible."""
        impossible = pd.Series(False, index=parsed_dates.index)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        for idx, (orig, parsed, success) in enumerate(zip(original_series, parsed_dates, parse_success)):
            if not success or pd.isna(parsed) or pd.isna(orig):
                continue
            
            orig_str = str(orig)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            if re.search(r'(02|2)[-/](3[01]|29)', orig_str):  # Feb 30, 31
                impossible.iloc[idx] = True
            elif re.search(r'(04|4|06|6|09|9|11)[-/](31)', orig_str):  # 30-day months with 31
                impossible.iloc[idx] = True
        
        return impossible
    
    def _detect_suspicious_temporal_patterns(self, parsed_dates: pd.Series) -> pd.Series:
        """Detect suspicious temporal patterns."""
        suspicious = pd.Series(False, index=parsed_dates.index)
        
        valid_dates = parsed_dates.dropna()
        
        if len(valid_dates) == 0:
            return suspicious
        
# Copyright (c) 2025 Mohamed Z. Hatim
        date_counts = valid_dates.value_counts()
        common_dates = date_counts[date_counts > max(2, len(valid_dates) * 0.1)]
        
        if len(common_dates) > 0:
            for common_date in common_dates.index:
                suspicious |= (parsed_dates == common_date)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if len(valid_dates) > 10:
            date_range = (valid_dates.max() - valid_dates.min()).days
            if date_range < 7 and len(valid_dates) > 5:  # All dates within a week
                suspicious[parsed_dates.notna()] = True
        
        return suspicious
    
    def _cross_validate_dates(self, df: pd.DataFrame, 
                            date_cols: List[str]) -> Dict[str, Any]:
        """Cross-validate multiple date columns."""
        results = {
            'flags': {},
            'inconsistencies': []
        }
        
        flags_df = pd.DataFrame(index=df.index)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        parsed_dates = {}
        for col in date_cols:
            if col in df.columns:
                parsed, success = self._parse_dates_robust(df[col])
                parsed_dates[col] = parsed
        
        if len(parsed_dates) < 2:
            return results
        
# Copyright (c) 2025 Mohamed Z. Hatim
        date_cols_list = list(parsed_dates.keys())
        
        for i in range(len(date_cols_list)):
            for j in range(i + 1, len(date_cols_list)):
                col1, col2 = date_cols_list[i], date_cols_list[j]
                
# Copyright (c) 2025 Mohamed Z. Hatim
                inconsistent = self._check_date_logic(
                    parsed_dates[col1], parsed_dates[col2], col1, col2
                )
                
                flag_name = f'inconsistent_{col1}_{col2}'
                flags_df[flag_name] = inconsistent
                results['flags'][flag_name] = inconsistent
                
                if inconsistent.any():
                    results['inconsistencies'].append({
                        'columns': [col1, col2],
                        'count': inconsistent.sum(),
                        'description': f'{col1} and {col2} have logical inconsistencies'
                    })
        
        return results
    
    def _check_date_logic(self, dates1: pd.Series, dates2: pd.Series,
                         col1: str, col2: str) -> pd.Series:
        """Check logical consistency between two date columns."""
        inconsistent = pd.Series(False, index=dates1.index)
        
        valid_mask = dates1.notna() & dates2.notna()
        
        if not valid_mask.any():
            return inconsistent
        
# Copyright (c) 2025 Mohamed Z. Hatim
        relationships = self._infer_date_relationships(col1, col2)
        
        for relationship in relationships:
            if relationship == 'before':
# Copyright (c) 2025 Mohamed Z. Hatim
                inconsistent[valid_mask] |= (dates1[valid_mask] > dates2[valid_mask])
            elif relationship == 'after':
# Copyright (c) 2025 Mohamed Z. Hatim
                inconsistent[valid_mask] |= (dates1[valid_mask] < dates2[valid_mask])
            elif relationship == 'same_year':
# Copyright (c) 2025 Mohamed Z. Hatim
                inconsistent[valid_mask] |= (
                    dates1[valid_mask].dt.year != dates2[valid_mask].dt.year
                )
        
        return inconsistent
    
    def _infer_date_relationships(self, col1: str, col2: str) -> List[str]:
        """Infer logical relationships between date columns."""
        relationships = []
        
        col1_lower = col1.lower()
        col2_lower = col2.lower()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        before_after_patterns = [
            ('start', 'end'), ('begin', 'end'), ('first', 'last'),
            ('birth', 'death'), ('collection', 'identification'),
            ('field', 'lab'), ('observed', 'recorded')
        ]
        
        for before_term, after_term in before_after_patterns:
            if before_term in col1_lower and after_term in col2_lower:
                relationships.append('before')
            elif after_term in col1_lower and before_term in col2_lower:
                relationships.append('after')
        
# Copyright (c) 2025 Mohamed Z. Hatim
        same_year_patterns = ['year', 'annual', 'season']
        if any(pattern in col1_lower and pattern in col2_lower for pattern in same_year_patterns):
            relationships.append('same_year')
        
        return relationships
    
    def _check_temporal_consistency(self, df: pd.DataFrame,
                                   event_date_col: str,
                                   other_date_cols: List[str]) -> Dict[str, Any]:
        """Check temporal consistency with main event date."""
        results = {
            'main_event_column': event_date_col,
            'consistency_checks': {},
            'flags': {}
        }
        
        event_dates, event_success = self._parse_dates_robust(df[event_date_col])
        
        for date_col in other_date_cols:
            if date_col == event_date_col or date_col not in df.columns:
                continue
            
            other_dates, other_success = self._parse_dates_robust(df[date_col])
            
# Copyright (c) 2025 Mohamed Z. Hatim
            valid_mask = event_success & other_success
            
            if valid_mask.any():
# Copyright (c) 2025 Mohamed Z. Hatim
                time_diff = (other_dates - event_dates)[valid_mask]
                
# Copyright (c) 2025 Mohamed Z. Hatim
                extreme_diff = abs(time_diff) > timedelta(days=365)
                
                flag_series = pd.Series(False, index=df.index)
                flag_series[valid_mask] = extreme_diff
                
                results['flags'][f'{date_col}_extreme_diff'] = flag_series
                results['consistency_checks'][date_col] = {
                    'mean_difference_days': time_diff.dt.days.mean(),
                    'extreme_differences': extreme_diff.sum(),
                    'max_difference_days': abs(time_diff).dt.days.max()
                }
        
        return results
    
    def _validate_seasonal_patterns(self, df: pd.DataFrame,
                                   date_cols: List[str]) -> Dict[str, Any]:
        """Validate seasonal and phenological patterns."""
        results = {
            'seasonal_distribution': {},
            'phenological_flags': {},
            'monthly_patterns': {}
        }
        
        for date_col in date_cols:
            if date_col not in df.columns:
                continue
            
            parsed_dates, success = self._parse_dates_robust(df[date_col])
            valid_dates = parsed_dates[success]
            
            if len(valid_dates) == 0:
                continue
            
# Copyright (c) 2025 Mohamed Z. Hatim
            monthly_dist = valid_dates.dt.month.value_counts().sort_index()
            results['monthly_patterns'][date_col] = monthly_dist.to_dict()
            
# Copyright (c) 2025 Mohamed Z. Hatim
            seasons = valid_dates.dt.month.map(self._month_to_season)
            seasonal_dist = seasons.value_counts()
            results['seasonal_distribution'][date_col] = seasonal_dist.to_dict()
            
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
            unusual_seasons = pd.Series(False, index=df.index)
            
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
            winter_months = valid_dates.dt.month.isin([12, 1, 2])
            if 'flower' in date_col.lower() or 'bloom' in date_col.lower():
                unusual_seasons[success] = winter_months
            
            results['phenological_flags'][date_col] = unusual_seasons
        
        return results
    
    def _month_to_season(self, month: int) -> str:
        """Convert month number to season (Northern Hemisphere)."""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Fall'
        else:
            return 'Unknown'
    
    def extract_date_components(self, df: pd.DataFrame,
                              date_cols: Union[str, List[str]]) -> pd.DataFrame:
        """
        Extract date components (year, month, day, day of year, etc.).
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with date columns
        date_cols : str or list
            Date column name(s)
            
        Returns:
        --------
        pd.DataFrame
            Dataset with additional date component columns
        """
        if isinstance(date_cols, str):
            date_cols = [date_cols]
        
        result_df = df.copy()
        
        for date_col in date_cols:
            if date_col not in df.columns:
                continue
            
            parsed_dates, success = self._parse_dates_robust(df[date_col])
            
# Copyright (c) 2025 Mohamed Z. Hatim
            result_df[f'{date_col}_year'] = parsed_dates.dt.year
            result_df[f'{date_col}_month'] = parsed_dates.dt.month
            result_df[f'{date_col}_day'] = parsed_dates.dt.day
            result_df[f'{date_col}_day_of_year'] = parsed_dates.dt.dayofyear
            result_df[f'{date_col}_week_of_year'] = parsed_dates.dt.isocalendar().week
            result_df[f'{date_col}_season'] = parsed_dates.dt.month.map(self._month_to_season)
            result_df[f'{date_col}_parsed_successfully'] = success
        
        return result_df
    
    def generate_temporal_quality_report(self, df: pd.DataFrame,
                                       date_cols: Union[str, List[str]],
                                       event_date_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive temporal data quality report.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset to validate
        date_cols : str or list
            Date column name(s) to validate
        event_date_col : str, optional
            Main event date column
            
        Returns:
        --------
        dict
            Comprehensive temporal quality report
        """
        if isinstance(date_cols, str):
            date_cols = [date_cols]
        
        report = {
            'dataset_summary': {
                'total_records': len(df),
                'date_columns': date_cols
            }
        }
        
# Copyright (c) 2025 Mohamed Z. Hatim
        validation_results = self.validate_dates(df, date_cols, event_date_col)
        report['date_validation'] = validation_results
        
# Copyright (c) 2025 Mohamed Z. Hatim
        summary_stats = {}
        for date_col in date_cols:
            if date_col in df.columns:
                parsed_dates, success = self._parse_dates_robust(df[date_col])
                valid_dates = parsed_dates[success]
                
                if len(valid_dates) > 0:
                    summary_stats[date_col] = {
                        'earliest_date': valid_dates.min().strftime('%Y-%m-%d'),
                        'latest_date': valid_dates.max().strftime('%Y-%m-%d'),
                        'date_range_years': (valid_dates.max() - valid_dates.min()).days / 365.25,
                        'valid_date_count': len(valid_dates),
                        'valid_date_percentage': (len(valid_dates) / len(df)) * 100
                    }
        
        report['summary_statistics'] = summary_stats
        
# Copyright (c) 2025 Mohamed Z. Hatim
        recommendations = []
        total_flags = validation_results['flags']
        
        for flag_type, count in total_flags.items():
            if count > 0:
                if 'missing' in flag_type:
                    recommendations.append(f"Fill {count} missing dates in {flag_type.replace('_missing', '')}")
                elif 'unparseable' in flag_type:
                    recommendations.append(f"Fix {count} unparseable dates in {flag_type.replace('_unparseable', '')}")
                elif 'future' in flag_type:
                    recommendations.append(f"Verify {count} future dates in {flag_type.replace('_future_dates', '')}")
                elif 'suspicious' in flag_type:
                    recommendations.append(f"Review {count} suspicious dates in {flag_type}")
        
        report['recommendations'] = recommendations
        
        return report