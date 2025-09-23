"""
Tests for VegZ diversity analysis functionality.
"""

import pytest
import pandas as pd
import numpy as np

from VegZ.diversity import DiversityAnalyzer


class TestDiversityAnalyzer:
    """Test diversity analysis functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample species abundance data."""
# Copyright (c) 2025 Mohamed Z. Hatim
        data = pd.DataFrame({
            'Species_A': [10, 5, 0, 8, 0],
            'Species_B': [5, 10, 15, 2, 0],
            'Species_C': [0, 5, 5, 0, 20],
            'Species_D': [2, 0, 10, 5, 5],
            'Species_E': [0, 0, 0, 15, 10]
        }, index=[f'Site_{i}' for i in range(5)])
        return data
    
    @pytest.fixture
    def even_community(self):
        """Create perfectly even community for testing."""
# Copyright (c) 2025 Mohamed Z. Hatim
        data = pd.DataFrame({
            'Species_A': [10, 10, 10],
            'Species_B': [10, 10, 10],
            'Species_C': [10, 10, 10],
            'Species_D': [10, 10, 10]
        }, index=['Site_1', 'Site_2', 'Site_3'])
        return data
    
    @pytest.fixture
    def uneven_community(self):
        """Create uneven community for testing."""
# Copyright (c) 2025 Mohamed Z. Hatim
        data = pd.DataFrame({
            'Species_A': [90, 90, 90],
            'Species_B': [5, 5, 5],
            'Species_C': [3, 3, 3], 
            'Species_D': [2, 2, 2]
        }, index=['Site_1', 'Site_2', 'Site_3'])
        return data
    
    def test_diversity_analyzer_initialization(self):
        """Test DiversityAnalyzer initialization."""
        analyzer = DiversityAnalyzer()
        
        assert hasattr(analyzer, 'available_indices')
        assert 'shannon' in analyzer.available_indices
        assert 'simpson' in analyzer.available_indices
        assert 'richness' in analyzer.available_indices
        assert 'chao1' in analyzer.available_indices
    
    def test_shannon_diversity(self, sample_data):
        """Test Shannon diversity calculation."""
        analyzer = DiversityAnalyzer()
        shannon = analyzer.shannon_diversity(sample_data)
        
        assert isinstance(shannon, pd.Series)
        assert len(shannon) == len(sample_data)
        assert all(shannon >= 0)  # Shannon is always non-negative
        
# Copyright (c) 2025 Mohamed Z. Hatim
        zero_site = sample_data.iloc[0:1] * 0
        shannon_zero = analyzer.shannon_diversity(zero_site)
        assert shannon_zero.iloc[0] == 0
    
    def test_simpson_diversity(self, sample_data):
        """Test Simpson diversity calculation."""
        analyzer = DiversityAnalyzer()
        simpson = analyzer.simpson_diversity(sample_data)
        
        assert isinstance(simpson, pd.Series)
        assert len(simpson) == len(sample_data)
        assert all(simpson >= 0)  # Simpson is always non-negative
        assert all(simpson <= 1)  # Simpson is always <= 1
    
    def test_species_richness(self, sample_data):
        """Test species richness calculation."""
        analyzer = DiversityAnalyzer()
        richness = analyzer.species_richness(sample_data)
        
        assert isinstance(richness, pd.Series)
        assert len(richness) == len(sample_data)
        assert all(richness >= 0)
        assert all(richness <= len(sample_data.columns))
        
# Copyright (c) 2025 Mohamed Z. Hatim
        expected_richness = (sample_data > 0).sum(axis=1)
        pd.testing.assert_series_equal(richness, expected_richness)
    
    def test_pielou_evenness(self, even_community, uneven_community):
        """Test Pielou's evenness calculation."""
        analyzer = DiversityAnalyzer()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        evenness_even = analyzer.pielou_evenness(even_community)
        evenness_uneven = analyzer.pielou_evenness(uneven_community)
        
        assert isinstance(evenness_even, pd.Series)
        assert isinstance(evenness_uneven, pd.Series)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        assert all(evenness_even > 0.9)
# Copyright (c) 2025 Mohamed Z. Hatim
        assert all(evenness_uneven < evenness_even.min())
    
    def test_fisher_alpha(self, sample_data):
        """Test Fisher's alpha calculation."""
        analyzer = DiversityAnalyzer()
        fisher = analyzer.fisher_alpha(sample_data)
        
        assert isinstance(fisher, pd.Series)
        assert len(fisher) == len(sample_data)
        assert all(fisher >= 0)
    
    def test_chao1_estimator(self, sample_data):
        """Test Chao1 richness estimator."""
        analyzer = DiversityAnalyzer()
        chao1 = analyzer.chao1_estimator(sample_data)
        
        assert isinstance(chao1, pd.Series)
        assert len(chao1) == len(sample_data)
        assert all(chao1 >= 0)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        observed_richness = analyzer.species_richness(sample_data)
        assert all(chao1 >= observed_richness)
    
    def test_calculate_all_indices(self, sample_data):
        """Test calculation of all available indices."""
        analyzer = DiversityAnalyzer()
        all_diversity = analyzer.calculate_all_indices(sample_data)
        
        assert isinstance(all_diversity, pd.DataFrame)
        assert len(all_diversity) == len(sample_data)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        expected_indices = ['shannon', 'simpson', 'richness', 'evenness']
        for index in expected_indices:
            assert index in all_diversity.columns
        
# Copyright (c) 2025 Mohamed Z. Hatim
        assert not all_diversity[expected_indices].isnull().any().any()
    
    def test_hill_numbers(self, sample_data):
        """Test Hill numbers calculation."""
        analyzer = DiversityAnalyzer()
        q_values = [0, 0.5, 1, 1.5, 2]
        hill_numbers = analyzer.hill_numbers(sample_data, q_values)
        
        assert isinstance(hill_numbers, pd.DataFrame)
        assert len(hill_numbers) == len(sample_data)
        assert len(hill_numbers.columns) == len(q_values)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        expected_cols = [f'Hill_q{q}' for q in q_values]
        assert list(hill_numbers.columns) == expected_cols
        
# Copyright (c) 2025 Mohamed Z. Hatim
        assert all(hill_numbers.min() >= 0)
        
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
        for idx in hill_numbers.index:
            row = hill_numbers.loc[idx]
# Copyright (c) 2025 Mohamed Z. Hatim
            decreasing = all(row.iloc[i] >= row.iloc[i+1] - 1e-10 for i in range(len(row)-1))
            assert decreasing, f"Hill numbers should be decreasing for site {idx}"
    
    def test_beta_diversity(self, sample_data):
        """Test beta diversity calculations."""
        analyzer = DiversityAnalyzer()
        
        methods = ['whittaker', 'sorensen', 'jaccard']
        for method in methods:
            beta = analyzer.beta_diversity(sample_data, method=method)
            
            assert isinstance(beta, pd.DataFrame)
            assert beta.shape[0] == len(sample_data)
            assert beta.shape[1] == len(sample_data)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            np.testing.assert_array_almost_equal(beta.values, beta.values.T, decimal=10)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            assert all(np.diag(beta) == 0)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            assert all(beta.min() >= 0)
    
    def test_rarefaction_curve(self, sample_data):
        """Test rarefaction curve calculation."""
        analyzer = DiversityAnalyzer()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        subset_data = sample_data.iloc[:3]  # First 3 sites
        
        rarefaction = analyzer.rarefaction_curve(
            subset_data, 
            sample_sizes=[5, 10, 15, 20]
        )
        
        assert isinstance(rarefaction, pd.DataFrame)
        assert 'sample_id' in rarefaction.columns
        assert 'sample_size' in rarefaction.columns
        assert 'expected_species' in rarefaction.columns
        
# Copyright (c) 2025 Mohamed Z. Hatim
        for sample_id in rarefaction['sample_id'].unique():
            site_data = rarefaction[rarefaction['sample_id'] == sample_id]
            if len(site_data) > 1:
# Copyright (c) 2025 Mohamed Z. Hatim
                assert site_data['expected_species'].iloc[-1] >= site_data['expected_species'].iloc[0]
    
    def test_diversity_with_empty_sites(self):
        """Test diversity calculation with sites containing no species."""
# Copyright (c) 2025 Mohamed Z. Hatim
        data = pd.DataFrame({
            'Species_A': [0, 0, 5, 10],
            'Species_B': [0, 0, 10, 5],
            'Species_C': [0, 0, 0, 15]
        }, index=[f'Site_{i}' for i in range(4)])
        
        analyzer = DiversityAnalyzer()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        shannon = analyzer.shannon_diversity(data)
        simpson = analyzer.simpson_diversity(data)
        richness = analyzer.species_richness(data)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        assert shannon.iloc[0] == 0
        assert shannon.iloc[1] == 0
        assert simpson.iloc[0] == 0
        assert simpson.iloc[1] == 0
        assert richness.iloc[0] == 0
        assert richness.iloc[1] == 0
    
    def test_calculate_index_method(self, sample_data):
        """Test the generic calculate_index method."""
        analyzer = DiversityAnalyzer()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        indices_to_test = ['shannon', 'simpson', 'richness', 'evenness', 'chao1']
        
        for index_name in indices_to_test:
            result = analyzer.calculate_index(sample_data, index_name)
            
            assert isinstance(result, pd.Series)
            assert len(result) == len(sample_data)
            assert not result.isnull().any(), f"Index {index_name} should not have null values"
        
# Copyright (c) 2025 Mohamed Z. Hatim
        with pytest.raises(ValueError, match="Unknown diversity index"):
            analyzer.calculate_index(sample_data, 'invalid_index')


if __name__ == '__main__':
    pytest.main([__file__])