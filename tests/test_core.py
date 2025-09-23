"""
Tests for VegZ core functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from VegZ import VegZ, quick_diversity_analysis, quick_elbow_analysis


class TestVegZCore:
    """Test core VegZ functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample vegetation data for testing."""
# Copyright (c) 2025 Mohamed Z. Hatim
        np.random.seed(42)
        n_sites = 30
        n_species = 15
        
# Copyright (c) 2025 Mohamed Z. Hatim
        group1 = np.random.poisson(5, (8, n_species)) * np.random.binomial(1, 0.4, (8, n_species))
# Copyright (c) 2025 Mohamed Z. Hatim
        group2 = np.random.poisson(3, (7, n_species)) * np.random.binomial(1, 0.3, (7, n_species))
# Copyright (c) 2025 Mohamed Z. Hatim
        group3 = np.random.poisson(8, (8, n_species)) * np.random.binomial(1, 0.2, (8, n_species))
# Copyright (c) 2025 Mohamed Z. Hatim
        group4 = np.random.poisson(2, (7, n_species)) * np.random.binomial(1, 0.5, (7, n_species))
        
        data = np.vstack([group1, group2, group3, group4])
        
        species_names = [f'Species_{i+1:02d}' for i in range(n_species)]
        site_names = [f'SITE_{i+1:03d}' for i in range(n_sites)]
        
        return pd.DataFrame(data, index=site_names, columns=species_names)
    
    def test_vegz_initialization(self):
        """Test VegZ initialization."""
        veg = VegZ()
        assert veg.data is None
        assert veg.species_matrix is None
        assert veg.environmental_data is None
        assert isinstance(veg.metadata, dict)
    
    def test_species_matrix_assignment(self, sample_data):
        """Test species matrix assignment."""
        veg = VegZ()
        veg.species_matrix = sample_data
        
        assert veg.species_matrix is not None
        assert veg.species_matrix.shape == sample_data.shape
        assert list(veg.species_matrix.columns) == list(sample_data.columns)
    
    def test_diversity_calculation(self, sample_data):
        """Test diversity indices calculation."""
        veg = VegZ()
        veg.species_matrix = sample_data
        
        diversity = veg.calculate_diversity(['shannon', 'simpson', 'richness'])
        
        assert isinstance(diversity, pd.DataFrame)
        assert 'shannon' in diversity.columns
        assert 'simpson' in diversity.columns
        assert 'richness' in diversity.columns
        assert len(diversity) == len(sample_data)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        assert all(diversity['shannon'] >= 0)
        assert all(diversity['simpson'] >= 0)
        assert all(diversity['simpson'] <= 1)
        assert all(diversity['richness'] >= 0)
    
    def test_pca_analysis(self, sample_data):
        """Test PCA analysis."""
        veg = VegZ()
        veg.species_matrix = sample_data
        
        pca_results = veg.pca_analysis(transform='hellinger', n_components=4)
        
        assert isinstance(pca_results, dict)
        assert 'scores' in pca_results
        assert 'loadings' in pca_results
        assert 'explained_variance_ratio' in pca_results
        
        assert pca_results['scores'].shape[0] == len(sample_data)
        assert pca_results['scores'].shape[1] == 4
        assert len(pca_results['explained_variance_ratio']) == 4
    
    def test_nmds_analysis(self, sample_data):
        """Test NMDS analysis."""
        veg = VegZ()
        veg.species_matrix = sample_data
        
        nmds_results = veg.nmds_analysis(distance_metric='bray_curtis', n_dimensions=2)
        
        assert isinstance(nmds_results, dict)
        assert 'scores' in nmds_results
        assert 'stress' in nmds_results
        
        assert nmds_results['scores'].shape[0] == len(sample_data)
        assert nmds_results['scores'].shape[1] == 2
        assert isinstance(nmds_results['stress'], float)
    
    def test_hierarchical_clustering(self, sample_data):
        """Test hierarchical clustering."""
        veg = VegZ()
        veg.species_matrix = sample_data
        
        hier_results = veg.hierarchical_clustering(n_clusters=4)
        
        assert isinstance(hier_results, dict)
        assert 'linkage_matrix' in hier_results
        assert 'cluster_labels' in hier_results
        
        assert len(hier_results['cluster_labels']) == len(sample_data)
        assert hier_results['cluster_labels'].max() <= 4
    
    def test_kmeans_clustering(self, sample_data):
        """Test k-means clustering."""
        veg = VegZ()
        veg.species_matrix = sample_data
        
        kmeans_results = veg.kmeans_clustering(n_clusters=3)
        
        assert isinstance(kmeans_results, dict)
        assert 'cluster_labels' in kmeans_results
        assert 'centroids' in kmeans_results
        assert 'inertia' in kmeans_results
        
        assert len(kmeans_results['cluster_labels']) == len(sample_data)
        assert kmeans_results['cluster_labels'].max() < 3  # 0-indexed
    
    def test_elbow_analysis(self, sample_data):
        """Test elbow analysis functionality."""
        veg = VegZ()
        veg.species_matrix = sample_data
        
        elbow_results = veg.elbow_analysis(
            k_range=range(1, 8),
            methods=['knee_locator', 'derivative'],
            plot_results=False
        )
        
        assert isinstance(elbow_results, dict)
        assert 'elbow_points' in elbow_results
        assert 'recommendations' in elbow_results
        assert 'metrics' in elbow_results
        
# Copyright (c) 2025 Mohamed Z. Hatim
        assert 'knee_locator' in elbow_results['elbow_points']
        assert 'derivative' in elbow_results['elbow_points']
        
# Copyright (c) 2025 Mohamed Z. Hatim
        assert 'consensus' in elbow_results['recommendations']
    
    def test_quick_elbow_analysis(self, sample_data):
        """Test quick elbow analysis."""
        veg = VegZ()
        veg.species_matrix = sample_data
        
        optimal_k = veg.quick_elbow_analysis(max_k=8)
        
        assert isinstance(optimal_k, int)
        assert 1 <= optimal_k <= 8
    
    def test_indicator_species_analysis(self, sample_data):
        """Test indicator species analysis."""
        veg = VegZ()
        veg.species_matrix = sample_data
        
# Copyright (c) 2025 Mohamed Z. Hatim
        clusters = veg.kmeans_clustering(n_clusters=3)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        indicators = veg.indicator_species_analysis(clusters['cluster_labels'])
        
        assert isinstance(indicators, pd.DataFrame)
        assert 'species' in indicators.columns
        assert 'cluster' in indicators.columns
        assert 'indicator_value' in indicators.columns
        
# Copyright (c) 2025 Mohamed Z. Hatim
        assert all(indicators['indicator_value'] >= 0)
        assert all(indicators['indicator_value'] <= 100)


class TestQuickFunctions:
    """Test quick analysis functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        data = np.random.poisson(3, (20, 10)) * np.random.binomial(1, 0.3, (20, 10))
        
        species_names = [f'Sp_{i+1}' for i in range(10)]
        site_names = [f'Site_{i+1}' for i in range(20)]
        
        return pd.DataFrame(data, index=site_names, columns=species_names)
    
    def test_quick_diversity_analysis(self, sample_data):
        """Test quick diversity analysis function."""
        diversity = quick_diversity_analysis(sample_data)
        
        assert isinstance(diversity, pd.DataFrame)
        assert len(diversity) == len(sample_data)
        assert 'shannon' in diversity.columns
        assert 'simpson' in diversity.columns
        assert 'richness' in diversity.columns
    
    def test_quick_elbow_analysis(self, sample_data):
        """Test quick elbow analysis function."""
        results = quick_elbow_analysis(sample_data, max_k=6, plot_results=False)
        
        assert isinstance(results, dict)
        assert 'recommendations' in results
        assert 'elbow_points' in results
        assert results['recommendations']['consensus'] is not None


class TestDataTransformations:
    """Test data transformation methods."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with known properties."""
        data = np.array([
            [10, 5, 0, 2],
            [0, 8, 3, 1],
            [5, 0, 7, 4]
        ])
        return pd.DataFrame(data, columns=['A', 'B', 'C', 'D'])
    
    def test_hellinger_transformation(self, sample_data):
        """Test Hellinger transformation."""
        veg = VegZ()
        transformed = veg._transform_data(sample_data, 'hellinger')
        
        assert transformed.shape == sample_data.shape
# Copyright (c) 2025 Mohamed Z. Hatim
        assert np.all(transformed >= 0)
        assert np.all(transformed <= 1)
    
    def test_log_transformation(self, sample_data):
        """Test log transformation."""
        veg = VegZ()
        transformed = veg._transform_data(sample_data, 'log')
        
        assert transformed.shape == sample_data.shape
# Copyright (c) 2025 Mohamed Z. Hatim
        assert np.all(np.isfinite(transformed))
    
    def test_sqrt_transformation(self, sample_data):
        """Test sqrt transformation."""
        veg = VegZ()
        transformed = veg._transform_data(sample_data, 'sqrt')
        
        assert transformed.shape == sample_data.shape
        assert np.all(transformed >= 0)  # Square root is always non-negative


if __name__ == '__main__':
    pytest.main([__file__])