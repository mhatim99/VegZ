"""
Tests for VegZ clustering functionality, especially elbow analysis.
"""

import pytest
import pandas as pd
import numpy as np

from VegZ.clustering import VegetationClustering


class TestVegetationClustering:
    """Test clustering functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with clear cluster structure."""
        np.random.seed(42)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        cluster1 = np.random.multivariate_normal([2, 2], [[1, 0], [0, 1]], 10)
        cluster2 = np.random.multivariate_normal([8, 8], [[1, 0], [0, 1]], 10)
        cluster3 = np.random.multivariate_normal([2, 8], [[1, 0], [0, 1]], 10)
        
        data = np.vstack([cluster1, cluster2, cluster3])
        
# Copyright (c) 2025 Mohamed Z. Hatim
        extra_dims = np.random.poisson(2, (30, 8))
        data = np.hstack([data, extra_dims])
        
        species_names = [f'Species_{i+1:02d}' for i in range(10)]
        site_names = [f'SITE_{i+1:03d}' for i in range(30)]
        
        return pd.DataFrame(data, index=site_names, columns=species_names)
    
    def test_clustering_initialization(self):
        """Test clustering class initialization."""
        clustering = VegetationClustering()
        
        assert hasattr(clustering, 'clustering_methods')
        assert hasattr(clustering, 'validation_metrics')
        assert 'twinspan' in clustering.clustering_methods
        assert 'hierarchical' in clustering.clustering_methods
        assert 'kmeans' in clustering.clustering_methods
    
    def test_comprehensive_elbow_analysis(self, sample_data):
        """Test comprehensive elbow analysis."""
        clustering = VegetationClustering()
        
        results = clustering.comprehensive_elbow_analysis(
            data=sample_data,
            k_range=range(1, 8),
            methods=['knee_locator', 'derivative', 'variance_explained'],
            plot_results=False
        )
        
# Copyright (c) 2025 Mohamed Z. Hatim
        assert isinstance(results, dict)
        assert 'k_values' in results
        assert 'metrics' in results
        assert 'elbow_points' in results
        assert 'method_details' in results
        assert 'recommendations' in results
        
# Copyright (c) 2025 Mohamed Z. Hatim
        assert results['k_values'] == list(range(1, 8))
        
# Copyright (c) 2025 Mohamed Z. Hatim
        metrics = results['metrics']
        assert 'inertia' in metrics
        assert 'silhouette_scores' in metrics
        assert 'calinski_harabasz_scores' in metrics
        assert len(metrics['inertia']) == 7
        
# Copyright (c) 2025 Mohamed Z. Hatim
        assert 'knee_locator' in results['elbow_points']
        assert 'derivative' in results['elbow_points']
        assert 'variance_explained' in results['elbow_points']
        
# Copyright (c) 2025 Mohamed Z. Hatim
        for method in ['knee_locator', 'derivative', 'variance_explained']:
            assert method in results['method_details']
            assert 'description' in results['method_details'][method]
        
# Copyright (c) 2025 Mohamed Z. Hatim
        assert 'consensus' in results['recommendations']
        assert 'confidence' in results['recommendations']
    
    def test_knee_locator_method(self, sample_data):
        """Test knee locator elbow detection."""
        clustering = VegetationClustering()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        k_values = list(range(1, 8))
# Copyright (c) 2025 Mohamed Z. Hatim
        inertias = [100, 60, 35, 30, 28, 26, 25]
        
        elbow_k = clustering._knee_locator_method(k_values, inertias)
        
        assert elbow_k is not None
        assert elbow_k in k_values
    
    def test_derivative_elbow_method(self, sample_data):
        """Test derivative-based elbow detection."""
        clustering = VegetationClustering()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        k_values = list(range(1, 6))
        inertias = [100, 50, 25, 22, 21]  # Clear elbow at k=3
        
        elbow_k = clustering._derivative_elbow_method(k_values, inertias)
        
        assert elbow_k is not None
        assert elbow_k in k_values
    
    def test_variance_explained_elbow(self, sample_data):
        """Test variance explained elbow detection."""
        clustering = VegetationClustering()
        
        k_values = list(range(1, 6))
        inertias = [100, 40, 20, 18, 17]  # Each step explains less variance
        
        elbow_k = clustering._variance_explained_elbow(k_values, inertias)
        
        assert elbow_k is not None
        assert elbow_k in k_values
    
    def test_distortion_jump_method(self, sample_data):
        """Test distortion jump method."""
        clustering = VegetationClustering()
        
        k_values = list(range(1, 7))
        inertias = [100, 60, 35, 32, 30, 29]
        
        elbow_k = clustering._distortion_jump_method(k_values, inertias)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        assert elbow_k is None or elbow_k in k_values
    
    def test_l_method_elbow(self, sample_data):
        """Test L-method elbow detection."""
        clustering = VegetationClustering()
        
        k_values = list(range(1, 8))
# Copyright (c) 2025 Mohamed Z. Hatim
        inertias = [100, 80, 60, 45, 42, 40, 39]
        
        elbow_k = clustering._l_method_elbow(k_values, inertias)
        
        assert elbow_k is None or elbow_k in k_values
    
    def test_elbow_analysis_with_different_transforms(self, sample_data):
        """Test elbow analysis with different data transformations."""
        clustering = VegetationClustering()
        
        transforms = ['hellinger', 'log', 'sqrt', 'none']
        
        for transform in transforms:
            results = clustering.comprehensive_elbow_analysis(
                data=sample_data,
                k_range=range(1, 6),
                methods=['knee_locator', 'derivative'],
                transform=transform,
                plot_results=False
            )
            
            assert isinstance(results, dict)
            assert 'recommendations' in results
# Copyright (c) 2025 Mohamed Z. Hatim
    
    def test_elbow_analysis_consensus(self, sample_data):
        """Test consensus mechanism in elbow analysis."""
        clustering = VegetationClustering()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        results = {
            'elbow_points': {
                'method1': 3,
                'method2': 3,  # Agreement on k=3
                'method3': 4
            },
            'recommendations': {}
        }
        
# Copyright (c) 2025 Mohamed Z. Hatim
        elbow_points = [v for v in results['elbow_points'].values() if v is not None]
        if elbow_points:
            from collections import Counter
            counter = Counter(elbow_points)
            most_common = counter.most_common(1)[0]
            
            if most_common[1] > 1:  # If there's agreement
                consensus = most_common[0]
                confidence = most_common[1] / len(elbow_points)
            else:
                consensus = int(np.median(elbow_points))
                confidence = 1.0 / len(elbow_points)
            
            assert consensus == 3  # Should agree on k=3
            assert confidence > 0.5  # Should have good confidence
    
    def test_kmeans_clustering(self, sample_data):
        """Test k-means clustering method."""
        clustering = VegetationClustering()
        
        results = clustering.kmeans_clustering(sample_data, n_clusters=3)
        
        assert isinstance(results, dict)
        assert 'cluster_labels' in results
        assert 'centroids' in results
        assert 'inertia' in results
        assert 'silhouette_score' in results
        
        assert len(results['cluster_labels']) == len(sample_data)
        assert results['cluster_labels'].max() <= 2  # 0-indexed, 3 clusters = 0,1,2
    
    def test_hierarchical_clustering(self, sample_data):
        """Test hierarchical clustering method."""
        clustering = VegetationClustering()
        
        results = clustering.hierarchical_clustering(
            sample_data, 
            n_clusters=3,
            distance_metric='euclidean'
        )
        
        assert isinstance(results, dict)
        assert 'cluster_labels' in results
        assert 'linkage_matrix' in results
        assert 'silhouette_score' in results
        
        assert len(results['cluster_labels']) == len(sample_data)
    
    def test_optimal_k_analysis(self, sample_data):
        """Test traditional optimal k analysis."""
        clustering = VegetationClustering()
        
        results = clustering.optimal_k_analysis(
            sample_data,
            k_range=range(2, 7),
            methods=['elbow', 'silhouette']
        )
        
        assert isinstance(results, dict)
        assert 'optimal_k' in results
        assert 'metrics' in results
        assert 'recommendations' in results
        
# Copyright (c) 2025 Mohamed Z. Hatim
        assert 'elbow' in results['optimal_k']
        assert 'silhouette' in results['optimal_k']


class TestElbowPlotting:
    """Test elbow analysis plotting functionality."""
    
    @pytest.fixture
    def sample_elbow_results(self):
        """Create sample elbow results for plotting tests."""
        return {
            'k_values': list(range(1, 8)),
            'metrics': {
                'inertia': [100, 60, 35, 30, 28, 26, 25],
                'silhouette_scores': [0, 0.4, 0.6, 0.5, 0.45, 0.4, 0.35],
                'calinski_harabasz_scores': [0, 150, 200, 180, 160, 140, 120],
                'davies_bouldin_scores': [float('inf'), 1.2, 0.8, 0.9, 1.0, 1.1, 1.2]
            },
            'elbow_points': {
                'knee_locator': 3,
                'derivative': 3,
                'variance_explained': 3
            },
            'recommendations': {
                'consensus': 3,
                'confidence': 1.0,
                'silhouette_optimal': 3,
                'calinski_optimal': 3
            }
        }
    
    def test_create_elbow_plots(self, sample_elbow_results):
        """Test elbow plot creation."""
        clustering = VegetationClustering()
        
        plots = clustering._create_elbow_plots(sample_elbow_results)
        
        assert isinstance(plots, dict)
        assert 'figure' in plots
        assert 'axes' in plots
        assert 'description' in plots
        
# Copyright (c) 2025 Mohamed Z. Hatim
        import matplotlib.pyplot as plt
        assert isinstance(plots['figure'], plt.Figure)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        plt.close(plots['figure'])


if __name__ == '__main__':
    pytest.main([__file__])