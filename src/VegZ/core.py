"""
VegZ Core Module - Main functionality for vegetation data analysis.

Copyright (c) 2025 Mohamed Z. Hatim

This module provides the core functionality for vegetation data analysis including:
- Data loading and preprocessing
- Diversity calculations
- Multivariate analysis
- Clustering
- Statistical analysis
- Visualization
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Tuple, Optional, Any
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


class VegZ:
    """Main VegZ class providing comprehensive vegetation analysis tools."""
    
    def __init__(self):
        """Initialize VegZ with default parameters."""
        self.data = None
        self.species_matrix = None
        self.environmental_data = None
        self.metadata = {}
        
    # Copyright (c) 2025 Mohamed Z. Hatim
    # Copyright (c) 2025 Mohamed Z. Hatim
    # Copyright (c) 2025 Mohamed Z. Hatim
    
    def load_data(self, filepath: str, 
                  format_type: str = 'csv',
                  species_cols: Optional[List[str]] = None,
                  **kwargs) -> pd.DataFrame:
        """
        Load vegetation data from various formats.
        
        Parameters:
        -----------
        filepath : str
            Path to data file
        format_type : str
            File format ('csv', 'excel', 'txt')
        species_cols : list, optional
            Column names containing species data
        **kwargs
            Additional parameters for pandas readers
            
        Returns:
        --------
        pd.DataFrame
            Loaded data
        """
        if format_type.lower() == 'csv':
            self.data = pd.read_csv(filepath, **kwargs)
        elif format_type.lower() in ['excel', 'xlsx', 'xls']:
            self.data = pd.read_excel(filepath, **kwargs)
        elif format_type.lower() == 'txt':
            self.data = pd.read_csv(filepath, sep='\t', **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        if species_cols is None:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            # Copyright (c) 2025 Mohamed Z. Hatim
            self.species_matrix = self.data[numeric_cols]
        else:
            self.species_matrix = self.data[species_cols]
        
        return self.data
    
    def standardize_species_names(self, species_column: str = 'species') -> pd.DataFrame:
        """
        Clean and standardize species names.
        
        Parameters:
        -----------
        species_column : str
            Column containing species names
            
        Returns:
        --------
        pd.DataFrame
            Data with standardized species names
        """
        if self.data is None or species_column not in self.data.columns:
            raise ValueError("Data not loaded or species column not found")
        
        def clean_name(name):
            if pd.isna(name):
                return ''
            name = str(name).strip()
            # Copyright (c) 2025 Mohamed Z. Hatim
            words = name.split()
            if len(words) >= 2:
                # Copyright (c) 2025 Mohamed Z. Hatim
                words[0] = words[0].capitalize()
                words[1] = words[1].lower()
                return ' '.join(words[:2])  # Keep only genus and species
            return name
        
        self.data[f'{species_column}_clean'] = self.data[species_column].apply(clean_name)
        return self.data
    
    def filter_rare_species(self, min_occurrences: int = 3, 
                           min_abundance: float = 0.0) -> pd.DataFrame:
        """
        Filter out rare species based on occurrence frequency and abundance.
        
        Parameters:
        -----------
        min_occurrences : int
            Minimum number of sites where species must occur
        min_abundance : float
            Minimum total abundance threshold
            
        Returns:
        --------
        pd.DataFrame
            Filtered species matrix
        """
        if self.species_matrix is None:
            raise ValueError("Species matrix not available")
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        occurrences = (self.species_matrix > 0).sum(axis=0)
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        total_abundance = self.species_matrix.sum(axis=0)
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        keep_species = (occurrences >= min_occurrences) & (total_abundance >= min_abundance)
        
        self.species_matrix = self.species_matrix.loc[:, keep_species]
        
        print(f"Retained {keep_species.sum()} species out of {len(keep_species)} original species")
        
        return self.species_matrix
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    # Copyright (c) 2025 Mohamed Z. Hatim
    # Copyright (c) 2025 Mohamed Z. Hatim
    
    def calculate_diversity(self, indices: List[str] = ['shannon', 'simpson', 'richness']) -> pd.DataFrame:
        """
        Calculate diversity indices.
        
        Parameters:
        -----------
        indices : list
            List of diversity indices to calculate
            
        Returns:
        --------
        pd.DataFrame
            Diversity indices for each sample
        """
        if self.species_matrix is None:
            raise ValueError("Species matrix not available")
        
        results = pd.DataFrame(index=self.species_matrix.index)
        
        for index in indices:
            if index.lower() == 'shannon':
                results['shannon'] = self._shannon_diversity()
            elif index.lower() == 'simpson':
                results['simpson'] = self._simpson_diversity()
            elif index.lower() == 'richness':
                results['richness'] = self._species_richness()
            elif index.lower() == 'evenness':
                results['evenness'] = self._evenness()
            else:
                warnings.warn(f"Unknown diversity index: {index}")
        
        return results
    
    def _shannon_diversity(self) -> pd.Series:
        """Calculate Shannon diversity index."""
        def shannon(row):
            proportions = row[row > 0] / row.sum()
            return -np.sum(proportions * np.log(proportions))
        
        return self.species_matrix.apply(shannon, axis=1)
    
    def _simpson_diversity(self) -> pd.Series:
        """Calculate Simpson diversity index."""
        def simpson(row):
            proportions = row[row > 0] / row.sum()
            return 1 - np.sum(proportions ** 2)
        
        return self.species_matrix.apply(simpson, axis=1)
    
    def _species_richness(self) -> pd.Series:
        """Calculate species richness."""
        return (self.species_matrix > 0).sum(axis=1)
    
    def _evenness(self) -> pd.Series:
        """Calculate Pielou's evenness."""
        shannon = self._shannon_diversity()
        richness = self._species_richness()
        return shannon / np.log(richness.replace(0, 1))
    
    def rarefaction_curve(self, sample_sizes: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate rarefaction curves.
        
        Parameters:
        -----------
        sample_sizes : list, optional
            Sample sizes for rarefaction
            
        Returns:
        --------
        pd.DataFrame
            Rarefaction curves data
        """
        if self.species_matrix is None:
            raise ValueError("Species matrix not available")
        
        if sample_sizes is None:
            max_individuals = int(self.species_matrix.sum(axis=1).max())
            sample_sizes = list(range(1, max_individuals + 1, max(1, max_individuals // 50)))
        
        results = []
        
        for idx, row in self.species_matrix.iterrows():
            species_counts = row[row > 0].astype(int)
            total_individuals = species_counts.sum()
            
            for sample_size in sample_sizes:
                if sample_size <= total_individuals:
                    expected_species = self._rarefaction_expected_species(species_counts, sample_size)
                    results.append({
                        'sample_id': idx,
                        'sample_size': sample_size,
                        'expected_species': expected_species
                    })
        
        return pd.DataFrame(results)
    
    def _rarefaction_expected_species(self, species_counts: pd.Series, sample_size: int) -> float:
        """Calculate expected number of species for rarefaction."""
        total = species_counts.sum()
        expected = 0
        
        for count in species_counts:
            # Copyright (c) 2025 Mohamed Z. Hatim
            prob_not_selected = 1
            for i in range(sample_size):
                prob_not_selected *= (total - count - i) / (total - i)
            
            # Copyright (c) 2025 Mohamed Z. Hatim
            prob_selected = 1 - prob_not_selected
            expected += prob_selected
        
        return expected
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    # Copyright (c) 2025 Mohamed Z. Hatim
    # Copyright (c) 2025 Mohamed Z. Hatim
    
    def pca_analysis(self, transform: str = 'hellinger', 
                     n_components: Optional[int] = None) -> Dict[str, Any]:
        """
        Principal Component Analysis.
        
        Parameters:
        -----------
        transform : str
            Data transformation method
        n_components : int, optional
            Number of components to retain
            
        Returns:
        --------
        dict
            PCA results including scores, loadings, and variance explained
        """
        if self.species_matrix is None:
            raise ValueError("Species matrix not available")
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        transformed_data = self._transform_data(self.species_matrix, transform)
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(transformed_data)
        
        results = {
            'scores': pd.DataFrame(scores, 
                                 index=self.species_matrix.index,
                                 columns=[f'PC{i+1}' for i in range(scores.shape[1])]),
            'loadings': pd.DataFrame(pca.components_.T,
                                   index=self.species_matrix.columns,
                                   columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])]),
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'pca_object': pca
        }
        
        return results
    
    def nmds_analysis(self, distance_metric: str = 'bray_curtis',
                      n_dimensions: int = 2,
                      transform: str = 'hellinger') -> Dict[str, Any]:
        """
        Non-metric Multidimensional Scaling.
        
        Parameters:
        -----------
        distance_metric : str
            Distance metric to use
        n_dimensions : int
            Number of dimensions
        transform : str
            Data transformation method
            
        Returns:
        --------
        dict
            NMDS results
        """
        if self.species_matrix is None:
            raise ValueError("Species matrix not available")
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        transformed_data = self._transform_data(self.species_matrix, transform)
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        if distance_metric == 'bray_curtis':
            distances = self._bray_curtis_distance(transformed_data)
        elif distance_metric == 'euclidean':
            distances = pdist(transformed_data, metric='euclidean')
        else:
            distances = pdist(transformed_data, metric=distance_metric)
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        mds = MDS(n_components=n_dimensions, dissimilarity='precomputed', random_state=42)
        scores = mds.fit_transform(squareform(distances))
        
        results = {
            'scores': pd.DataFrame(scores,
                                 index=self.species_matrix.index,
                                 columns=[f'NMDS{i+1}' for i in range(n_dimensions)]),
            'stress': mds.stress_,
            'distances': distances,
            'mds_object': mds
        }
        
        return results
    
    def _transform_data(self, data: pd.DataFrame, method: str) -> np.ndarray:
        """Apply data transformation."""
        if method == 'hellinger':
            row_sums = data.sum(axis=1)
            row_sums[row_sums == 0] = 1
            proportions = data.div(row_sums, axis=0)
            return np.sqrt(proportions.values)
        elif method == 'log':
            return np.log1p(data.values)
        elif method == 'sqrt':
            return np.sqrt(data.values)
        elif method == 'standardize':
            scaler = StandardScaler()
            return scaler.fit_transform(data.values)
        else:
            return data.values
    
    def _bray_curtis_distance(self, data: np.ndarray) -> np.ndarray:
        """Calculate Bray-Curtis distance."""
        n_samples = data.shape[0]
        distances = []
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                numerator = np.sum(np.abs(data[i] - data[j]))
                denominator = np.sum(data[i] + data[j])
                if denominator == 0:
                    distance = 0
                else:
                    distance = numerator / denominator
                distances.append(distance)
        
        return np.array(distances)
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    # Copyright (c) 2025 Mohamed Z. Hatim
    # Copyright (c) 2025 Mohamed Z. Hatim
    
    def hierarchical_clustering(self, distance_metric: str = 'bray_curtis',
                               linkage_method: str = 'average',
                               n_clusters: Optional[int] = None) -> Dict[str, Any]:
        """
        Hierarchical clustering analysis.
        
        Parameters:
        -----------
        distance_metric : str
            Distance metric for clustering
        linkage_method : str
            Linkage method ('average', 'complete', 'single', 'ward')
        n_clusters : int, optional
            Number of clusters to extract
            
        Returns:
        --------
        dict
            Clustering results
        """
        if self.species_matrix is None:
            raise ValueError("Species matrix not available")
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        if distance_metric == 'bray_curtis':
            distances = self._bray_curtis_distance(self.species_matrix.values)
        else:
            distances = pdist(self.species_matrix.values, metric=distance_metric)
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        linkage_matrix = linkage(distances, method=linkage_method)
        
        results = {
            'linkage_matrix': linkage_matrix,
            'distances': distances
        }
        
        if n_clusters is not None:
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            results['cluster_labels'] = pd.Series(cluster_labels, 
                                                index=self.species_matrix.index,
                                                name='cluster')
        
        return results
    
    def kmeans_clustering(self, n_clusters: int = 3,
                         transform: str = 'hellinger') -> Dict[str, Any]:
        """
        K-means clustering analysis.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters
        transform : str
            Data transformation method
            
        Returns:
        --------
        dict
            K-means results
        """
        if self.species_matrix is None:
            raise ValueError("Species matrix not available")
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        transformed_data = self._transform_data(self.species_matrix, transform)
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(transformed_data)
        
        results = {
            'cluster_labels': pd.Series(cluster_labels,
                                      index=self.species_matrix.index,
                                      name='cluster'),
            'centroids': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'kmeans_object': kmeans
        }
        
        return results
    
    def indicator_species_analysis(self, clusters: pd.Series) -> pd.DataFrame:
        """
        Indicator species analysis for clusters.
        
        Parameters:
        -----------
        clusters : pd.Series
            Cluster assignments
            
        Returns:
        --------
        pd.DataFrame
            Indicator species results
        """
        if self.species_matrix is None:
            raise ValueError("Species matrix not available")
        
        results = []
        
        for species in self.species_matrix.columns:
            for cluster in clusters.unique():
                cluster_mask = clusters == cluster
                
                # Copyright (c) 2025 Mohamed Z. Hatim
                freq_in_cluster = (self.species_matrix.loc[cluster_mask, species] > 0).mean()
                freq_out_cluster = (self.species_matrix.loc[~cluster_mask, species] > 0).mean()
                
                abund_in_cluster = self.species_matrix.loc[cluster_mask, species].mean()
                abund_out_cluster = self.species_matrix.loc[~cluster_mask, species].mean()
                
                # Copyright (c) 2025 Mohamed Z. Hatim
                if freq_out_cluster + freq_in_cluster > 0 and abund_out_cluster + abund_in_cluster > 0:
                    rel_freq = freq_in_cluster / (freq_in_cluster + freq_out_cluster)
                    rel_abund = abund_in_cluster / (abund_in_cluster + abund_out_cluster)
                    indicator_value = rel_freq * rel_abund * 100
                else:
                    indicator_value = 0
                
                results.append({
                    'species': species,
                    'cluster': cluster,
                    'indicator_value': indicator_value,
                    'frequency_in_cluster': freq_in_cluster,
                    'abundance_in_cluster': abund_in_cluster
                })
        
        return pd.DataFrame(results)
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    # Copyright (c) 2025 Mohamed Z. Hatim
    # Copyright (c) 2025 Mohamed Z. Hatim
    
    def plot_diversity(self, diversity_data: pd.DataFrame, 
                      index_name: str = 'shannon') -> plt.Figure:
        """
        Plot diversity indices.
        
        Parameters:
        -----------
        diversity_data : pd.DataFrame
            Diversity indices data
        index_name : str
            Name of diversity index to plot
            
        Returns:
        --------
        plt.Figure
            Diversity plot
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        if index_name in diversity_data.columns:
            diversity_data[index_name].hist(bins=20, ax=ax, edgecolor='black', alpha=0.7)
            ax.set_xlabel(f'{index_name.title()} Diversity')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {index_name.title()} Diversity')
        else:
            ax.text(0.5, 0.5, f'Index "{index_name}" not found', 
                   transform=ax.transAxes, ha='center', va='center')
        
        plt.tight_layout()
        return fig
    
    def plot_ordination(self, ordination_results: Dict[str, Any],
                       color_by: Optional[pd.Series] = None) -> plt.Figure:
        """
        Plot ordination results.
        
        Parameters:
        -----------
        ordination_results : dict
            Results from PCA or NMDS analysis
        color_by : pd.Series, optional
            Variable to color points by
            
        Returns:
        --------
        plt.Figure
            Ordination plot
        """
        scores = ordination_results['scores']
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        if color_by is not None:
            scatter = ax.scatter(scores.iloc[:, 0], scores.iloc[:, 1], 
                               c=color_by, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, ax=ax, label=color_by.name if color_by.name else 'Color')
        else:
            ax.scatter(scores.iloc[:, 0], scores.iloc[:, 1], alpha=0.7)
        
        ax.set_xlabel(f'{scores.columns[0]}')
        ax.set_ylabel(f'{scores.columns[1]}')
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        if 'explained_variance_ratio' in ordination_results:
            var_exp = ordination_results['explained_variance_ratio']
            ax.set_xlabel(f'{scores.columns[0]} ({var_exp[0]:.1%})')
            ax.set_ylabel(f'{scores.columns[1]} ({var_exp[1]:.1%})')
        
        ax.set_title('Ordination Plot')
        plt.tight_layout()
        return fig
    
    def plot_cluster_dendrogram(self, clustering_results: Dict[str, Any]) -> plt.Figure:
        """
        Plot hierarchical clustering dendrogram.
        
        Parameters:
        -----------
        clustering_results : dict
            Results from hierarchical clustering
            
        Returns:
        --------
        plt.Figure
            Dendrogram plot
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        dendrogram(clustering_results['linkage_matrix'], 
                  ax=ax, orientation='top')
        
        ax.set_title('Hierarchical Clustering Dendrogram')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Distance')
        
        plt.tight_layout()
        return fig
    
    def plot_species_accumulation(self, rarefaction_data: pd.DataFrame) -> plt.Figure:
        """
        Plot species accumulation curves.
        
        Parameters:
        -----------
        rarefaction_data : pd.DataFrame
            Rarefaction curve data
            
        Returns:
        --------
        plt.Figure
            Species accumulation plot
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        for sample_id in rarefaction_data['sample_id'].unique():
            sample_data = rarefaction_data[rarefaction_data['sample_id'] == sample_id]
            ax.plot(sample_data['sample_size'], sample_data['expected_species'], 
                   alpha=0.3, color='gray')
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        mean_curve = rarefaction_data.groupby('sample_size')['expected_species'].mean()
        ax.plot(mean_curve.index, mean_curve.values, 'b-', linewidth=2, label='Mean')
        
        ax.set_xlabel('Sample Size (Number of Individuals)')
        ax.set_ylabel('Expected Number of Species')
        ax.set_title('Species Accumulation Curves')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    # Copyright (c) 2025 Mohamed Z. Hatim
    # Copyright (c) 2025 Mohamed Z. Hatim
    
    def summary_statistics(self) -> Dict[str, Any]:
        """
        Generate summary statistics for the dataset.
        
        Returns:
        --------
        dict
            Summary statistics
        """
        if self.species_matrix is None:
            raise ValueError("Species matrix not available")
        
        stats = {
            'n_sites': len(self.species_matrix),
            'n_species': len(self.species_matrix.columns),
            'total_abundance': self.species_matrix.sum().sum(),
            'mean_species_per_site': (self.species_matrix > 0).sum(axis=1).mean(),
            'mean_abundance_per_site': self.species_matrix.sum(axis=1).mean(),
            'species_occurrence_frequency': (self.species_matrix > 0).sum(axis=0).describe(),
            'site_abundance_distribution': self.species_matrix.sum(axis=1).describe()
        }
        
        return stats
    
    def export_results(self, results: Dict[str, Any], 
                      output_path: str, format_type: str = 'csv') -> None:
        """
        Export analysis results.
        
        Parameters:
        -----------
        results : dict
            Analysis results
        output_path : str
            Output file path
        format_type : str
            Output format
        """
        if format_type.lower() == 'csv':
            if isinstance(results, pd.DataFrame):
                results.to_csv(output_path)
            elif isinstance(results, dict):
                # Copyright (c) 2025 Mohamed Z. Hatim
                for key, value in results.items():
                    if isinstance(value, pd.DataFrame):
                        filepath = f"{output_path}_{key}.csv"
                        value.to_csv(filepath)
                        print(f"Exported {key} to {filepath}")
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    # Copyright (c) 2025 Mohamed Z. Hatim
    # Copyright (c) 2025 Mohamed Z. Hatim
    
    def elbow_analysis(self, k_range: range = range(1, 16),
                      methods: List[str] = ['knee_locator', 'derivative', 'variance_explained'],
                      transform: str = 'hellinger',
                      plot_results: bool = True) -> Dict[str, Any]:
        """
        Comprehensive elbow analysis to determine optimal number of clusters.
        
        Parameters:
        -----------
        k_range : range
            Range of k values to test (default: 1 to 15)
        methods : list
            Elbow detection methods to use
            Available: 'knee_locator', 'derivative', 'variance_explained', 
                      'distortion_jump', 'l_method'
        transform : str
            Data transformation method ('hellinger', 'log', 'sqrt', 'none')
        plot_results : bool
            Whether to create visualization plots
            
        Returns:
        --------
        dict
            Comprehensive elbow analysis results including:
            - optimal k recommendations from each method
            - consensus recommendation
            - confidence scores
            - detailed metrics for all k values
            - visualization plots (if requested)
        """
        if self.species_matrix is None:
            raise ValueError("Species matrix not available. Please load data first.")
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        from .clustering import VegetationClustering
        clustering = VegetationClustering()
        
        return clustering.comprehensive_elbow_analysis(
            data=self.species_matrix,
            k_range=k_range,
            methods=methods,
            transform=transform,
            plot_results=plot_results
        )
    
    def quick_elbow_analysis(self, max_k: int = 10) -> int:
        """
        Quick elbow analysis using the most reliable method.
        
        Parameters:
        -----------
        max_k : int
            Maximum number of clusters to test
            
        Returns:
        --------
        int
            Recommended optimal number of clusters
        """
        if self.species_matrix is None:
            raise ValueError("Species matrix not available. Please load data first.")
        
        results = self.elbow_analysis(
            k_range=range(1, max_k + 1),
            methods=['knee_locator', 'derivative'],
            plot_results=False
        )
        
        if results['recommendations']['consensus']:
            return results['recommendations']['consensus']
        else:
            # Copyright (c) 2025 Mohamed Z. Hatim
            return results['recommendations'].get('silhouette_optimal', 3)


# Copyright (c) 2025 Mohamed Z. Hatim
def quick_diversity_analysis(data: pd.DataFrame, 
                           species_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Quick diversity analysis."""
    veg = VegZ()
    veg.data = data
    if species_cols:
        veg.species_matrix = data[species_cols]
    else:
        veg.species_matrix = data.select_dtypes(include=[np.number])
    
    return veg.calculate_diversity()


def quick_ordination(data: pd.DataFrame,
                    species_cols: Optional[List[str]] = None,
                    method: str = 'pca') -> Dict[str, Any]:
    """Quick ordination analysis."""
    veg = VegZ()
    veg.data = data
    if species_cols:
        veg.species_matrix = data[species_cols]
    else:
        veg.species_matrix = data.select_dtypes(include=[np.number])
    
    if method.lower() == 'pca':
        return veg.pca_analysis()
    elif method.lower() == 'nmds':
        return veg.nmds_analysis()
    else:
        raise ValueError(f"Unknown ordination method: {method}")


def quick_clustering(data: pd.DataFrame,
                    species_cols: Optional[List[str]] = None,
                    n_clusters: int = 3,
                    method: str = 'kmeans') -> Dict[str, Any]:
    """Quick clustering analysis."""
    veg = VegZ()
    veg.data = data
    if species_cols:
        veg.species_matrix = data[species_cols]
    else:
        veg.species_matrix = data.select_dtypes(include=[np.number])
    
    if method.lower() == 'kmeans':
        return veg.kmeans_clustering(n_clusters=n_clusters)
    elif method.lower() == 'hierarchical':
        return veg.hierarchical_clustering(n_clusters=n_clusters)
    else:
        raise ValueError(f"Unknown clustering method: {method}")


def quick_elbow_analysis(data: pd.DataFrame,
                        species_cols: Optional[List[str]] = None,
                        max_k: int = 10,
                        plot_results: bool = True) -> Dict[str, Any]:
    """
    Quick elbow analysis to determine optimal number of clusters.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    species_cols : list, optional  
        Column names containing species data
    max_k : int
        Maximum number of clusters to test
    plot_results : bool
        Whether to create visualization plots
        
    Returns:
    --------
    dict
        Elbow analysis results including optimal k recommendation
    """
    veg = VegZ()
    veg.data = data
    if species_cols:
        veg.species_matrix = data[species_cols]
    else:
        veg.species_matrix = data.select_dtypes(include=[np.number])
    
    return veg.elbow_analysis(
        k_range=range(1, max_k + 1),
        methods=['knee_locator', 'derivative', 'variance_explained'],
        plot_results=plot_results
    )