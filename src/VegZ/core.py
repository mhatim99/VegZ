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
from typing import List, Dict, Optional, Any
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings

from ._compat import make_mds

#: Transformations understood by :meth:`VegZ._transform_data`.
TRANSFORMATIONS = ('hellinger', 'chord', 'wisconsin', 'log', 'sqrt',
                   'standardize', 'none')


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
        
        if species_cols is None:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            self.species_matrix = self.data[numeric_cols]
            warnings.warn(
                "species_cols was not supplied, so every numeric column was "
                "treated as a species. Coordinate, elevation or ID columns "
                "will corrupt the analysis - pass species_cols explicitly.",
                UserWarning,
            )
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
                           min_abundance: float = 0.0,
                           verbose: bool = False) -> pd.DataFrame:
        """
        Filter out rare species based on occurrence frequency and abundance.

        Parameters:
        -----------
        min_occurrences : int
            Minimum number of sites where species must occur
        min_abundance : float
            Minimum total abundance threshold
        verbose : bool
            Print a one-line summary of how many species were retained.

        Returns:
        --------
        pd.DataFrame
            Filtered species matrix
        """
        if self.species_matrix is None:
            raise ValueError("Species matrix not available")

        occurrences = (self.species_matrix > 0).sum(axis=0)
        total_abundance = self.species_matrix.sum(axis=0)

        keep_species = (occurrences >= min_occurrences) & (total_abundance >= min_abundance)

        n_kept, n_total = int(keep_species.sum()), len(keep_species)
        self.species_matrix = self.species_matrix.loc[:, keep_species]
        self.metadata['filter_rare_species'] = {
            'min_occurrences': min_occurrences,
            'min_abundance': min_abundance,
            'n_retained': n_kept,
            'n_original': n_total,
        }

        if verbose:
            print(f"Retained {n_kept} species out of {n_total} original species")

        return self.species_matrix
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    # Copyright (c) 2025 Mohamed Z. Hatim
    # Copyright (c) 2025 Mohamed Z. Hatim
    
    def calculate_diversity(self,
                            indices: Optional[List[str]] = None) -> pd.DataFrame:
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

        # A mutable default would be shared between every call.
        if indices is None:
            indices = ['shannon', 'simpson', 'richness']

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
            total = row.sum()
            if total == 0:
                return 0.0
            proportions = row[row > 0] / total
            return -np.sum(proportions * np.log(proportions))

        return self.species_matrix.apply(shannon, axis=1)
    
    def _simpson_diversity(self) -> pd.Series:
        """Calculate Simpson diversity index (D = sum of squared proportions)."""
        def simpson(row):
            total = row.sum()
            if total == 0:
                return 0.0
            proportions = row[row > 0] / total
            # Copyright (c) 2025 Mohamed Z. Hatim
            return np.sum(proportions ** 2)

        return self.species_matrix.apply(simpson, axis=1)
    
    def _species_richness(self) -> pd.Series:
        """Calculate species richness."""
        return (self.species_matrix > 0).sum(axis=1)
    
    def _evenness(self) -> pd.Series:
        """Calculate Pielou's evenness."""
        shannon = self._shannon_diversity()
        richness = self._species_richness()
        log_richness = np.log(richness.replace({0: np.nan, 1: np.nan}))
        evenness = shannon / log_richness
        return evenness.fillna(0)
    
    def rarefaction_curve(self, sample_sizes: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate individual-based (Hurlbert) rarefaction curves.

        Parameters:
        -----------
        sample_sizes : list, optional
            Numbers of individuals to rarefy to.

        Returns:
        --------
        pd.DataFrame
            Long-format rarefaction data with ``sample_id``, ``sample_size``,
            ``expected_species`` and ``variance`` columns.
        """
        if self.species_matrix is None:
            raise ValueError("Species matrix not available")

        from .diversity import DiversityAnalyzer

        return DiversityAnalyzer().rarefaction_curve(
            self.species_matrix, sample_sizes=sample_sizes
        )

    def species_accumulation_curve(self, n_permutations: int = 100,
                                   random_state: Optional[int] = None) -> pd.DataFrame:
        """
        Sample-based species accumulation curve with permutation confidence bands.

        Parameters:
        -----------
        n_permutations : int
            Number of random site orderings.
        random_state : int, optional
            Seed for reproducible permutations.

        Returns:
        --------
        pd.DataFrame
            Columns ``n_sites``, ``mean``, ``std``, ``ci_lower``, ``ci_upper``.
        """
        if self.species_matrix is None:
            raise ValueError("Species matrix not available")

        from .diversity import DiversityAnalyzer

        return DiversityAnalyzer().species_accumulation_curve(
            self.species_matrix, n_permutations=n_permutations,
            random_state=random_state
        )
    
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
        
        columns = [f'PC{i+1}' for i in range(scores.shape[1])]
        site_scores = pd.DataFrame(scores, index=self.species_matrix.index,
                                   columns=columns)
        loadings = pd.DataFrame(pca.components_.T,
                                index=self.species_matrix.columns,
                                columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])])

        results = {
            'scores': site_scores,
            # 'site_scores'/'species_scores' are the names MultivariateAnalyzer
            # and the plotting helpers use; exposing both keeps results from the
            # two entry points interchangeable.
            'site_scores': site_scores,
            'loadings': loadings,
            'species_scores': loadings * np.sqrt(pca.explained_variance_),
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'eigenvalues': pca.explained_variance_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'method': 'PCA',
            'pca_object': pca
        }

        return results
    
    def nmds_analysis(self, distance_metric: str = 'bray_curtis',
                      n_dimensions: int = 2,
                      transform: str = 'none',
                      n_init: int = 10,
                      max_iter: int = 300,
                      random_state: Optional[int] = 42) -> Dict[str, Any]:
        """
        Non-metric Multidimensional Scaling (NMDS).

        Parameters:
        -----------
        distance_metric : str
            Distance metric to use ('bray_curtis', 'euclidean', or any metric
            accepted by :func:`scipy.spatial.distance.pdist`).
        n_dimensions : int
            Number of dimensions
        transform : str
            Data transformation applied before computing distances. Defaults to
            ``'none'``: Bray-Curtis is already a relativising measure, so
            transforming first (e.g. Hellinger) changes what the distance means.
        n_init : int
            Number of random restarts of the stress minimisation.
        max_iter : int
            Maximum SMACOF iterations per restart.
        random_state : int, optional
            Seed for reproducible configurations.

        Returns:
        --------
        dict
            NMDS results including scores, Kruskal stress-1 and the distance
            matrix used.
        """
        if self.species_matrix is None:
            raise ValueError("Species matrix not available")

        if transform == 'standardize' and distance_metric == 'bray_curtis':
            raise ValueError(
                "Bray-Curtis distance requires non-negative data. Cannot use "
                "with 'standardize' transform. Use 'hellinger', 'log', or "
                "'sqrt' instead."
            )

        transformed_data = self._transform_data(self.species_matrix, transform)
        distances = self._pairwise_distances(transformed_data, distance_metric)

        # metric=False gives *non*-metric MDS, which is what NMDS means.
        mds = make_mds(n_components=n_dimensions, metric=False, precomputed=True,
                       n_init=n_init, max_iter=max_iter, random_state=random_state)
        scores = mds.fit_transform(squareform(distances))

        site_scores = pd.DataFrame(
            scores, index=self.species_matrix.index,
            columns=[f'NMDS{i+1}' for i in range(n_dimensions)]
        )

        results = {
            'scores': site_scores,
            # Alias matching MultivariateAnalyzer / the plotting helpers.
            'site_scores': site_scores,
            'stress': mds.stress_,
            'distances': distances,
            'distance_metric': distance_metric,
            'method': 'NMDS',
            'mds_object': mds
        }

        return results

    def _transform_data(self, data: pd.DataFrame, method: str) -> np.ndarray:
        """
        Apply a data transformation and return a NumPy array.

        Supported: ``hellinger``, ``chord``, ``wisconsin``, ``log`` (log1p),
        ``sqrt``, ``standardize`` and ``none``.
        """
        method = (method or 'none').lower()
        values = data.values.astype(float)

        if method in ('none', 'raw'):
            return values
        if method == 'hellinger':
            row_sums = values.sum(axis=1)
            row_sums[row_sums == 0] = 1.0
            return np.sqrt(values / row_sums[:, None])
        if method == 'chord':
            norms = np.sqrt((values ** 2).sum(axis=1))
            norms[norms == 0] = 1.0
            return values / norms[:, None]
        if method == 'wisconsin':
            col_max = values.max(axis=0)
            col_max[col_max == 0] = 1.0
            relativised = values / col_max
            row_sums = relativised.sum(axis=1)
            row_sums[row_sums == 0] = 1.0
            return relativised / row_sums[:, None]
        if method == 'log':
            return np.log1p(values)
        if method == 'sqrt':
            return np.sqrt(np.maximum(values, 0))
        if method == 'standardize':
            return StandardScaler().fit_transform(values)

        raise ValueError(
            f"Unknown transformation '{method}'. Valid options: "
            f"{', '.join(TRANSFORMATIONS)}"
        )

    @staticmethod
    def _pairwise_distances(data: np.ndarray, metric: str) -> np.ndarray:
        """Condensed pairwise distance vector for a site-by-species array."""
        aliases = {
            'bray_curtis': 'braycurtis',
            'manhattan': 'cityblock',
            'city_block': 'cityblock',
        }
        scipy_metric = aliases.get(metric, metric)
        return pdist(np.asarray(data, dtype=float), metric=scipy_metric)

    def _bray_curtis_distance(self, data: np.ndarray) -> np.ndarray:
        """Condensed Bray-Curtis distance vector (kept for backwards compatibility)."""
        return self._pairwise_distances(np.asarray(data, dtype=float), 'bray_curtis')
    
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

        if linkage_method == 'ward' and distance_metric != 'euclidean':
            warnings.warn("Ward linkage requires Euclidean distance. Switching to Euclidean.")
            distance_metric = 'euclidean'

        distances = self._pairwise_distances(self.species_matrix.values, distance_metric)

        linkage_matrix = linkage(distances, method=linkage_method)

        results = {
            'linkage_matrix': linkage_matrix,
            'distances': distances,
            'distance_metric': distance_metric,
            'linkage_method': linkage_method,
            'site_labels': self.species_matrix.index.tolist(),
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
    
    def indicator_species_analysis(self, clusters: pd.Series,
                                   permutations: int = 0,
                                   random_state: Optional[int] = None) -> pd.DataFrame:
        """
        Dufrene-Legendre indicator species analysis (IndVal) for clusters.

        For species *j* and cluster *k*:

        ``A_kj`` = mean abundance of *j* in *k* divided by the sum of its mean
        abundances across all clusters (specificity), and ``B_kj`` = proportion
        of sites within *k* where *j* occurs (fidelity). ``IndVal = A*B*100``.

        Parameters:
        -----------
        clusters : pd.Series
            Cluster assignments, indexed like the species matrix.
        permutations : int
            If > 0, run a permutation test of the maximum IndVal per species
            and add a ``p_value`` column.
        random_state : int, optional
            Seed for the permutation test.

        Returns:
        --------
        pd.DataFrame
            One row per species x cluster combination with ``indicator_value``,
            ``specificity`` (A), ``fidelity`` (B) and supporting counts.
        """
        if self.species_matrix is None:
            raise ValueError("Species matrix not available")

        from .statistics import EcologicalStatistics

        clusters = pd.Series(clusters)
        if not clusters.index.equals(self.species_matrix.index):
            clusters = clusters.set_axis(self.species_matrix.index)

        return EcologicalStatistics().indicator_species_analysis(
            self.species_matrix, clusters,
            permutations=permutations, random_state=random_state,
            as_frame=True,
        )
    
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
        if 'scores' in ordination_results:
            scores = ordination_results['scores']
        elif 'site_scores' in ordination_results:
            scores = ordination_results['site_scores']
        elif 'coordinates' in ordination_results:
            scores = ordination_results['coordinates']
        else:
            raise ValueError("No ordination scores found in results")

        if scores.shape[1] < 2:
            raise ValueError(
                "Ordination plot needs at least two axes; the supplied result "
                f"has {scores.shape[1]}."
            )

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        if color_by is not None:
            scatter = ax.scatter(scores.iloc[:, 0], scores.iloc[:, 1], 
                               c=color_by, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, ax=ax, label=color_by.name if color_by.name else 'Color')
        else:
            ax.scatter(scores.iloc[:, 0], scores.iloc[:, 1], alpha=0.7)
        
        ax.set_xlabel(f'{scores.columns[0]}')
        ax.set_ylabel(f'{scores.columns[1]}')
        
        var_exp = ordination_results.get('explained_variance_ratio')
        if var_exp is not None and len(var_exp) >= 2:
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
        
        summary = {
            'n_sites': len(self.species_matrix),
            'n_species': len(self.species_matrix.columns),
            'total_abundance': self.species_matrix.sum().sum(),
            'mean_species_per_site': (self.species_matrix > 0).sum(axis=1).mean(),
            'mean_abundance_per_site': self.species_matrix.sum(axis=1).mean(),
            'species_occurrence_frequency': (self.species_matrix > 0).sum(axis=0).describe(),
            'site_abundance_distribution': self.species_matrix.sum(axis=1).describe()
        }

        return summary
    
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
                      methods: Optional[List[str]] = None,
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

        if methods is None:
            methods = ['knee_locator', 'derivative', 'variance_explained']

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


def _prepare(data: pd.DataFrame, species_cols: Optional[List[str]]) -> 'VegZ':
    """Build a :class:`VegZ` instance for the quick_* helper functions."""
    veg = VegZ()
    veg.data = data
    if species_cols:
        veg.species_matrix = data[species_cols]
    else:
        veg.species_matrix = data.select_dtypes(include=[np.number])
        warnings.warn(
            "species_cols was not supplied, so every numeric column was "
            "treated as a species. Coordinate, elevation or ID columns will "
            "corrupt the analysis - pass species_cols explicitly.",
            UserWarning,
        )
    return veg


def quick_diversity_analysis(data: pd.DataFrame, 
                           species_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Quick diversity analysis."""
    veg = _prepare(data, species_cols)
    
    return veg.calculate_diversity()


def quick_ordination(data: pd.DataFrame,
                    species_cols: Optional[List[str]] = None,
                    method: str = 'pca') -> Dict[str, Any]:
    """Quick ordination analysis."""
    veg = _prepare(data, species_cols)
    
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
    veg = _prepare(data, species_cols)
    
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
    veg = _prepare(data, species_cols)
    
    return veg.elbow_analysis(
        k_range=range(1, max_k + 1),
        methods=['knee_locator', 'derivative', 'variance_explained'],
        plot_results=plot_results
    )