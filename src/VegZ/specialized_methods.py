"""
Specialized Methods Module

This module provides specialized ecological analysis methods including phylogenetic diversity,
metacommunity analysis, food web analysis, and other advanced ecological techniques.

Copyright (c) 2025 Mohamed Z. Hatim
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra
    GRAPH_METHODS_AVAILABLE = True
except ImportError:
    GRAPH_METHODS_AVAILABLE = False
    warnings.warn("Scipy graph methods not available, some network analyses will be limited")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    warnings.warn("NetworkX not available, advanced network analyses will be limited")


class PhylogeneticDiversityAnalyzer:
    """
    Phylogenetic diversity analysis for vegetation communities.
    
    Calculates phylogenetic diversity metrics using phylogenetic trees
    and community composition data.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize PhylogeneticDiversityAnalyzer."""
        self.random_state = random_state
        self.phylo_tree = None
        self.distance_matrix = None
        
    def load_phylogeny(self, phylo_data: Union[pd.DataFrame, Dict[str, Any]], 
                      format: str = 'distance_matrix') -> None:
        """
        Load phylogenetic data.
        
        Parameters
        ----------
        phylo_data : Union[pd.DataFrame, Dict[str, Any]]
            Phylogenetic data (distance matrix, tree, etc.)
        format : str, optional
            Format of phylogenetic data ('distance_matrix', 'newick'), by default 'distance_matrix'
        """
        if format == 'distance_matrix':
            if isinstance(phylo_data, pd.DataFrame):
                self.distance_matrix = phylo_data
            else:
                raise ValueError("Distance matrix must be a DataFrame")
        elif format == 'newick':
# Copyright (c) 2025 Mohamed Z. Hatim
            warnings.warn("Newick format support is simplified. Use specialized phylogenetic libraries for full support.")
            self.phylo_tree = phylo_data
        else:
            raise ValueError(f"Unknown phylogenetic data format: {format}")
    
    def calculate_phylogenetic_diversity(self, 
                                       community_data: pd.DataFrame,
                                       metrics: List[str] = None) -> Dict[str, Any]:
        """
        Calculate phylogenetic diversity metrics.
        
        Parameters
        ----------
        community_data : pd.DataFrame
            Community composition data (sites x species)
        metrics : List[str], optional
            Phylogenetic diversity metrics to calculate
            
        Returns
        -------
        Dict[str, Any]
            Phylogenetic diversity results
        """
        if metrics is None:
            metrics = ['pd', 'mpd', 'mntd', 'nri', 'nti']
        
        if self.distance_matrix is None:
            raise ValueError("Phylogenetic data not loaded. Use load_phylogeny() first.")
        
# Copyright (c) 2025 Mohamed Z. Hatim
        common_species = list(set(community_data.columns) & set(self.distance_matrix.index))
        if len(common_species) == 0:
            raise ValueError("No common species between community data and phylogeny")
        
        community_subset = community_data[common_species]
        phylo_subset = self.distance_matrix.loc[common_species, common_species]
        
        results = {
            'metrics': {},
            'species_used': common_species,
            'n_species': len(common_species)
        }
        
# Copyright (c) 2025 Mohamed Z. Hatim
        site_results = {}
        for site in community_subset.index:
            site_composition = community_subset.loc[site]
            present_species = site_composition[site_composition > 0].index.tolist()
            
            if len(present_species) < 2:
# Copyright (c) 2025 Mohamed Z. Hatim
                continue
            
            site_phylo = phylo_subset.loc[present_species, present_species]
            abundances = site_composition.loc[present_species]
            
            site_metrics = {}
            
# Copyright (c) 2025 Mohamed Z. Hatim
            if 'pd' in metrics:
                site_metrics['pd'] = self._calculate_faith_pd(site_phylo, abundances)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            if 'mpd' in metrics:
                site_metrics['mpd'] = self._calculate_mpd(site_phylo, abundances)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            if 'mntd' in metrics:
                site_metrics['mntd'] = self._calculate_mntd(site_phylo, abundances)
            
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
            if 'nri' in metrics or 'nti' in metrics:
                null_mpd, null_mntd = self._calculate_null_phylo_metrics(
                    phylo_subset, len(present_species), n_iterations=99
                )
                
                if 'nri' in metrics:
                    site_metrics['nri'] = (null_mpd['mean'] - site_metrics.get('mpd', 0)) / null_mpd['std'] if null_mpd['std'] > 0 else 0
                if 'nti' in metrics:
                    site_metrics['nti'] = (null_mntd['mean'] - site_metrics.get('mntd', 0)) / null_mntd['std'] if null_mntd['std'] > 0 else 0
            
            site_results[site] = site_metrics
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if site_results:
            results['metrics'] = pd.DataFrame(site_results).T
        
        return results
    
    def _calculate_faith_pd(self, phylo_distances: pd.DataFrame, 
                          abundances: pd.Series) -> float:
        """Calculate Faith's Phylogenetic Diversity."""
        if len(phylo_distances) < 2:
            return 0
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if GRAPH_METHODS_AVAILABLE:
# Copyright (c) 2025 Mohamed Z. Hatim
            distance_matrix = phylo_distances.values
            mst = minimum_spanning_tree(distance_matrix).toarray()
            pd_value = np.sum(mst)
        else:
# Copyright (c) 2025 Mohamed Z. Hatim
            distances = phylo_distances.values
            unique_distances = distances[np.triu_indices(len(distances), k=1)]
            pd_value = np.sum(unique_distances) / len(unique_distances)
        
        return pd_value
    
    def _calculate_mpd(self, phylo_distances: pd.DataFrame, 
                      abundances: pd.Series) -> float:
        """Calculate Mean Pairwise Distance."""
        distances = phylo_distances.values
        weights = abundances.values
        
# Copyright (c) 2025 Mohamed Z. Hatim
        total_weight = 0
        weighted_distance = 0
        
        for i in range(len(distances)):
            for j in range(i + 1, len(distances)):
                weight = weights[i] * weights[j]
                weighted_distance += distances[i, j] * weight
                total_weight += weight
        
        return weighted_distance / total_weight if total_weight > 0 else 0
    
    def _calculate_mntd(self, phylo_distances: pd.DataFrame, 
                       abundances: pd.Series) -> float:
        """Calculate Mean Nearest Taxon Distance."""
        distances = phylo_distances.values
        np.fill_diagonal(distances, np.inf)  # Exclude self-distances
        
# Copyright (c) 2025 Mohamed Z. Hatim
        nearest_distances = []
        for i in range(len(distances)):
            min_dist = np.min(distances[i, :])
            if np.isfinite(min_dist):
                nearest_distances.append(min_dist)
        
        return np.mean(nearest_distances) if nearest_distances else 0
    
    def _calculate_null_phylo_metrics(self, phylo_matrix: pd.DataFrame, 
                                    n_species: int, 
                                    n_iterations: int = 99) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate null model expectations for phylogenetic metrics."""
        null_mpd_values = []
        null_mntd_values = []
        
        for _ in range(n_iterations):
# Copyright (c) 2025 Mohamed Z. Hatim
            sampled_species = np.random.choice(
                phylo_matrix.index, size=n_species, replace=False
            )
            null_phylo = phylo_matrix.loc[sampled_species, sampled_species]
            null_abundances = pd.Series(1, index=sampled_species)  # Equal abundances
            
            null_mpd = self._calculate_mpd(null_phylo, null_abundances)
            null_mntd = self._calculate_mntd(null_phylo, null_abundances)
            
            null_mpd_values.append(null_mpd)
            null_mntd_values.append(null_mntd)
        
        return (
            {'mean': np.mean(null_mpd_values), 'std': np.std(null_mpd_values)},
            {'mean': np.mean(null_mntd_values), 'std': np.std(null_mntd_values)}
        )


class MetacommunityAnalyzer:
    """
    Metacommunity analysis for studying species distributions across multiple sites.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize MetacommunityAnalyzer."""
        self.random_state = random_state
        
    def elements_of_metacommunity_structure(self, 
                                          community_data: pd.DataFrame,
                                          method: str = 'reciprocal_averaging') -> Dict[str, Any]:
        """
        Analyze Elements of Metacommunity Structure (EMS).
        
        Parameters
        ----------
        community_data : pd.DataFrame
            Community composition data (sites x species)
        method : str, optional
            Ordination method to use, by default 'reciprocal_averaging'
            
        Returns
        -------
        Dict[str, Any]
            EMS analysis results
        """
# Copyright (c) 2025 Mohamed Z. Hatim
        pa_data = (community_data > 0).astype(int)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        coherence = self._calculate_coherence(pa_data)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        turnover = self._calculate_turnover(pa_data)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        boundary_clumping = self._calculate_boundary_clumping(pa_data)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        structure_type = self._classify_metacommunity_structure(
            coherence, turnover, boundary_clumping
        )
        
        return {
            'coherence': coherence,
            'turnover': turnover, 
            'boundary_clumping': boundary_clumping,
            'structure_type': structure_type,
            'interpretation': self._interpret_structure(structure_type)
        }
    
    def _calculate_coherence(self, pa_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate species coherence."""
        n_sites, n_species = pa_data.shape
        
# Copyright (c) 2025 Mohamed Z. Hatim
        embedded_absences = 0
        total_possible = 0
        
        for species in pa_data.columns:
            presence_sites = pa_data[pa_data[species] == 1].index
            if len(presence_sites) < 2:
                continue
            
# Copyright (c) 2025 Mohamed Z. Hatim
            site_richness = pa_data.sum(axis=1)
            sorted_sites = site_richness.sort_values(ascending=False).index
            
# Copyright (c) 2025 Mohamed Z. Hatim
            presence_positions = [i for i, site in enumerate(sorted_sites) if site in presence_sites]
            if len(presence_positions) < 2:
                continue
            
            first_pos = min(presence_positions)
            last_pos = max(presence_positions)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            for pos in range(first_pos + 1, last_pos):
                site = sorted_sites[pos]
                if pa_data.loc[site, species] == 0:
                    embedded_absences += 1
            
            total_possible += last_pos - first_pos - 1
        
        coherence_value = 1 - (embedded_absences / total_possible) if total_possible > 0 else 1
        
        return {
            'value': coherence_value,
            'embedded_absences': embedded_absences,
            'total_possible': total_possible
        }
    
    def _calculate_turnover(self, pa_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate species turnover."""
# Copyright (c) 2025 Mohamed Z. Hatim
        site_richness = pa_data.sum(axis=1)
        total_richness = (pa_data.sum(axis=0) > 0).sum()
        mean_alpha = site_richness.mean()
        
        beta_w = (total_richness / mean_alpha) - 1
        
# Copyright (c) 2025 Mohamed Z. Hatim
        pairwise_turnover = []
        for i, site1 in enumerate(pa_data.index):
            for site2 in pa_data.index[i+1:]:
                shared = (pa_data.loc[site1] & pa_data.loc[site2]).sum()
                unique1 = (pa_data.loc[site1] & ~pa_data.loc[site2]).sum()
                unique2 = (~pa_data.loc[site1] & pa_data.loc[site2]).sum()
                
# Copyright (c) 2025 Mohamed Z. Hatim
                turnover = (unique1 + unique2) / (2 * shared + unique1 + unique2) if (shared + unique1 + unique2) > 0 else 0
                pairwise_turnover.append(turnover)
        
        return {
            'whittaker_beta': beta_w,
            'mean_pairwise_turnover': np.mean(pairwise_turnover),
            'pairwise_turnover': pairwise_turnover
        }
    
    def _calculate_boundary_clumping(self, pa_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate boundary clumping."""
# Copyright (c) 2025 Mohamed Z. Hatim
# Copyright (c) 2025 Mohamed Z. Hatim
        
        site_richness = pa_data.sum(axis=1).sort_values(ascending=False)
        species_ranges = {}
        
        for species in pa_data.columns:
            presence_sites = pa_data[pa_data[species] == 1].index
            if len(presence_sites) == 0:
                continue
            
# Copyright (c) 2025 Mohamed Z. Hatim
            positions = [list(site_richness.index).index(site) for site in presence_sites]
            range_size = max(positions) - min(positions) + 1
            species_ranges[species] = range_size / len(presence_sites)
        
        mean_clumping = np.mean(list(species_ranges.values())) if species_ranges else 1
        
        return {
            'mean_clumping': mean_clumping,
            'species_clumping': species_ranges
        }
    
    def _classify_metacommunity_structure(self, coherence: Dict, turnover: Dict, 
                                        boundary_clumping: Dict) -> str:
        """Classify metacommunity structure type."""
        coherent = coherence['value'] > 0.5
        high_turnover = turnover['whittaker_beta'] > 1.5
        clumped = boundary_clumping['mean_clumping'] < 0.5
        
        if not coherent:
            return "random"
        elif coherent and not high_turnover:
            return "nested"
        elif coherent and high_turnover and not clumped:
            return "evenly_spaced"
        elif coherent and high_turnover and clumped:
            return "clumped"
        else:
            return "intermediate"
    
    def _interpret_structure(self, structure_type: str) -> str:
        """Provide interpretation of metacommunity structure."""
        interpretations = {
            "random": "Species distributions are random with respect to environmental gradients",
            "nested": "Species-poor sites are subsets of species-rich sites",
            "evenly_spaced": "Species have non-overlapping, evenly distributed ranges",
            "clumped": "Species ranges are clustered, suggesting habitat specialization",
            "intermediate": "Intermediate pattern between major structure types"
        }
        return interpretations.get(structure_type, "Unknown structure type")


class NetworkAnalyzer:
    """
    Network analysis for ecological interactions and co-occurrence patterns.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize NetworkAnalyzer."""
        self.random_state = random_state
        
    def build_cooccurrence_network(self, 
                                 community_data: pd.DataFrame,
                                 method: str = 'correlation',
                                 threshold: float = 0.5) -> Dict[str, Any]:
        """
        Build species co-occurrence network.
        
        Parameters
        ----------
        community_data : pd.DataFrame
            Community composition data (sites x species)
        method : str, optional
            Method for calculating associations, by default 'correlation'
        threshold : float, optional
            Threshold for network edges, by default 0.5
            
        Returns
        -------
        Dict[str, Any]
            Network analysis results
        """
# Copyright (c) 2025 Mohamed Z. Hatim
        if method == 'correlation':
            associations = community_data.corr()
        elif method == 'jaccard':
            associations = self._calculate_jaccard_matrix(community_data)
        else:
            raise ValueError(f"Unknown method: {method}")
        
# Copyright (c) 2025 Mohamed Z. Hatim
        np.fill_diagonal(associations.values, 0)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        adjacency = (np.abs(associations) >= threshold).astype(int)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        network_props = self._calculate_network_properties(adjacency, associations)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        network_obj = None
        if NETWORKX_AVAILABLE:
            network_obj = self._create_networkx_graph(associations, threshold)
            network_props.update(self._calculate_networkx_properties(network_obj))
        
        return {
            'associations': associations,
            'adjacency_matrix': pd.DataFrame(adjacency, 
                                           index=associations.index, 
                                           columns=associations.columns),
            'network_properties': network_props,
            'network_object': network_obj,
            'method': method,
            'threshold': threshold
        }
    
    def _calculate_jaccard_matrix(self, community_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Jaccard similarity matrix."""
        pa_data = (community_data > 0).astype(int)
        n_species = pa_data.shape[1]
        jaccard_matrix = np.zeros((n_species, n_species))
        
        for i in range(n_species):
            for j in range(n_species):
                if i == j:
                    jaccard_matrix[i, j] = 1.0
                    continue
                
                species_i = pa_data.iloc[:, i]
                species_j = pa_data.iloc[:, j]
                
                intersection = np.sum(species_i & species_j)
                union = np.sum(species_i | species_j)
                
                jaccard_matrix[i, j] = intersection / union if union > 0 else 0
        
        return pd.DataFrame(jaccard_matrix, 
                          index=community_data.columns, 
                          columns=community_data.columns)
    
    def _calculate_network_properties(self, adjacency: Union[np.ndarray, pd.DataFrame], 
                                    associations: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic network properties."""
# Copyright (c) 2025 Mohamed Z. Hatim
        if isinstance(adjacency, pd.DataFrame):
            adjacency_values = adjacency.values
        else:
            adjacency_values = adjacency
            
        n_nodes = adjacency_values.shape[0]
        n_edges = np.sum(adjacency_values) // 2  # Undirected network
        
# Copyright (c) 2025 Mohamed Z. Hatim
        degrees = np.sum(adjacency_values, axis=1)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        max_edges = n_nodes * (n_nodes - 1) // 2
        density = n_edges / max_edges if max_edges > 0 else 0
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if GRAPH_METHODS_AVAILABLE:
# Copyright (c) 2025 Mohamed Z. Hatim
            distance_matrix = 1 / (np.abs(associations.values) + 1e-10)
            np.fill_diagonal(distance_matrix, 0)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            dist_matrix = dijkstra(distance_matrix, directed=False)
            finite_distances = dist_matrix[np.isfinite(dist_matrix) & (dist_matrix > 0)]
            avg_path_length = np.mean(finite_distances) if len(finite_distances) > 0 else np.inf
        else:
            avg_path_length = np.nan
        
        return {
            'n_nodes': int(n_nodes),
            'n_edges': int(n_edges),
            'density': float(density),
            'mean_degree': float(np.mean(degrees)),
            'degree_distribution': degrees.tolist(),  # Convert to list to avoid pandas Series issues
            'avg_path_length': float(avg_path_length) if not np.isnan(avg_path_length) else np.nan
        }
    
    def _create_networkx_graph(self, associations: pd.DataFrame, 
                              threshold: float) -> 'nx.Graph':
        """Create NetworkX graph object."""
        if not NETWORKX_AVAILABLE:
            return None
        
        G = nx.Graph()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        for species in associations.index:
            G.add_node(species)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        for i, species1 in enumerate(associations.index):
            for species2 in associations.index[i+1:]:
                association_strength = abs(associations.loc[species1, species2])
                if association_strength >= threshold:
                    G.add_edge(species1, species2, weight=association_strength)
        
        return G
    
    def _calculate_networkx_properties(self, G: 'nx.Graph') -> Dict[str, Any]:
        """Calculate advanced network properties using NetworkX."""
        if not NETWORKX_AVAILABLE or G is None:
            return {}
        
        props = {}
        
        try:
# Copyright (c) 2025 Mohamed Z. Hatim
            props['clustering_coefficient'] = nx.average_clustering(G)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            props['betweenness_centrality'] = nx.betweenness_centrality(G)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            try:
                props['eigenvector_centrality'] = nx.eigenvector_centrality(G, max_iter=1000)
            except:
                props['eigenvector_centrality'] = {}
            
# Copyright (c) 2025 Mohamed Z. Hatim
            props['n_components'] = nx.number_connected_components(G)
            props['largest_component_size'] = len(max(nx.connected_components(G), key=len))
            
# Copyright (c) 2025 Mohamed Z. Hatim
            if nx.is_connected(G):
                props['average_shortest_path'] = nx.average_shortest_path_length(G)
                
# Copyright (c) 2025 Mohamed Z. Hatim
                random_G = nx.erdos_renyi_graph(G.number_of_nodes(), 
                                              G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1) / 2))
                if nx.is_connected(random_G):
                    props['small_world_sigma'] = (props['clustering_coefficient'] / nx.average_clustering(random_G)) / (props['average_shortest_path'] / nx.average_shortest_path_length(random_G))
            
        except Exception as e:
            warnings.warn(f"Some network properties could not be calculated: {str(e)}")
        
        return props


class CommunityAssemblyAnalyzer:
    """
    Analysis of community assembly processes and null models.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize CommunityAssemblyAnalyzer."""
        self.random_state = random_state
        np.random.seed(random_state)
        
    def assembly_process_analysis(self, 
                                community_data: pd.DataFrame,
                                trait_data: pd.DataFrame = None,
                                phylo_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Analyze community assembly processes.
        
        Parameters
        ----------
        community_data : pd.DataFrame
            Community composition data (sites x species)
        trait_data : pd.DataFrame, optional
            Species trait data
        phylo_data : pd.DataFrame, optional
            Phylogenetic distance matrix
            
        Returns
        -------
        Dict[str, Any]
            Assembly process analysis results
        """
        results = {}
        
# Copyright (c) 2025 Mohamed Z. Hatim
        taxonomic_patterns = self._analyze_taxonomic_patterns(community_data)
        results['taxonomic'] = taxonomic_patterns
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if trait_data is not None:
            trait_patterns = self._analyze_trait_patterns(community_data, trait_data)
            results['trait_based'] = trait_patterns
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if phylo_data is not None:
            phylo_patterns = self._analyze_phylogenetic_patterns(community_data, phylo_data)
            results['phylogenetic'] = phylo_patterns
        
# Copyright (c) 2025 Mohamed Z. Hatim
        results['interpretation'] = self._interpret_assembly_processes(results)
        
        return results
    
    def _analyze_taxonomic_patterns(self, community_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze taxonomic diversity patterns."""
# Copyright (c) 2025 Mohamed Z. Hatim
        richness = community_data.sum(axis=1)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        evenness = []
        for site in community_data.index:
            site_data = community_data.loc[site]
            present_species = site_data[site_data > 0]
            if len(present_species) > 1:
# Copyright (c) 2025 Mohamed Z. Hatim
                proportions = present_species / present_species.sum()
                shannon = -np.sum(proportions * np.log(proportions))
                max_shannon = np.log(len(present_species))
                evenness.append(shannon / max_shannon if max_shannon > 0 else 0)
            else:
                evenness.append(0)
        
        return {
            'richness': richness,
            'evenness': pd.Series(evenness, index=community_data.index),
            'total_richness': (community_data.sum(axis=0) > 0).sum(),
            'mean_richness': richness.mean(),
            'richness_std': richness.std()
        }
    
    def _analyze_trait_patterns(self, community_data: pd.DataFrame, 
                               trait_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trait-based assembly patterns."""
# Copyright (c) 2025 Mohamed Z. Hatim
        common_species = list(set(community_data.columns) & set(trait_data.index))
        if len(common_species) == 0:
            return {'error': 'No common species between community and trait data'}
        
        community_subset = community_data[common_species]
        trait_subset = trait_data.loc[common_species]
        
# Copyright (c) 2025 Mohamed Z. Hatim
        cwm_traits = []
        trait_variance = []
        
        for site in community_subset.index:
            site_abundances = community_subset.loc[site]
            present_species = site_abundances[site_abundances > 0].index
            
            if len(present_species) == 0:
                continue
            
            weights = site_abundances.loc[present_species]
            weights = weights / weights.sum()
            
            site_traits = trait_subset.loc[present_species]
            
# Copyright (c) 2025 Mohamed Z. Hatim
            cwm_site = {}
            variance_site = {}
            
            for trait in site_traits.select_dtypes(include=[np.number]).columns:
                trait_values = site_traits[trait].fillna(site_traits[trait].mean())
                cwm_site[trait] = (trait_values * weights).sum()
                
# Copyright (c) 2025 Mohamed Z. Hatim
                variance_site[trait] = ((trait_values - cwm_site[trait])**2 * weights).sum()
            
            cwm_traits.append(cwm_site)
            trait_variance.append(variance_site)
        
        cwm_df = pd.DataFrame(cwm_traits, index=community_subset.index[:len(cwm_traits)])
        variance_df = pd.DataFrame(trait_variance, index=community_subset.index[:len(trait_variance)])
        
        return {
            'cwm_traits': cwm_df,
            'trait_variance': variance_df,
            'species_used': common_species
        }
    
    def _analyze_phylogenetic_patterns(self, community_data: pd.DataFrame,
                                     phylo_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze phylogenetic assembly patterns."""
# Copyright (c) 2025 Mohamed Z. Hatim
        phylo_analyzer = PhylogeneticDiversityAnalyzer()
        phylo_analyzer.load_phylogeny(phylo_data)
        
        try:
            phylo_results = phylo_analyzer.calculate_phylogenetic_diversity(community_data)
            return phylo_results
        except Exception as e:
            return {'error': f'Phylogenetic analysis failed: {str(e)}'}
    
    def _interpret_assembly_processes(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Interpret assembly processes based on patterns."""
        interpretation = {}
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if 'taxonomic' in results:
            richness_cv = results['taxonomic']['richness_std'] / results['taxonomic']['mean_richness']
            if richness_cv > 0.5:
                interpretation['richness_pattern'] = "High variation in species richness suggests environmental filtering"
            else:
                interpretation['richness_pattern'] = "Low variation in species richness suggests neutral assembly"
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if 'trait_based' in results and 'error' not in results['trait_based']:
            cwm_traits = results['trait_based']['cwm_traits']
            if not cwm_traits.empty:
                trait_cv = cwm_traits.std() / cwm_traits.mean()
                high_variation_traits = trait_cv[trait_cv > 0.3].index.tolist()
                if high_variation_traits:
                    interpretation['trait_pattern'] = f"High variation in {', '.join(high_variation_traits)} suggests environmental filtering on these traits"
                else:
                    interpretation['trait_pattern'] = "Low trait variation suggests weak environmental filtering"
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if 'phylogenetic' in results and 'metrics' in results['phylogenetic']:
            phylo_metrics = results['phylogenetic']['metrics']
            if not phylo_metrics.empty and 'nri' in phylo_metrics.columns:
                mean_nri = phylo_metrics['nri'].mean()
                if mean_nri > 1.96:  # Significant clustering
                    interpretation['phylogenetic_pattern'] = "Phylogenetic clustering suggests environmental filtering"
                elif mean_nri < -1.96:  # Significant overdispersion
                    interpretation['phylogenetic_pattern'] = "Phylogenetic overdispersion suggests competitive exclusion"
                else:
                    interpretation['phylogenetic_pattern'] = "Random phylogenetic pattern suggests neutral assembly"
        
        return interpretation


# Copyright (c) 2025 Mohamed Z. Hatim
def quick_phylogenetic_diversity(community_data: pd.DataFrame,
                                phylo_distances: pd.DataFrame) -> Dict[str, Any]:
    """
    Quick phylogenetic diversity analysis.
    
    Parameters
    ----------
    community_data : pd.DataFrame
        Community composition data
    phylo_distances : pd.DataFrame
        Phylogenetic distance matrix
        
    Returns
    -------
    Dict[str, Any]
        Phylogenetic diversity results
    """
    analyzer = PhylogeneticDiversityAnalyzer()
    analyzer.load_phylogeny(phylo_distances)
    return analyzer.calculate_phylogenetic_diversity(community_data)


def quick_metacommunity_analysis(community_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Quick metacommunity structure analysis.
    
    Parameters
    ----------
    community_data : pd.DataFrame
        Community composition data
        
    Returns
    -------
    Dict[str, Any]
        Metacommunity analysis results
    """
    analyzer = MetacommunityAnalyzer()
    return analyzer.elements_of_metacommunity_structure(community_data)


def quick_cooccurrence_network(community_data: pd.DataFrame,
                              threshold: float = 0.5) -> Dict[str, Any]:
    """
    Quick species co-occurrence network analysis.
    
    Parameters
    ----------
    community_data : pd.DataFrame
        Community composition data
    threshold : float, optional
        Correlation threshold for network edges
        
    Returns
    -------
    Dict[str, Any]
        Network analysis results
    """
    analyzer = NetworkAnalyzer()
    return analyzer.build_cooccurrence_network(community_data, threshold=threshold)