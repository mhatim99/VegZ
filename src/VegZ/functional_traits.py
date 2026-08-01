"""
Functional Trait Analysis Module

This module provides comprehensive functional trait analysis for vegetation data,
including trait diversity, functional groups, and trait-environment relationships.

Copyright (c) 2025 Mohamed Z. Hatim
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Any
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from .dataset import _ordered_intersection
from scipy.spatial import ConvexHull, QhullError
from scipy.sparse.csgraph import minimum_spanning_tree

CONVEX_HULL_AVAILABLE = True
NEAREST_NEIGHBORS_AVAILABLE = True


class FunctionalTraitAnalyzer:
    """
    Comprehensive functional trait analyzer for vegetation data.
    
    Provides trait diversity calculations, functional group identification,
    and trait-environment relationship analysis.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the FunctionalTraitAnalyzer.
        
        Parameters
        ----------
        random_state : int, optional
            Random state for reproducibility, by default 42
        """
        self.random_state = random_state
        self.trait_data = None
        self.abundance_data = None
        self.functional_groups = None
        self.trait_diversity_results = {}
        
    def load_trait_data(self, 
                       trait_data: pd.DataFrame,
                       abundance_data: pd.DataFrame = None,
                       species_column: str = 'species') -> None:
        """
        Load trait and abundance data.
        
        Parameters
        ----------
        trait_data : pd.DataFrame
            Species trait data
        abundance_data : pd.DataFrame, optional
            Species abundance data by sites
        species_column : str, optional
            Name of species column, by default 'species'
        """
        self.trait_data = trait_data.set_index(species_column) if species_column in trait_data.columns else trait_data
        self.abundance_data = abundance_data
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if self.abundance_data is not None:
            common_species = _ordered_intersection(self.trait_data.index,
                                                   self.abundance_data.columns)
            if len(common_species) == 0:
                warnings.warn("No common species found between trait and abundance data")
            else:
                dropped = len(self.trait_data.index) - len(common_species)
                if dropped:
                    warnings.warn(
                        f"{dropped} species in the trait table have no abundance "
                        "data and were dropped."
                    )
                self.trait_data = self.trait_data.loc[common_species]
                self.abundance_data = self.abundance_data[common_species]
    
    def load_dataset(self, dataset) -> None:
        """
        Load traits and abundances from a :class:`~VegZ.dataset.VegData`.

        The container has already aligned species labels across the trait and
        abundance tables, so nothing has to be re-intersected here.

        Parameters
        ----------
        dataset : VegData
            Must carry a trait table.
        """
        if dataset.traits is None:
            raise ValueError("VegData has no trait table to load")

        self.trait_data = dataset.traits
        self.abundance_data = dataset.species

    def calculate_functional_diversity(self,
                                     sites: List[str] = None,
                                     traits: List[str] = None,
                                     standardize: bool = True) -> Dict[str, Any]:
        """
        Calculate functional diversity indices.
        
        Parameters
        ----------
        sites : List[str], optional
            List of sites to analyze. If None, analyze all sites
        traits : List[str], optional
            List of traits to use. If None, use all numeric traits
        standardize : bool, optional
            Whether to standardize trait values, by default True
            
        Returns
        -------
        Dict[str, Any]
            Functional diversity results
        """
        if self.trait_data is None:
            raise ValueError("Trait data not loaded. Use load_trait_data() first.")
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if traits is None:
            traits = self.trait_data.select_dtypes(include=[np.number]).columns.tolist()
        
        trait_matrix = self.trait_data[traits].copy()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        trait_matrix = trait_matrix.fillna(trait_matrix.mean())
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if standardize:
            scaler = StandardScaler()
            trait_matrix_scaled = pd.DataFrame(
                scaler.fit_transform(trait_matrix),
                index=trait_matrix.index,
                columns=trait_matrix.columns
            )
        else:
            trait_matrix_scaled = trait_matrix
        
# Copyright (c) 2025 Mohamed Z. Hatim
        trait_distances = pdist(trait_matrix_scaled.values, metric='euclidean')
        trait_dist_matrix = squareform(trait_distances)
        trait_dist_df = pd.DataFrame(
            trait_dist_matrix,
            index=trait_matrix_scaled.index,
            columns=trait_matrix_scaled.index
        )
        
        results = {
            'trait_matrix': trait_matrix_scaled,
            'trait_distances': trait_dist_df,
            'traits_used': traits
        }
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if self.abundance_data is not None:
            if sites is None:
                sites = self.abundance_data.index.tolist()
            
            site_diversity = {}
            for site in sites:
                if site not in self.abundance_data.index:
                    continue
                
                site_abundances = self.abundance_data.loc[site]
                present_species = site_abundances[site_abundances > 0].index.tolist()
                
                if len(present_species) == 0:
                    continue
                
# Copyright (c) 2025 Mohamed Z. Hatim
                site_traits = trait_matrix_scaled.loc[present_species]
                site_weights = site_abundances.loc[present_species]
                site_weights = site_weights / site_weights.sum()  # Normalize
                
# Copyright (c) 2025 Mohamed Z. Hatim
                fd_indices = self._calculate_fd_indices(
                    site_traits, site_weights, trait_dist_df.loc[present_species, present_species]
                )
                site_diversity[site] = fd_indices
            
            results['site_diversity'] = pd.DataFrame(site_diversity).T
        
        self.trait_diversity_results = results
        return results
    
    def _calculate_fd_indices(self,
                              traits: pd.DataFrame,
                              weights: pd.Series,
                              distances: pd.DataFrame) -> Dict[str, float]:
        """
        Functional diversity indices for a single site.

        Follows Villeger, Mason & Mouillot (2008) for FRic, FEve and FDiv,
        Laliberte & Legendre (2010) for FDis, and Rao (1982) for RaoQ.
        """
        indices: Dict[str, float] = {}

        trait_values = np.asarray(traits.values, dtype=float)
        w = np.asarray(weights.values, dtype=float)
        if w.sum() > 0:
            w = w / w.sum()
        dist = np.asarray(distances.values, dtype=float)
        n_species, n_traits = trait_values.shape

        indices['FRic'] = self._functional_richness(trait_values)
        indices['FEve'] = self._functional_evenness(dist, w)
        indices['FDis'], centroid_distances = self._functional_dispersion(trait_values, w)
        indices['FDiv'] = self._functional_divergence(trait_values, w, centroid_distances)
        # Rao's quadratic entropy: w' D w, vectorised.
        indices['RaoQ'] = float(w @ dist @ w) if n_species else 0.0
        indices['n_species'] = float(n_species)

        return indices

    @staticmethod
    def _functional_richness(trait_values: np.ndarray) -> float:
        """
        Convex-hull volume of the species in trait space (FRic).

        Falls back to the product of trait ranges when there are too few
        species to build a hull in the available dimensions.
        """
        n_species, n_traits = trait_values.shape
        if n_species == 0:
            return 0.0
        if n_traits == 1:
            return float(np.ptp(trait_values[:, 0]))

        # A hull in d dimensions needs at least d + 1 points; reduce dimensions
        # rather than give up when species are scarce.
        n_dims = int(min(n_traits, max(1, n_species - 1), 3))
        if n_dims < 2 or n_species < n_dims + 1:
            return float(np.prod(np.ptp(trait_values, axis=0)))

        points = trait_values
        if n_dims < n_traits:
            points = PCA(n_components=n_dims).fit_transform(trait_values)

        try:
            return float(ConvexHull(points).volume)
        except (QhullError, ValueError):
            # Degenerate (co-planar) configuration.
            return float(np.prod(np.ptp(points, axis=0)))

    @staticmethod
    def _functional_evenness(dist: np.ndarray, weights: np.ndarray) -> float:
        """
        Functional evenness (Villeger et al. 2008), bounded in [0, 1].

        Built on the minimum spanning tree of the trait-distance matrix: each
        branch gets a partial weighted evenness ``PEW_l = EW_l / sum(EW)``, and
        ``FEve = (sum_l min(PEW_l, 1/(S-1)) - 1/(S-1)) / (1 - 1/(S-1))``.
        """
        n_species = dist.shape[0]
        if n_species < 3:
            return 0.0

        mst = minimum_spanning_tree(dist).toarray()
        branches = np.argwhere(mst > 0)
        if branches.size == 0:
            return 0.0

        # Branch length weighted by the summed abundance of its two endpoints.
        ew = []
        for i, j in branches:
            weight_sum = weights[i] + weights[j]
            if weight_sum > 0:
                ew.append(mst[i, j] / weight_sum)
        ew = np.asarray(ew, dtype=float)

        if ew.size == 0 or ew.sum() <= 0:
            return 0.0

        pew = ew / ew.sum()
        threshold = 1.0 / (n_species - 1)
        numerator = float(np.sum(np.minimum(pew, threshold)) - threshold)
        denominator = 1.0 - threshold

        return float(np.clip(numerator / denominator, 0.0, 1.0)) if denominator > 0 else 0.0

    @staticmethod
    def _functional_dispersion(trait_values: np.ndarray, weights: np.ndarray):
        """Abundance-weighted mean distance to the weighted centroid (FDis)."""
        if trait_values.shape[0] == 0:
            return 0.0, np.zeros(0)

        centroid = weights @ trait_values
        centroid_distances = np.sqrt(((trait_values - centroid) ** 2).sum(axis=1))
        return float(weights @ centroid_distances), centroid_distances

    @staticmethod
    def _functional_divergence(trait_values: np.ndarray, weights: np.ndarray,
                               centroid_distances: np.ndarray) -> float:
        """
        Functional divergence (Villeger et al. 2008), bounded in [0, 1].

        ``FDiv = (delta_d + dG_bar) / (delta_|d| + dG_bar)``, where distances
        are measured from the centre of gravity of the *vertices* of the trait
        space and ``delta_d`` is the abundance-weighted mean deviation from
        their mean distance. A plain
        ``sum(w * |d - d_bar|) / d_bar`` is a coefficient of variation, not
        FDiv, and is not bounded by 1.
        """
        n_species = trait_values.shape[0]
        if n_species < 3:
            return 0.0

        # Centre of gravity of the vertices; with few species every point is a
        # vertex, which is also what Villeger et al. specify for that case.
        try:
            if trait_values.shape[1] >= 2 and n_species > trait_values.shape[1]:
                vertices = ConvexHull(trait_values).vertices
            else:
                vertices = np.arange(n_species)
        except (QhullError, ValueError):
            vertices = np.arange(n_species)

        centre = trait_values[vertices].mean(axis=0)
        dist_to_centre = np.sqrt(((trait_values - centre) ** 2).sum(axis=1))

        mean_distance = float(dist_to_centre.mean())
        if mean_distance <= 0:
            return 0.0

        deviations = dist_to_centre - mean_distance
        delta_d = float(np.sum(weights * deviations))
        delta_abs_d = float(np.sum(weights * np.abs(deviations)))

        denominator = delta_abs_d + mean_distance
        if denominator <= 0:
            return 0.0

        return float(np.clip((delta_d + mean_distance) / denominator, 0.0, 1.0))
    
    def identify_functional_groups(self,
                                 n_groups: int = None,
                                 traits: List[str] = None,
                                 method: str = 'hierarchical') -> Dict[str, Any]:
        """
        Identify functional groups based on trait similarity.
        
        Parameters
        ----------
        n_groups : int, optional
            Number of functional groups. If None, will be determined automatically
        traits : List[str], optional
            List of traits to use. If None, use all numeric traits
        method : str, optional
            Clustering method ('hierarchical', 'kmeans'), by default 'hierarchical'
            
        Returns
        -------
        Dict[str, Any]
            Functional group results
        """
        if self.trait_data is None:
            raise ValueError("Trait data not loaded. Use load_trait_data() first.")
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if traits is None:
            traits = self.trait_data.select_dtypes(include=[np.number]).columns.tolist()
        
        trait_matrix = self.trait_data[traits].copy()
        trait_matrix = trait_matrix.fillna(trait_matrix.mean())
        
# Copyright (c) 2025 Mohamed Z. Hatim
        scaler = StandardScaler()
        trait_matrix_scaled = pd.DataFrame(
            scaler.fit_transform(trait_matrix),
            index=trait_matrix.index,
            columns=trait_matrix.columns
        )
        
        if method == 'hierarchical':
# Copyright (c) 2025 Mohamed Z. Hatim
            distances = pdist(trait_matrix_scaled.values, metric='euclidean')
            linkage_matrix = linkage(distances, method='ward')
            
# Copyright (c) 2025 Mohamed Z. Hatim
            if n_groups is None:
# Copyright (c) 2025 Mohamed Z. Hatim
                from sklearn.metrics import silhouette_score
                silhouette_scores = []
                K_range = range(2, max(3, min(11, len(trait_matrix) // 2 + 1)))

                for k in K_range:
                    cluster_labels = fcluster(linkage_matrix, k, criterion='maxclust')
                    if len(np.unique(cluster_labels)) > 1:
                        score = silhouette_score(trait_matrix_scaled, cluster_labels)
                        silhouette_scores.append(score)
                    else:
                        silhouette_scores.append(0)
                
                n_groups = list(K_range)[int(np.argmax(silhouette_scores))] if silhouette_scores else 2
            
            cluster_labels = fcluster(linkage_matrix, n_groups, criterion='maxclust')
            
        elif method == 'kmeans':
            if n_groups is None:
# Copyright (c) 2025 Mohamed Z. Hatim
                inertias = []
                K_range = range(2, max(3, min(11, len(trait_matrix) // 2 + 1)))
                for k in K_range:
                    kmeans = KMeans(n_clusters=k, random_state=self.random_state)
                    kmeans.fit(trait_matrix_scaled)
                    inertias.append(kmeans.inertia_)
                
# Copyright (c) 2025 Mohamed Z. Hatim
                # Index i of the double difference maps to K_range[i + 1];
                # a +2 offset overruns the list for small ranges.
                k_list = list(K_range)
                delta_deltas = np.diff(np.diff(inertias))
                if len(delta_deltas) > 0:
                    idx = min(int(np.argmax(delta_deltas)) + 1, len(k_list) - 1)
                    n_groups = k_list[idx]
                else:
                    n_groups = k_list[0] if k_list else 2
            
            kmeans = KMeans(n_clusters=n_groups, random_state=self.random_state)
            cluster_labels = kmeans.fit_predict(trait_matrix_scaled)
            cluster_labels += 1  # Start from 1 instead of 0
            linkage_matrix = None
        
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
# Copyright (c) 2025 Mohamed Z. Hatim
        functional_groups = pd.Series(cluster_labels, index=trait_matrix.index, name='functional_group')
        
# Copyright (c) 2025 Mohamed Z. Hatim
        group_characteristics = {}
        for group in range(1, n_groups + 1):
            group_species = functional_groups[functional_groups == group].index
            group_traits = trait_matrix.loc[group_species]
            
            group_characteristics[f'Group_{group}'] = {
                'n_species': len(group_species),
                'species': group_species.tolist(),
                'mean_traits': group_traits.mean(),
                'std_traits': group_traits.std(),
                'trait_ranges': group_traits.max() - group_traits.min()
            }
        
        results = {
            'functional_groups': functional_groups,
            'group_characteristics': group_characteristics,
            'n_groups': n_groups,
            'linkage_matrix': linkage_matrix,
            'trait_matrix_scaled': trait_matrix_scaled,
            'traits_used': traits,
            'method': method
        }
        
        self.functional_groups = results
        return results
    
    def trait_environment_relationships(self,
                                      environmental_data: pd.DataFrame,
                                      traits: List[str] = None,
                                      env_variables: List[str] = None) -> Dict[str, Any]:
        """
        Analyze relationships between traits and environmental variables.
        
        Parameters
        ----------
        environmental_data : pd.DataFrame
            Environmental data by sites
        traits : List[str], optional
            List of traits to analyze. If None, use all numeric traits
        env_variables : List[str], optional
            List of environmental variables. If None, use all numeric columns
            
        Returns
        -------
        Dict[str, Any]
            Trait-environment relationship results
        """
        if self.trait_data is None or self.abundance_data is None:
            raise ValueError("Both trait and abundance data required. Use load_trait_data() first.")
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if traits is None:
            traits = self.trait_data.select_dtypes(include=[np.number]).columns.tolist()
        if env_variables is None:
            env_variables = environmental_data.select_dtypes(include=[np.number]).columns.tolist()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        cwm_traits = self._calculate_cwm_traits(traits)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        common_sites = _ordered_intersection(cwm_traits.index, environmental_data.index)
        if len(common_sites) == 0:
            raise ValueError("No common sites found between CWM traits and environmental data")

        cwm_traits_common = cwm_traits.loc[common_sites]
        env_data_common = environmental_data.loc[common_sites, env_variables]
        
# Copyright (c) 2025 Mohamed Z. Hatim
        correlations = {}
        p_values = {}
        
        for trait in traits:
            correlations[trait] = {}
            p_values[trait] = {}
            
            for env_var in env_variables:
                if trait in cwm_traits_common.columns and env_var in env_data_common.columns:
                    corr, p_val = stats.pearsonr(
                        cwm_traits_common[trait].fillna(cwm_traits_common[trait].mean()),
                        env_data_common[env_var].fillna(env_data_common[env_var].mean())
                    )
                    correlations[trait][env_var] = corr
                    p_values[trait][env_var] = p_val
        
# Copyright (c) 2025 Mohamed Z. Hatim
        corr_df = pd.DataFrame(correlations).T
        pval_df = pd.DataFrame(p_values).T
        
# Copyright (c) 2025 Mohamed Z. Hatim
        rda_results = None
        try:
            rda_results = self._perform_rda(cwm_traits_common, env_data_common)
        except Exception as e:
            warnings.warn(f"RDA analysis failed: {str(e)}")
        
        results = {
            'cwm_traits': cwm_traits_common,
            'environmental_data': env_data_common,
            'correlations': corr_df,
            'p_values': pval_df,
            'significant_correlations': corr_df[pval_df < 0.05],
            'rda_results': rda_results
        }
        
        return results
    
    def _calculate_cwm_traits(self, traits: List[str]) -> pd.DataFrame:
        """Calculate community-weighted mean traits."""
        cwm_traits = []
        
        for site in self.abundance_data.index:
            site_abundances = self.abundance_data.loc[site]
            present_species = site_abundances[site_abundances > 0].index.tolist()
            
            if len(present_species) == 0:
                cwm_traits.append({trait: np.nan for trait in traits})
                continue
            
# Copyright (c) 2025 Mohamed Z. Hatim
            site_traits = self.trait_data.loc[present_species, traits]
            site_weights = site_abundances.loc[present_species]
            site_weights = site_weights / site_weights.sum()  # Normalize
            
# Copyright (c) 2025 Mohamed Z. Hatim
            cwm_site = {}
            for trait in traits:
                if trait in site_traits.columns:
                    trait_values = site_traits[trait].fillna(site_traits[trait].mean())
                    cwm_site[trait] = (trait_values * site_weights).sum()
                else:
                    cwm_site[trait] = np.nan
            
            cwm_traits.append(cwm_site)
        
        return pd.DataFrame(cwm_traits, index=self.abundance_data.index)
    
    def _perform_rda(self, traits: pd.DataFrame, env_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform Redundancy Analysis (simplified version using PCA)."""
        from sklearn.linear_model import LinearRegression
        from sklearn.decomposition import PCA
        
# Copyright (c) 2025 Mohamed Z. Hatim
        complete_data = pd.concat([traits, env_data], axis=1).dropna()
        if len(complete_data) == 0:
            return None
        
        traits_complete = complete_data[traits.columns]
        env_complete = complete_data[env_data.columns]
        
# Copyright (c) 2025 Mohamed Z. Hatim
        scaler_traits = StandardScaler()
        scaler_env = StandardScaler()
        
        traits_scaled = scaler_traits.fit_transform(traits_complete)
        env_scaled = scaler_env.fit_transform(env_complete)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        explained_variance = []
        
        for i in range(traits_scaled.shape[1]):
            reg = LinearRegression()
            reg.fit(env_scaled, traits_scaled[:, i])
            predicted = reg.predict(env_scaled)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            ss_res = np.sum((traits_scaled[:, i] - predicted) ** 2)
            ss_tot = np.sum((traits_scaled[:, i] - np.mean(traits_scaled[:, i])) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            explained_variance.append(r2)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        predicted_traits = np.column_stack([
            LinearRegression().fit(env_scaled, traits_scaled[:, i]).predict(env_scaled)
            for i in range(traits_scaled.shape[1])
        ])
        
        pca = PCA()
        canonical_scores = pca.fit_transform(predicted_traits)
        
        return {
            'explained_variance': dict(zip(traits.columns, explained_variance)),
            'total_explained_variance': np.mean(explained_variance),
            'canonical_axes': pca.components_,
            'canonical_scores': canonical_scores,
            'eigenvalues': pca.explained_variance_ratio_
        }
    
    def calculate_functional_beta_diversity(self,
                                          sites: List[str] = None,
                                          traits: List[str] = None) -> Dict[str, Any]:
        """
        Calculate functional beta diversity between sites.
        
        Parameters
        ----------
        sites : List[str], optional
            List of sites to analyze. If None, use all sites
        traits : List[str], optional
            List of traits to use. If None, use all numeric traits
            
        Returns
        -------
        Dict[str, Any]
            Functional beta diversity results
        """
        if self.trait_data is None or self.abundance_data is None:
            raise ValueError("Both trait and abundance data required.")
        
        if sites is None:
            sites = self.abundance_data.index.tolist()
        if traits is None:
            traits = self.trait_data.select_dtypes(include=[np.number]).columns.tolist()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        fd_results = self.calculate_functional_diversity(sites, traits)
        
        if 'site_diversity' not in fd_results:
            raise ValueError("Site-level functional diversity calculation failed")
        
        site_fd = fd_results['site_diversity']
        
# Copyright (c) 2025 Mohamed Z. Hatim
        beta_diversity = {}
        
# Copyright (c) 2025 Mohamed Z. Hatim
        all_species = []
        all_abundances = []
        
        for site in sites:
            if site in self.abundance_data.index:
                site_abundances = self.abundance_data.loc[site]
                present_species = site_abundances[site_abundances > 0].index.tolist()
                all_species.extend(present_species)
                all_abundances.extend(site_abundances.loc[present_species].tolist())
        
        if all_species:
# Copyright (c) 2025 Mohamed Z. Hatim
            pooled_abundances = pd.Series(all_abundances, index=all_species)
            pooled_abundances = pooled_abundances.groupby(pooled_abundances.index).sum()
            pooled_abundances = pooled_abundances / pooled_abundances.sum()
            
# Copyright (c) 2025 Mohamed Z. Hatim
            pooled_traits = self.trait_data.loc[pooled_abundances.index, traits]
            pooled_traits = pooled_traits.fillna(pooled_traits.mean())
            
# Copyright (c) 2025 Mohamed Z. Hatim
            scaler = StandardScaler()
            pooled_traits_scaled = pd.DataFrame(
                scaler.fit_transform(pooled_traits),
                index=pooled_traits.index,
                columns=pooled_traits.columns
            )
            
            pooled_distances = pdist(pooled_traits_scaled.values, metric='euclidean')
            pooled_dist_matrix = squareform(pooled_distances)
            pooled_dist_df = pd.DataFrame(
                pooled_dist_matrix,
                index=pooled_traits_scaled.index,
                columns=pooled_traits_scaled.index
            )
            
            gamma_fd = self._calculate_fd_indices(
                pooled_traits_scaled, pooled_abundances, pooled_dist_df
            )
        else:
            # No species anywhere: there is no gamma diversity to report.
            gamma_fd = {}

        mean_alpha_fd = site_fd.mean().to_dict()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        for index in gamma_fd:
            if index in mean_alpha_fd:
                beta_diversity[f'beta_{index}'] = gamma_fd[index] - mean_alpha_fd[index]
        
        return {
            'gamma_diversity': gamma_fd,
            'mean_alpha_diversity': mean_alpha_fd,
            'beta_diversity': beta_diversity,
            'site_diversity': site_fd,
            'sites_analyzed': sites
        }
    
    def plot_functional_space(self, 
                            traits: List[str] = None,
                            color_by: str = None,
                            n_components: int = 2) -> plt.Figure:
        """
        Plot functional space using PCA.
        
        Parameters
        ----------
        traits : List[str], optional
            Traits to use for PCA. If None, use all numeric traits
        color_by : str, optional
            Variable to color points by ('functional_group', etc.)
        n_components : int, optional
            Number of PCA components to plot, by default 2
            
        Returns
        -------
        plt.Figure
            Functional space plot
        """
        if self.trait_data is None:
            raise ValueError("Trait data not loaded.")
        
        if traits is None:
            traits = self.trait_data.select_dtypes(include=[np.number]).columns.tolist()
        
        trait_matrix = self.trait_data[traits].fillna(self.trait_data[traits].mean())
        
# Copyright (c) 2025 Mohamed Z. Hatim
        scaler = StandardScaler()
        trait_matrix_scaled = scaler.fit_transform(trait_matrix)
        
        pca = PCA(n_components=n_components)
        pca_scores = pca.fit_transform(trait_matrix_scaled)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        fig, ax = plt.subplots(figsize=(10, 8))
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if color_by == 'functional_group' and self.functional_groups is not None:
            colors = self.functional_groups['functional_groups']
            scatter = ax.scatter(pca_scores[:, 0], pca_scores[:, 1], c=colors, cmap='tab10')
            plt.colorbar(scatter, label='Functional Group')
        else:
            ax.scatter(pca_scores[:, 0], pca_scores[:, 1], alpha=0.7)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        for i, species in enumerate(trait_matrix.index):
            ax.annotate(species, (pca_scores[i, 0], pca_scores[i, 1]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.set_title('Functional Space (PCA)')
        
        plt.tight_layout()
        return fig
    
    def plot_trait_distributions(self, traits: List[str] = None) -> plt.Figure:
        """
        Plot trait value distributions.
        
        Parameters
        ----------
        traits : List[str], optional
            Traits to plot. If None, use all numeric traits
            
        Returns
        -------
        plt.Figure
            Trait distribution plots
        """
        if self.trait_data is None:
            raise ValueError("Trait data not loaded.")
        
        if traits is None:
            traits = self.trait_data.select_dtypes(include=[np.number]).columns.tolist()
        
        n_traits = len(traits)
        n_cols = min(3, n_traits)
        n_rows = (n_traits + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = axes.flatten() if n_traits > 1 else [axes]
        
        for i, trait in enumerate(traits):
            trait_values = self.trait_data[trait].dropna()
            
            axes[i].hist(trait_values, bins=20, alpha=0.7, edgecolor='black')
            axes[i].set_xlabel(trait)
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Distribution of {trait}')
            
# Copyright (c) 2025 Mohamed Z. Hatim
            mean_val = trait_values.mean()
            axes[i].axvline(mean_val, color='red', linestyle='--', 
                          label=f'Mean: {mean_val:.2f}')
            axes[i].legend()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        for i in range(n_traits, len(axes)):
            axes[i].remove()
        
        plt.tight_layout()
        return fig


class TraitSyndromes:
    """
    Class for analyzing trait syndromes and trade-offs.
    """
    
    def __init__(self, trait_analyzer: FunctionalTraitAnalyzer):
        """Initialize with a FunctionalTraitAnalyzer instance."""
        self.trait_analyzer = trait_analyzer
    
    def identify_trait_syndromes(self, 
                               traits: List[str] = None,
                               method: str = 'pca') -> Dict[str, Any]:
        """
        Identify trait syndromes using multivariate analysis.
        
        Parameters
        ----------
        traits : List[str], optional
            Traits to analyze. If None, use all numeric traits
        method : str, optional
            Analysis method ('pca', 'factor_analysis'), by default 'pca'
            
        Returns
        -------
        Dict[str, Any]
            Trait syndrome results
        """
        if self.trait_analyzer.trait_data is None:
            raise ValueError("Trait data not loaded in analyzer.")
        
        if traits is None:
            traits = self.trait_analyzer.trait_data.select_dtypes(include=[np.number]).columns.tolist()
        
        trait_matrix = self.trait_analyzer.trait_data[traits].fillna(
            self.trait_analyzer.trait_data[traits].mean()
        )
        
# Copyright (c) 2025 Mohamed Z. Hatim
        scaler = StandardScaler()
        trait_matrix_scaled = pd.DataFrame(
            scaler.fit_transform(trait_matrix),
            index=trait_matrix.index,
            columns=trait_matrix.columns
        )
        
        if method == 'pca':
            pca = PCA()
            pca_scores = pca.fit_transform(trait_matrix_scaled)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            significant_pcs = (pca.explained_variance_ > 1) | (pca.explained_variance_ratio_ > 0.05)
            n_significant = np.sum(significant_pcs)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            loadings = pd.DataFrame(
                pca.components_[:n_significant].T,
                index=traits,
                columns=[f'PC{i+1}' for i in range(n_significant)]
            )
            
# Copyright (c) 2025 Mohamed Z. Hatim
            syndromes = {}
            for pc in loadings.columns:
                high_positive = loadings[loadings[pc] > 0.6][pc].index.tolist()
                high_negative = loadings[loadings[pc] < -0.6][pc].index.tolist()
                
                syndromes[pc] = {
                    'positive_traits': high_positive,
                    'negative_traits': high_negative,
                    'explained_variance': pca.explained_variance_ratio_[int(pc[2:]) - 1],
                    'interpretation': self._interpret_syndrome(high_positive, high_negative)
                }
            
            results = {
                'method': 'pca',
                'loadings': loadings,
                'scores': pd.DataFrame(pca_scores[:, :n_significant], 
                                     index=trait_matrix.index,
                                     columns=[f'PC{i+1}' for i in range(n_significant)]),
                'syndromes': syndromes,
                'explained_variance_ratio': pca.explained_variance_ratio_[:n_significant],
                'total_variance_explained': np.sum(pca.explained_variance_ratio_[:n_significant])
            }
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return results
    
    def _interpret_syndrome(self, positive_traits: List[str], negative_traits: List[str]) -> str:
        """Provide biological interpretation of trait syndrome."""
# Copyright (c) 2025 Mohamed Z. Hatim
        interpretations = {
            'acquisitive': ['leaf_area', 'sla', 'leaf_n', 'leaf_p'],
            'conservative': ['leaf_thickness', 'ldmc', 'wood_density'],
            'size': ['plant_height', 'leaf_area', 'seed_mass'],
            'reproductive': ['seed_mass', 'seed_number', 'reproductive_height']
        }
        
# Copyright (c) 2025 Mohamed Z. Hatim
        for syndrome_name, syndrome_traits in interpretations.items():
            if len(set(positive_traits) & set(syndrome_traits)) >= 2:
                return f"Likely represents {syndrome_name} strategy"
        
        return "Syndrome interpretation unclear - manual interpretation needed"
    
    def analyze_trait_trade_offs(self, 
                               trait_pairs: List[Tuple[str, str]] = None) -> Dict[str, Any]:
        """
        Analyze trade-offs between trait pairs.
        
        Parameters
        ----------
        trait_pairs : List[Tuple[str, str]], optional
            Specific trait pairs to analyze. If None, analyze all pairs
            
        Returns
        -------
        Dict[str, Any]
            Trade-off analysis results
        """
        if self.trait_analyzer.trait_data is None:
            raise ValueError("Trait data not loaded in analyzer.")
        
        numeric_traits = self.trait_analyzer.trait_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if trait_pairs is None:
# Copyright (c) 2025 Mohamed Z. Hatim
            trait_pairs = [(t1, t2) for i, t1 in enumerate(numeric_traits) 
                          for t2 in numeric_traits[i+1:]]
        
        trade_offs = {}
        
        for trait1, trait2 in trait_pairs:
            if trait1 not in self.trait_analyzer.trait_data.columns or trait2 not in self.trait_analyzer.trait_data.columns:
                continue
            
# Copyright (c) 2025 Mohamed Z. Hatim
            data_subset = self.trait_analyzer.trait_data[[trait1, trait2]].dropna()
            
            if len(data_subset) < 3:
                continue
            
# Copyright (c) 2025 Mohamed Z. Hatim
            corr, p_value = stats.pearsonr(data_subset[trait1], data_subset[trait2])
            
# Copyright (c) 2025 Mohamed Z. Hatim
            if p_value < 0.05:
                if corr < -0.3:
                    trade_off_type = "Strong trade-off"
                elif corr < -0.1:
                    trade_off_type = "Weak trade-off"
                elif corr > 0.3:
                    trade_off_type = "Strong synergy"
                elif corr > 0.1:
                    trade_off_type = "Weak synergy"
                else:
                    trade_off_type = "No clear relationship"
            else:
                trade_off_type = "No significant relationship"
            
            trade_offs[f"{trait1}_vs_{trait2}"] = {
                'correlation': corr,
                'p_value': p_value,
                'n_observations': len(data_subset),
                'trade_off_type': trade_off_type,
                'trait1_mean': data_subset[trait1].mean(),
                'trait2_mean': data_subset[trait2].mean()
            }
        
        return {
            'trade_offs': trade_offs,
            'summary': self._summarize_trade_offs(trade_offs)
        }
    
    def _summarize_trade_offs(self, trade_offs: Dict[str, Any]) -> Dict[str, int]:
        """Summarize trade-off analysis results."""
        summary = {
            'strong_trade_offs': 0,
            'weak_trade_offs': 0,
            'strong_synergies': 0,
            'weak_synergies': 0,
            'no_relationship': 0
        }
        
        for analysis in trade_offs.values():
            trade_off_type = analysis['trade_off_type']
            if 'Strong trade-off' in trade_off_type:
                summary['strong_trade_offs'] += 1
            elif 'Weak trade-off' in trade_off_type:
                summary['weak_trade_offs'] += 1
            elif 'Strong synergy' in trade_off_type:
                summary['strong_synergies'] += 1
            elif 'Weak synergy' in trade_off_type:
                summary['weak_synergies'] += 1
            else:
                summary['no_relationship'] += 1
        
        return summary


# Copyright (c) 2025 Mohamed Z. Hatim
def quick_functional_diversity(trait_data: pd.DataFrame,
                             abundance_data: pd.DataFrame,
                             species_column: str = 'species') -> Dict[str, Any]:
    """
    Quick functional diversity analysis.
    
    Parameters
    ----------
    trait_data : pd.DataFrame
        Species trait data
    abundance_data : pd.DataFrame
        Species abundance data by sites
    species_column : str, optional
        Name of species column, by default 'species'
        
    Returns
    -------
    Dict[str, Any]
        Functional diversity results
    """
    analyzer = FunctionalTraitAnalyzer()
    analyzer.load_trait_data(trait_data, abundance_data, species_column)
    return analyzer.calculate_functional_diversity()


def quick_functional_groups(trait_data: pd.DataFrame,
                          n_groups: int = None) -> Dict[str, Any]:
    """
    Quick functional group identification.
    
    Parameters
    ----------
    trait_data : pd.DataFrame
        Species trait data
    n_groups : int, optional
        Number of functional groups
        
    Returns
    -------
    Dict[str, Any]
        Functional group results
    """
    analyzer = FunctionalTraitAnalyzer()
    analyzer.load_trait_data(trait_data)
    return analyzer.identify_functional_groups(n_groups)