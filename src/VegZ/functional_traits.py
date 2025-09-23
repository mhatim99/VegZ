"""
Functional Trait Analysis Module

This module provides comprehensive functional trait analysis for vegetation data,
including trait diversity, functional groups, and trait-environment relationships.

Copyright (c) 2025 Mohamed Z. Hatim
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from scipy.spatial import ConvexHull
    CONVEX_HULL_AVAILABLE = True
except ImportError:
    CONVEX_HULL_AVAILABLE = False
    warnings.warn("ConvexHull not available, some functional diversity metrics will be limited")

try:
    from sklearn.neighbors import NearestNeighbors
    NEAREST_NEIGHBORS_AVAILABLE = True
except ImportError:
    NEAREST_NEIGHBORS_AVAILABLE = False
    warnings.warn("NearestNeighbors not available, some trait analysis methods will be limited")


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
            common_species = set(self.trait_data.index) & set(self.abundance_data.columns)
            if len(common_species) == 0:
                warnings.warn("No common species found between trait and abundance data")
            else:
                self.trait_data = self.trait_data.loc[list(common_species)]
                self.abundance_data = self.abundance_data[list(common_species)]
    
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
        """Calculate functional diversity indices for a single site."""
        indices = {}
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if len(traits) >= len(traits.columns) + 1 and CONVEX_HULL_AVAILABLE:
            try:
                if len(traits) > 3:
# Copyright (c) 2025 Mohamed Z. Hatim
                    pca = PCA(n_components=min(3, len(traits.columns)))
                    traits_pca = pca.fit_transform(traits.values)
                    hull = ConvexHull(traits_pca)
                    indices['FRic'] = hull.volume
                else:
                    hull = ConvexHull(traits.values)
                    indices['FRic'] = hull.volume if traits.shape[1] >= 2 else 0
            except:
                indices['FRic'] = 0
        else:
# Copyright (c) 2025 Mohamed Z. Hatim
            trait_ranges = traits.max() - traits.min()
            indices['FRic'] = trait_ranges.prod()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if len(traits) > 1:
# Copyright (c) 2025 Mohamed Z. Hatim
            try:
                mst_distances = []
                for i in range(len(traits)):
                    min_dist = float('inf')
                    for j in range(len(traits)):
                        if i != j:
                            min_dist = min(min_dist, distances.iloc[i, j])
                    if min_dist != float('inf'):
                        mst_distances.append(min_dist * weights.iloc[i])
                
                if mst_distances:
                    partial_weighted_evenness = np.array(mst_distances)
                    S = len(traits)
                    EW = min(partial_weighted_evenness)
                    indices['FEve'] = (sum(min(partial_weighted_evenness) - partial_weighted_evenness) / 
                                     (sum(min(partial_weighted_evenness) - partial_weighted_evenness) + 
                                      (S - 1) * EW)) if S > 1 else 0
                else:
                    indices['FEve'] = 0
            except:
                indices['FEve'] = 0
        else:
            indices['FEve'] = 0
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if len(traits) > 0:
# Copyright (c) 2025 Mohamed Z. Hatim
            cwm_traits = (traits * weights.values.reshape(-1, 1)).sum(axis=0)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            centroid_distances = np.sqrt(((traits - cwm_traits) ** 2).sum(axis=1))
            
# Copyright (c) 2025 Mohamed Z. Hatim
            mean_dist = (centroid_distances * weights).sum()
            
# Copyright (c) 2025 Mohamed Z. Hatim
            abs_deviations = np.abs(centroid_distances - mean_dist)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            if mean_dist > 0:
                indices['FDiv'] = (abs_deviations * weights).sum() / mean_dist
            else:
                indices['FDiv'] = 0
        else:
            indices['FDiv'] = 0
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if len(traits) > 0:
            cwm_traits = (traits * weights.values.reshape(-1, 1)).sum(axis=0)
            centroid_distances = np.sqrt(((traits - cwm_traits) ** 2).sum(axis=1))
            indices['FDis'] = (centroid_distances * weights).sum()
        else:
            indices['FDis'] = 0
        
# Copyright (c) 2025 Mohamed Z. Hatim
        indices['RaoQ'] = 0
        for i in range(len(weights)):
            for j in range(len(weights)):
                indices['RaoQ'] += weights.iloc[i] * weights.iloc[j] * distances.iloc[i, j]
        
        return indices
    
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
                K_range = range(2, min(11, len(trait_matrix) // 2))
                
                for k in K_range:
                    cluster_labels = fcluster(linkage_matrix, k, criterion='maxclust')
                    if len(np.unique(cluster_labels)) > 1:
                        score = silhouette_score(trait_matrix_scaled, cluster_labels)
                        silhouette_scores.append(score)
                    else:
                        silhouette_scores.append(0)
                
                n_groups = K_range[np.argmax(silhouette_scores)] if silhouette_scores else 3
            
            cluster_labels = fcluster(linkage_matrix, n_groups, criterion='maxclust')
            
        elif method == 'kmeans':
            if n_groups is None:
# Copyright (c) 2025 Mohamed Z. Hatim
                inertias = []
                K_range = range(2, min(11, len(trait_matrix) // 2))
                for k in K_range:
                    kmeans = KMeans(n_clusters=k, random_state=self.random_state)
                    kmeans.fit(trait_matrix_scaled)
                    inertias.append(kmeans.inertia_)
                
# Copyright (c) 2025 Mohamed Z. Hatim
                deltas = np.diff(inertias)
                delta_deltas = np.diff(deltas)
                n_groups = K_range[np.argmax(delta_deltas) + 2] if len(delta_deltas) > 0 else 3
            
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
        common_sites = set(cwm_traits.index) & set(environmental_data.index)
        if len(common_sites) == 0:
            raise ValueError("No common sites found between CWM traits and environmental data")
        
        cwm_traits_common = cwm_traits.loc[list(common_sites)]
        env_data_common = environmental_data.loc[list(common_sites), env_variables]
        
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
        canonical_axes = []
        
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
        
# Copyright (c) 2025 Mohamed Z. Hatim
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
            std_val = trait_values.std()
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