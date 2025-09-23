"""
Visualization Module

This module provides static visualization functions for vegetation data analysis results
using matplotlib and seaborn. For interactive visualizations, see interactive_viz.py.

Copyright (c) 2025 Mohamed Z. Hatim
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from scipy.cluster.hierarchy import dendrogram
from matplotlib.patches import Ellipse
from matplotlib.collections import LineCollection

# Copyright (c) 2025 Mohamed Z. Hatim
plt.style.use('default')
sns.set_palette('husl')


class VegetationPlotter:
    """
    Main plotting class for vegetation analysis visualizations.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize the VegetationPlotter.
        
        Parameters
        ----------
        style : str, optional
            Matplotlib style to use, by default 'seaborn-v0_8'
        figsize : Tuple[int, int], optional
            Default figure size, by default (10, 8)
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
            warnings.warn(f"Style '{style}' not available, using default")
        
        self.figsize = figsize
        self.colors = plt.cm.Set1.colors
    
    def plot_diversity_indices(self, 
                             diversity_data: Dict[str, Any],
                             indices: List[str] = None,
                             site_labels: bool = True) -> plt.Figure:
        """
        Plot diversity indices for multiple sites.
        
        Parameters
        ----------
        diversity_data : Dict[str, Any]
            Diversity analysis results
        indices : List[str], optional
            Specific indices to plot
        site_labels : bool, optional
            Whether to show site labels on x-axis
            
        Returns
        -------
        plt.Figure
            Figure object
        """
        if 'diversity_indices' not in diversity_data:
            raise ValueError("No diversity indices found in data")
        
        diversity_df = pd.DataFrame(diversity_data['diversity_indices']).T
        
        if indices is not None:
            available_indices = [idx for idx in indices if idx in diversity_df.columns]
            if not available_indices:
                raise ValueError(f"None of the specified indices {indices} found in data")
            diversity_df = diversity_df[available_indices]
        
        n_indices = len(diversity_df.columns)
        n_cols = min(3, n_indices)
        n_rows = (n_indices + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_indices == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, index in enumerate(diversity_df.columns):
            values = diversity_df[index]
            
# Copyright (c) 2025 Mohamed Z. Hatim
            bars = axes[i].bar(range(len(values)), values, alpha=0.7, color=self.colors[i % len(self.colors)])
            
# Copyright (c) 2025 Mohamed Z. Hatim
            for j, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.2f}', ha='center', va='bottom', fontsize=8)
            
            axes[i].set_title(f'{index.title()} Index')
            axes[i].set_ylabel(index.title())
            
            if site_labels and len(values) <= 20:
                axes[i].set_xticks(range(len(values)))
                axes[i].set_xticklabels(values.index, rotation=45, ha='right')
            else:
                axes[i].set_xlabel('Sites')
            
            axes[i].grid(True, alpha=0.3)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            mean_val = values.mean()
            axes[i].axhline(y=mean_val, color='red', linestyle='--', alpha=0.7,
                          label=f'Mean: {mean_val:.2f}')
            axes[i].legend()
        
# Copyright (c) 2025 Mohamed Z. Hatim
        for i in range(n_indices, len(axes)):
            axes[i].remove()
        
        plt.tight_layout()
        return fig
    
    def plot_species_accumulation(self, 
                                accumulation_data: Dict[str, Any],
                                show_ci: bool = True) -> plt.Figure:
        """
        Plot species accumulation curve.
        
        Parameters
        ----------
        accumulation_data : Dict[str, Any]
            Species accumulation results
        show_ci : bool, optional
            Whether to show confidence intervals
            
        Returns
        -------
        plt.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x_vals = range(1, len(accumulation_data['mean']) + 1)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        ax.plot(x_vals, accumulation_data['mean'], 'b-', linewidth=3, label='Observed')
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if show_ci and 'ci_lower' in accumulation_data and 'ci_upper' in accumulation_data:
            ax.fill_between(x_vals, accumulation_data['ci_lower'], accumulation_data['ci_upper'],
                           alpha=0.3, color='blue', label='95% CI')
        
        ax.set_xlabel('Number of Sites')
        ax.set_ylabel('Cumulative Number of Species')
        ax.set_title('Species Accumulation Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_rank_abundance(self, 
                          community_data: pd.DataFrame,
                          site: str = None,
                          log_scale: bool = True) -> plt.Figure:
        """
        Plot rank-abundance curve.
        
        Parameters
        ----------
        community_data : pd.DataFrame
            Community composition data
        site : str, optional
            Specific site to plot. If None, plot total abundance
        log_scale : bool, optional
            Whether to use log scale for y-axis
            
        Returns
        -------
        plt.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if site is not None:
            if site not in community_data.index:
                raise ValueError(f"Site '{site}' not found in data")
            abundances = community_data.loc[site].sort_values(ascending=False)
            abundances = abundances[abundances > 0]
            title = f'Rank-Abundance Curve - {site}'
        else:
            abundances = community_data.sum(axis=0).sort_values(ascending=False)
            abundances = abundances[abundances > 0]
            title = 'Rank-Abundance Curve - Total'
        
        ranks = range(1, len(abundances) + 1)
        
        ax.plot(ranks, abundances, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('Species Rank')
        ax.set_ylabel('Abundance')
        ax.set_title(title)
        
        if log_scale:
            ax.set_yscale('log')
        
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_ordination(self, 
                       ordination_results: Dict[str, Any],
                       axes_to_plot: Tuple[int, int] = (0, 1),
                       color_by: Union[str, pd.Series] = None,
                       show_species: bool = False,
                       show_labels: bool = True) -> plt.Figure:
        """
        Plot ordination results.
        
        Parameters
        ----------
        ordination_results : Dict[str, Any]
            Ordination analysis results
        axes_to_plot : Tuple[int, int], optional
            Which axes to plot, by default (0, 1)
        color_by : Union[str, pd.Series], optional
            Variable to color points by
        show_species : bool, optional
            Whether to show species arrows/points
        show_labels : bool, optional
            Whether to show site labels
            
        Returns
        -------
        plt.Figure
            Figure object
        """
        if 'site_scores' not in ordination_results:
            raise ValueError("Site scores not found in ordination results")
        
        site_scores = ordination_results['site_scores']
        axis1, axis2 = axes_to_plot
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        x = site_scores.iloc[:, axis1]
        y = site_scores.iloc[:, axis2]
        
        if color_by is not None:
            if isinstance(color_by, str):
# Copyright (c) 2025 Mohamed Z. Hatim
                colors = np.arange(len(x))  # Fallback coloring
            else:
                colors = color_by
            
            scatter = ax.scatter(x, y, c=colors, s=80, alpha=0.7, cmap='viridis')
            plt.colorbar(scatter, ax=ax, label='Color Variable')
        else:
            ax.scatter(x, y, s=80, alpha=0.7, color='blue')
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if show_labels:
            for i, site in enumerate(site_scores.index):
                ax.annotate(site, (x.iloc[i], y.iloc[i]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.8)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if show_species and 'species_scores' in ordination_results:
            species_scores = ordination_results['species_scores']
            
# Copyright (c) 2025 Mohamed Z. Hatim
            scale_factor = 0.8 * max(x.max() - x.min(), y.max() - y.min()) / \
                          max(species_scores.iloc[:, axis1].max() - species_scores.iloc[:, axis1].min(),
                              species_scores.iloc[:, axis2].max() - species_scores.iloc[:, axis2].min())
            
            for species in species_scores.index:
                sp_x = species_scores.loc[species].iloc[axis1] * scale_factor
                sp_y = species_scores.loc[species].iloc[axis2] * scale_factor
                
                ax.arrow(0, 0, sp_x, sp_y, head_width=0.02, head_length=0.02,
                        fc='red', ec='red', alpha=0.7)
                ax.text(sp_x * 1.1, sp_y * 1.1, species, fontsize=8, color='red')
        
# Copyright (c) 2025 Mohamed Z. Hatim
        explained_var = ordination_results.get('explained_variance_ratio', [])
        if len(explained_var) > max(axis1, axis2):
            xlabel = f'Axis {axis1 + 1} ({explained_var[axis1]:.1%})'
            ylabel = f'Axis {axis2 + 1} ({explained_var[axis2]:.1%})'
        else:
            xlabel = f'Axis {axis1 + 1}'
            ylabel = f'Axis {axis2 + 1}'
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        method = ordination_results.get('method', 'Ordination')
        ax.set_title(f'{method.upper()} Plot')
        
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        return fig
    
    def plot_dendrogram(self, 
                       clustering_results: Dict[str, Any],
                       labels: bool = True,
                       color_threshold: float = None) -> plt.Figure:
        """
        Plot hierarchical clustering dendrogram.
        
        Parameters
        ----------
        clustering_results : Dict[str, Any]
            Clustering analysis results
        labels : bool, optional
            Whether to show labels
        color_threshold : float, optional
            Color threshold for dendrogram
            
        Returns
        -------
        plt.Figure
            Figure object
        """
        if 'linkage_matrix' not in clustering_results:
            raise ValueError("Linkage matrix not found in clustering results")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        site_labels = clustering_results.get('site_labels', None) if labels else None
        
        dendrogram(clustering_results['linkage_matrix'],
                  labels=site_labels,
                  ax=ax,
                  leaf_rotation=90,
                  leaf_font_size=10,
                  color_threshold=color_threshold)
        
        ax.set_title('Hierarchical Clustering Dendrogram')
        ax.set_ylabel('Distance')
        ax.set_xlabel('Sites')
        
        return fig
    
    def plot_clusters_on_ordination(self, 
                                   ordination_results: Dict[str, Any],
                                   cluster_labels: List[int],
                                   axes_to_plot: Tuple[int, int] = (0, 1)) -> plt.Figure:
        """
        Plot clusters overlaid on ordination space.
        
        Parameters
        ----------
        ordination_results : Dict[str, Any]
            Ordination results
        cluster_labels : List[int]
            Cluster assignments for each site
        axes_to_plot : Tuple[int, int], optional
            Which ordination axes to plot
            
        Returns
        -------
        plt.Figure
            Figure object
        """
        site_scores = ordination_results['site_scores']
        axis1, axis2 = axes_to_plot
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        unique_clusters = sorted(set(cluster_labels))
        colors = self.colors[:len(unique_clusters)]
        
        for i, cluster in enumerate(unique_clusters):
            mask = np.array(cluster_labels) == cluster
            cluster_sites = site_scores.iloc[mask]
            
            ax.scatter(cluster_sites.iloc[:, axis1], cluster_sites.iloc[:, axis2],
                      color=colors[i], label=f'Cluster {cluster}', s=80, alpha=0.7)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        explained_var = ordination_results.get('explained_variance_ratio', [])
        if len(explained_var) > max(axis1, axis2):
            xlabel = f'Axis {axis1 + 1} ({explained_var[axis1]:.1%})'
            ylabel = f'Axis {axis2 + 1} ({explained_var[axis2]:.1%})'
        else:
            xlabel = f'Axis {axis1 + 1}'
            ylabel = f'Axis {axis2 + 1}'
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title('Clusters on Ordination Space')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_environmental_vectors(self, 
                                 ordination_results: Dict[str, Any],
                                 env_vectors: Dict[str, Any],
                                 significance_threshold: float = 0.05) -> plt.Figure:
        """
        Plot environmental vectors on ordination.
        
        Parameters
        ----------
        ordination_results : Dict[str, Any]
            Ordination results
        env_vectors : Dict[str, Any]
            Environmental vector fitting results
        significance_threshold : float, optional
            P-value threshold for showing vectors
            
        Returns
        -------
        plt.Figure
            Figure object
        """
        site_scores = ordination_results['site_scores']
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        ax.scatter(site_scores.iloc[:, 0], site_scores.iloc[:, 1], 
                  alpha=0.7, s=80, color='lightblue', edgecolors='black')
        
# Copyright (c) 2025 Mohamed Z. Hatim
        for var, data in env_vectors.items():
            if data['p_value'] < significance_threshold:
                arrow = data['arrow_coords']
                r2 = data['r_squared']
                
# Copyright (c) 2025 Mohamed Z. Hatim
                scale = 2.0
                ax.arrow(0, 0, arrow[0] * scale, arrow[1] * scale,
                        head_width=0.05, head_length=0.05,
                        fc='red', ec='red', linewidth=2)
                
# Copyright (c) 2025 Mohamed Z. Hatim
                label_pos = np.array(arrow) * scale * 1.2
                ax.text(label_pos[0], label_pos[1], f'{var}\n(R²={r2:.3f})',
                       fontsize=10, fontweight='bold', color='red',
                       ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
# Copyright (c) 2025 Mohamed Z. Hatim
        explained_var = ordination_results.get('explained_variance_ratio', [])
        if len(explained_var) >= 2:
            xlabel = f'Axis 1 ({explained_var[0]:.1%})'
            ylabel = f'Axis 2 ({explained_var[1]:.1%})'
        else:
            xlabel = 'Axis 1'
            ylabel = 'Axis 2'
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title('Environmental Vectors on Ordination')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        return fig
    
    def plot_correlation_matrix(self, 
                              correlation_matrix: pd.DataFrame,
                              method: str = 'pearson') -> plt.Figure:
        """
        Plot correlation matrix heatmap.
        
        Parameters
        ----------
        correlation_matrix : pd.DataFrame
            Correlation matrix
        method : str, optional
            Correlation method used
            
        Returns
        -------
        plt.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
# Copyright (c) 2025 Mohamed Z. Hatim
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.3f', ax=ax,
                   cbar_kws={'label': f'{method.title()} Correlation'})
        
        ax.set_title(f'{method.title()} Correlation Matrix')
        plt.tight_layout()
        
        return fig
    
    def plot_functional_space(self, 
                            trait_data: pd.DataFrame,
                            functional_groups: pd.Series = None,
                            traits_to_plot: List[str] = None) -> plt.Figure:
        """
        Plot functional trait space.
        
        Parameters
        ----------
        trait_data : pd.DataFrame
            Species trait data
        functional_groups : pd.Series, optional
            Functional group assignments
        traits_to_plot : List[str], optional
            Specific traits to plot (uses first 2-3 if not specified)
            
        Returns
        -------
        plt.Figure
            Figure object
        """
        numeric_traits = trait_data.select_dtypes(include=[np.number])
        
        if traits_to_plot is not None:
            traits_subset = numeric_traits[traits_to_plot]
        else:
            traits_subset = numeric_traits.iloc[:, :3]  # First 3 traits
        
        traits_subset = traits_subset.fillna(traits_subset.mean())
        
        if traits_subset.shape[1] < 2:
            raise ValueError("At least 2 numeric traits required for plotting")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if functional_groups is not None:
            unique_groups = sorted(functional_groups.unique())
            colors = self.colors[:len(unique_groups)]
            
            for i, group in enumerate(unique_groups):
                mask = functional_groups == group
                group_traits = traits_subset[mask]
                
                ax.scatter(group_traits.iloc[:, 0], group_traits.iloc[:, 1],
                          c=colors[i], label=f'Group {group}', s=80, alpha=0.7)
        else:
            ax.scatter(traits_subset.iloc[:, 0], traits_subset.iloc[:, 1],
                      s=80, alpha=0.7, color='blue')
        
# Copyright (c) 2025 Mohamed Z. Hatim
        for i, species in enumerate(traits_subset.index):
            ax.annotate(species, (traits_subset.iloc[i, 0], traits_subset.iloc[i, 1]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
        
        ax.set_xlabel(traits_subset.columns[0])
        ax.set_ylabel(traits_subset.columns[1])
        ax.set_title('Functional Trait Space')
        
        if functional_groups is not None:
            ax.legend()
        
        ax.grid(True, alpha=0.3)
        
        return fig


# Copyright (c) 2025 Mohamed Z. Hatim
def plot_diversity(diversity_results: Dict[str, Any], 
                  indices: List[str] = None) -> plt.Figure:
    """
    Quick diversity plotting function.
    
    Parameters
    ----------
    diversity_results : Dict[str, Any]
        Diversity analysis results
    indices : List[str], optional
        Specific indices to plot
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    plotter = VegetationPlotter()
    return plotter.plot_diversity_indices(diversity_results, indices)


def plot_ordination(ordination_results: Dict[str, Any], 
                   color_by: Union[str, pd.Series] = None) -> plt.Figure:
    """
    Quick ordination plotting function.
    
    Parameters
    ----------
    ordination_results : Dict[str, Any]
        Ordination analysis results
    color_by : Union[str, pd.Series], optional
        Variable to color points by
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    plotter = VegetationPlotter()
    return plotter.plot_ordination(ordination_results, color_by=color_by)


def plot_clustering(clustering_results: Dict[str, Any], 
                   ordination_results: Dict[str, Any] = None) -> plt.Figure:
    """
    Quick clustering plotting function.
    
    Parameters
    ----------
    clustering_results : Dict[str, Any]
        Clustering analysis results
    ordination_results : Dict[str, Any], optional
        Ordination results for overlay plotting
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    plotter = VegetationPlotter()
    
    if ordination_results is not None and 'cluster_labels' in clustering_results:
        return plotter.plot_clusters_on_ordination(
            ordination_results, clustering_results['cluster_labels']
        )
    else:
        return plotter.plot_dendrogram(clustering_results)


def plot_species_accumulation(accumulation_results: Dict[str, Any]) -> plt.Figure:
    """
    Quick species accumulation curve plotting.
    
    Parameters
    ----------
    accumulation_results : Dict[str, Any]
        Species accumulation results
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    plotter = VegetationPlotter()
    return plotter.plot_species_accumulation(accumulation_results)


def plot_environmental_overlay(ordination_results: Dict[str, Any],
                              env_vectors: Dict[str, Any]) -> plt.Figure:
    """
    Quick environmental vector overlay plotting.
    
    Parameters
    ----------
    ordination_results : Dict[str, Any]
        Ordination results
    env_vectors : Dict[str, Any]
        Environmental vector fitting results
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    plotter = VegetationPlotter()
    return plotter.plot_environmental_vectors(ordination_results, env_vectors)


def create_summary_plot(diversity_results: Dict[str, Any] = None,
                       ordination_results: Dict[str, Any] = None,
                       clustering_results: Dict[str, Any] = None) -> plt.Figure:
    """
    Create a comprehensive summary plot with multiple panels.
    
    Parameters
    ----------
    diversity_results : Dict[str, Any], optional
        Diversity analysis results
    ordination_results : Dict[str, Any], optional
        Ordination analysis results
    clustering_results : Dict[str, Any], optional
        Clustering analysis results
        
    Returns
    -------
    plt.Figure
        Figure object with multiple subplots
    """
# Copyright (c) 2025 Mohamed Z. Hatim
    n_plots = sum([diversity_results is not None, 
                   ordination_results is not None,
                   clustering_results is not None])
    
    if n_plots == 0:
        raise ValueError("At least one set of results must be provided")
    
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))
    if n_plots == 1:
        axes = [axes]
    
    plotter = VegetationPlotter()
    plot_idx = 0
    
# Copyright (c) 2025 Mohamed Z. Hatim
    if diversity_results is not None:
        if 'diversity_indices' in diversity_results:
            diversity_df = pd.DataFrame(diversity_results['diversity_indices']).T
# Copyright (c) 2025 Mohamed Z. Hatim
            first_index = diversity_df.columns[0]
            values = diversity_df[first_index]
            
            axes[plot_idx].bar(range(len(values)), values, alpha=0.7)
            axes[plot_idx].set_title(f'{first_index.title()} Index')
            axes[plot_idx].set_ylabel(first_index.title())
            axes[plot_idx].grid(True, alpha=0.3)
        
        plot_idx += 1
    
# Copyright (c) 2025 Mohamed Z. Hatim
    if ordination_results is not None:
        if 'site_scores' in ordination_results:
            site_scores = ordination_results['site_scores']
            axes[plot_idx].scatter(site_scores.iloc[:, 0], site_scores.iloc[:, 1], 
                                 alpha=0.7, s=60)
            
            explained_var = ordination_results.get('explained_variance_ratio', [])
            if len(explained_var) >= 2:
                xlabel = f'Axis 1 ({explained_var[0]:.1%})'
                ylabel = f'Axis 2 ({explained_var[1]:.1%})'
            else:
                xlabel = 'Axis 1'
                ylabel = 'Axis 2'
            
            axes[plot_idx].set_xlabel(xlabel)
            axes[plot_idx].set_ylabel(ylabel)
            axes[plot_idx].set_title('Ordination')
            axes[plot_idx].grid(True, alpha=0.3)
        
        plot_idx += 1
    
# Copyright (c) 2025 Mohamed Z. Hatim
    if clustering_results is not None:
        if 'linkage_matrix' in clustering_results:
# Copyright (c) 2025 Mohamed Z. Hatim
            dendrogram(clustering_results['linkage_matrix'], ax=axes[plot_idx], 
                      no_labels=True)
            axes[plot_idx].set_title('Clustering')
        
        plot_idx += 1
    
    plt.tight_layout()
    return fig