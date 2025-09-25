"""
Interactive Visualization and Reporting Module

This module provides comprehensive interactive visualization and automated reporting
capabilities for vegetation data analysis results.

Copyright (c) 2025 Mohamed Z. Hatim
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available, interactive plots will be limited")

try:
    import bokeh.plotting as bk
    from bokeh.layouts import column, row
    from bokeh.models import HoverTool, ColorBar, LinearColorMapper
    from bokeh.palettes import Viridis256
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False
    warnings.warn("Bokeh not available, some interactive features will be limited")

try:
    from jinja2 import Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    warnings.warn("Jinja2 not available, report generation will be limited")

try:
    import base64
    from io import BytesIO
    REPORT_FEATURES_AVAILABLE = True
except ImportError:
    REPORT_FEATURES_AVAILABLE = False


class InteractiveVisualizer:
    """
    Interactive visualization generator for vegetation analysis results.
    """
    
    def __init__(self):
        """Initialize the InteractiveVisualizer."""
        self.plots = {}
        self.dashboard_components = []
        
    def create_diversity_dashboard(self, 
                                 diversity_results: Dict[str, Any],
                                 data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Create an interactive diversity analysis dashboard.
        
        Parameters
        ----------
        diversity_results : Dict[str, Any]
            Results from diversity analysis
        data : pd.DataFrame, optional
            Original data for additional context
            
        Returns
        -------
        Dict[str, Any]
            Dashboard components and plots
        """
        dashboard = {}
        
        if not PLOTLY_AVAILABLE:
            warnings.warn("Plotly not available, creating static plots instead")
            return self._create_static_diversity_plots(diversity_results, data)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if 'diversity_indices' in diversity_results:
            diversity_df = pd.DataFrame(diversity_results['diversity_indices']).T
            
# Copyright (c) 2025 Mohamed Z. Hatim
            fig = px.bar(
                diversity_df.reset_index(), 
                x='index', 
                y=diversity_df.columns.tolist(),
                title='Diversity Indices Comparison',
                labels={'index': 'Sites', 'value': 'Diversity Value', 'variable': 'Index'},
                barmode='group'
            )
            fig.update_layout(height=500)
            dashboard['diversity_comparison'] = fig
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if 'species_accumulation' in diversity_results:
            accum_data = diversity_results['species_accumulation']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(accum_data['mean']) + 1)),
                y=accum_data['mean'],
                mode='lines+markers',
                name='Observed',
                line=dict(color='blue', width=3)
            ))
            
            if 'ci_lower' in accum_data and 'ci_upper' in accum_data:
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(accum_data['ci_upper']) + 1)),
                    y=accum_data['ci_upper'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,100,80,0)',
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(accum_data['ci_lower']) + 1)),
                    y=accum_data['ci_lower'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,100,80,0)',
                    name='95% CI',
                    fillcolor='rgba(0,100,80,0.2)'
                ))
            
            fig.update_layout(
                title='Species Accumulation Curve',
                xaxis_title='Number of Sites',
                yaxis_title='Number of Species',
                height=500
            )
            dashboard['accumulation_curve'] = fig
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if data is not None:
            species_totals = data.sum(axis=0).sort_values(ascending=False)
            relative_abundance = species_totals / species_totals.sum()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(relative_abundance) + 1)),
                y=relative_abundance.values,
                mode='markers',
                marker=dict(size=8, color='red'),
                text=relative_abundance.index,
                hovertemplate='<b>%{text}</b><br>Rank: %{x}<br>Relative Abundance: %{y:.4f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Rank-Abundance Plot',
                xaxis_title='Species Rank',
                yaxis_title='Relative Abundance',
                xaxis_type='log',
                yaxis_type='log',
                height=500
            )
            dashboard['rank_abundance'] = fig
        
        return dashboard
    
    def create_ordination_dashboard(self, 
                                  ordination_results: Dict[str, Any],
                                  environmental_data: pd.DataFrame = None,
                                  group_column: str = None) -> Dict[str, Any]:
        """
        Create an interactive ordination analysis dashboard.
        
        Parameters
        ----------
        ordination_results : Dict[str, Any]
            Results from ordination analysis
        environmental_data : pd.DataFrame, optional
            Environmental variables for overlay
        group_column : str, optional
            Column name for grouping sites
            
        Returns
        -------
        Dict[str, Any]
            Dashboard components and plots
        """
        dashboard = {}
        
        if not PLOTLY_AVAILABLE:
            warnings.warn("Plotly not available, creating static plots instead")
            return self._create_static_ordination_plots(ordination_results)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if 'site_scores' in ordination_results:
            site_scores = ordination_results['site_scores']
            
# Copyright (c) 2025 Mohamed Z. Hatim
            fig = go.Figure()
            
# Copyright (c) 2025 Mohamed Z. Hatim
            if group_column and environmental_data is not None and group_column in environmental_data.columns:
# Copyright (c) 2025 Mohamed Z. Hatim
                groups = environmental_data[group_column]
                unique_groups = groups.unique()
                colors = px.colors.qualitative.Set1[:len(unique_groups)]
                
                for i, group in enumerate(unique_groups):
                    group_mask = groups == group
                    group_sites = site_scores.loc[group_mask]
                    
                    fig.add_trace(go.Scatter(
                        x=group_sites.iloc[:, 0],
                        y=group_sites.iloc[:, 1],
                        mode='markers',
                        marker=dict(size=10, color=colors[i % len(colors)]),
                        name=str(group),
                        text=group_sites.index,
                        hovertemplate='<b>%{text}</b><br>Axis 1: %{x:.3f}<br>Axis 2: %{y:.3f}<extra></extra>'
                    ))
            else:
# Copyright (c) 2025 Mohamed Z. Hatim
                fig.add_trace(go.Scatter(
                    x=site_scores.iloc[:, 0],
                    y=site_scores.iloc[:, 1],
                    mode='markers',
                    marker=dict(size=10, color='blue'),
                    text=site_scores.index,
                    hovertemplate='<b>%{text}</b><br>Axis 1: %{x:.3f}<br>Axis 2: %{y:.3f}<extra></extra>',
                    name='Sites'
                ))
            
# Copyright (c) 2025 Mohamed Z. Hatim
            if 'species_scores' in ordination_results:
                species_scores = ordination_results['species_scores']
                
# Copyright (c) 2025 Mohamed Z. Hatim
                scale_factor = 0.8 * max(
                    site_scores.iloc[:, 0].max() - site_scores.iloc[:, 0].min(),
                    site_scores.iloc[:, 1].max() - site_scores.iloc[:, 1].min()
                ) / max(
                    species_scores.iloc[:, 0].max() - species_scores.iloc[:, 0].min(),
                    species_scores.iloc[:, 1].max() - species_scores.iloc[:, 1].min()
                )
                
                scaled_species = species_scores * scale_factor
                
# Copyright (c) 2025 Mohamed Z. Hatim
                for species in scaled_species.index:
                    fig.add_annotation(
                        ax=0, ay=0,
                        x=scaled_species.loc[species].iloc[0],
                        y=scaled_species.loc[species].iloc[1],
                        arrowhead=2,
                        arrowcolor="red",
                        arrowsize=1,
                        arrowwidth=2,
                        text=species,
                        textangle=0,
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="red",
                        borderwidth=1
                    )
            
# Copyright (c) 2025 Mohamed Z. Hatim
            axis1_var = ordination_results.get('explained_variance', [0, 0])[0] if len(ordination_results.get('explained_variance', [])) > 0 else 0
            axis2_var = ordination_results.get('explained_variance', [0, 0])[1] if len(ordination_results.get('explained_variance', [])) > 1 else 0
            
            fig.update_layout(
                title=f'Ordination Biplot ({ordination_results.get("method", "Unknown").upper()})',
                xaxis_title=f'Axis 1 ({axis1_var:.1%} variance)',
                yaxis_title=f'Axis 2 ({axis2_var:.1%} variance)',
                height=600,
                showlegend=True
            )
            
            dashboard['ordination_biplot'] = fig
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if 'explained_variance' in ordination_results:
            explained_var = ordination_results['explained_variance']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(explained_var) + 1)),
                y=explained_var,
                mode='lines+markers',
                marker=dict(size=10, color='blue'),
                line=dict(width=3)
            ))
            
            fig.update_layout(
                title='Scree Plot - Explained Variance',
                xaxis_title='Axis',
                yaxis_title='Proportion of Variance Explained',
                height=400
            )
            dashboard['scree_plot'] = fig
        
        return dashboard
    
    def create_clustering_dashboard(self, 
                                  clustering_results: Dict[str, Any],
                                  ordination_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create an interactive clustering analysis dashboard.
        
        Parameters
        ----------
        clustering_results : Dict[str, Any]
            Results from clustering analysis
        ordination_results : Dict[str, Any], optional
            Ordination results for visualization overlay
            
        Returns
        -------
        Dict[str, Any]
            Dashboard components and plots
        """
        dashboard = {}
        
        if not PLOTLY_AVAILABLE:
            return self._create_static_clustering_plots(clustering_results)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if 'dendrogram_data' in clustering_results:
# Copyright (c) 2025 Mohamed Z. Hatim
            fig = go.Figure()
            fig.add_annotation(
                text="Dendrogram visualization requires specialized implementation",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title='Hierarchical Clustering Dendrogram', height=400)
            dashboard['dendrogram'] = fig
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if 'validation_metrics' in clustering_results:
            metrics = clustering_results['validation_metrics']
            
# Copyright (c) 2025 Mohamed Z. Hatim
            if 'silhouette_scores' in metrics:
                silhouette_data = metrics['silhouette_scores']
                cluster_labels = clustering_results.get('cluster_labels', [])
                
                fig = go.Figure()
                
# Copyright (c) 2025 Mohamed Z. Hatim
                y_lower = 10
                for cluster in sorted(set(cluster_labels)):
                    cluster_silhouette = silhouette_data[np.array(cluster_labels) == cluster]
                    cluster_silhouette.sort()
                    
                    size_cluster = len(cluster_silhouette)
                    y_upper = y_lower + size_cluster
                    
                    fig.add_trace(go.Scatter(
                        x=cluster_silhouette,
                        y=list(range(y_lower, y_upper)),
                        fill='tonexty' if cluster > 0 else None,
                        mode='lines',
                        name=f'Cluster {cluster}',
                        line=dict(width=0.5)
                    ))
                    
                    y_lower = y_upper + 10
                
# Copyright (c) 2025 Mohamed Z. Hatim
                avg_silhouette = np.mean(silhouette_data)
                fig.add_vline(
                    x=avg_silhouette,
                    line=dict(color="red", dash="dash"),
                    annotation_text=f"Average: {avg_silhouette:.3f}"
                )
                
                fig.update_layout(
                    title='Silhouette Analysis',
                    xaxis_title='Silhouette Score',
                    yaxis_title='Cluster',
                    height=500
                )
                dashboard['silhouette_plot'] = fig
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if ordination_results and 'site_scores' in ordination_results and 'cluster_labels' in clustering_results:
            site_scores = ordination_results['site_scores']
            cluster_labels = clustering_results['cluster_labels']
            
            fig = go.Figure()
            
# Copyright (c) 2025 Mohamed Z. Hatim
            unique_clusters = sorted(set(cluster_labels))
            colors = px.colors.qualitative.Set1[:len(unique_clusters)]
            
            for i, cluster in enumerate(unique_clusters):
                cluster_mask = np.array(cluster_labels) == cluster
                cluster_sites = site_scores.iloc[cluster_mask]
                
                fig.add_trace(go.Scatter(
                    x=cluster_sites.iloc[:, 0],
                    y=cluster_sites.iloc[:, 1],
                    mode='markers',
                    marker=dict(size=10, color=colors[i % len(colors)]),
                    name=f'Cluster {cluster}',
                    text=cluster_sites.index,
                    hovertemplate='<b>%{text}</b><br>Cluster: ' + str(cluster) + '<br>Axis 1: %{x:.3f}<br>Axis 2: %{y:.3f}<extra></extra>'
                ))
            
            fig.update_layout(
                title='Clustering Results on Ordination Space',
                xaxis_title='Ordination Axis 1',
                yaxis_title='Ordination Axis 2',
                height=600
            )
            dashboard['cluster_ordination'] = fig
        
        return dashboard
    
    def create_trait_dashboard(self, 
                             trait_results: Dict[str, Any],
                             trait_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Create an interactive functional trait analysis dashboard.
        
        Parameters
        ----------
        trait_results : Dict[str, Any]
            Results from trait analysis
        trait_data : pd.DataFrame, optional
            Original trait data
            
        Returns
        -------
        Dict[str, Any]
            Dashboard components and plots
        """
        dashboard = {}
        
        if not PLOTLY_AVAILABLE:
            return {}
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if 'site_diversity' in trait_results:
            site_fd = trait_results['site_diversity']
            
# Copyright (c) 2025 Mohamed Z. Hatim
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['FRic', 'FEve', 'FDiv', 'FDis'],
                vertical_spacing=0.08,
                horizontal_spacing=0.08
            )
            
            for i, metric in enumerate(['FRic', 'FEve', 'FDiv', 'FDis']):
                if metric in site_fd.columns:
                    row = i // 2 + 1
                    col = i % 2 + 1
                    
                    fig.add_trace(
                        go.Bar(
                            x=site_fd.index,
                            y=site_fd[metric],
                            name=metric,
                            showlegend=False
                        ),
                        row=row, col=col
                    )
            
            fig.update_layout(title='Functional Diversity Indices by Site', height=600)
            dashboard['functional_diversity'] = fig
        
# Copyright (c) 2025 Mohamed Z. Hatim
        if trait_data is not None and 'functional_groups' in trait_results:
            functional_groups = trait_results['functional_groups']['functional_groups']
            numeric_traits = trait_data.select_dtypes(include=[np.number]).columns[:3]  # Use first 3 traits
            
            if len(numeric_traits) >= 2:
                trait_subset = trait_data[numeric_traits[:3]].fillna(trait_data[numeric_traits[:3]].mean())
                
                if len(numeric_traits) >= 3:
# Copyright (c) 2025 Mohamed Z. Hatim
                    fig = go.Figure(data=go.Scatter3d(
                        x=trait_subset.iloc[:, 0],
                        y=trait_subset.iloc[:, 1],
                        z=trait_subset.iloc[:, 2],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=functional_groups,
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Functional Group")
                        ),
                        text=trait_subset.index,
                        hovertemplate='<b>%{text}</b><br>' + 
                                    f'{numeric_traits[0]}: %{{x:.2f}}<br>' +
                                    f'{numeric_traits[1]}: %{{y:.2f}}<br>' +
                                    f'{numeric_traits[2]}: %{{z:.2f}}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title='Functional Trait Space (3D)',
                        scene=dict(
                            xaxis_title=numeric_traits[0],
                            yaxis_title=numeric_traits[1],
                            zaxis_title=numeric_traits[2]
                        ),
                        height=600
                    )
                else:
# Copyright (c) 2025 Mohamed Z. Hatim
                    fig = px.scatter(
                        x=trait_subset.iloc[:, 0],
                        y=trait_subset.iloc[:, 1],
                        color=functional_groups,
                        labels={
                            'x': numeric_traits[0],
                            'y': numeric_traits[1],
                            'color': 'Functional Group'
                        },
                        title='Functional Trait Space (2D)',
                        height=600
                    )
                
                dashboard['trait_space'] = fig
        
        return dashboard
    
    def _create_static_diversity_plots(self, diversity_results: Dict[str, Any], 
                                     data: pd.DataFrame = None) -> Dict[str, plt.Figure]:
        """Create static diversity plots when Plotly is not available."""
        plots = {}
        
        if 'diversity_indices' in diversity_results:
            diversity_df = pd.DataFrame(diversity_results['diversity_indices']).T
            
            fig, ax = plt.subplots(figsize=(10, 6))
            diversity_df.plot(kind='bar', ax=ax)
            ax.set_title('Diversity Indices Comparison')
            ax.set_xlabel('Sites')
            ax.set_ylabel('Diversity Value')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plots['diversity_comparison'] = fig
        
        return plots
    
    def _create_static_ordination_plots(self, ordination_results: Dict[str, Any]) -> Dict[str, plt.Figure]:
        """Create static ordination plots when Plotly is not available."""
        plots = {}
        
        if 'site_scores' in ordination_results:
            site_scores = ordination_results['site_scores']
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(site_scores.iloc[:, 0], site_scores.iloc[:, 1])
            
# Copyright (c) 2025 Mohamed Z. Hatim
            for i, site in enumerate(site_scores.index):
                ax.annotate(site, (site_scores.iloc[i, 0], site_scores.iloc[i, 1]))
            
            ax.set_xlabel(f'Axis 1')
            ax.set_ylabel(f'Axis 2')
            ax.set_title('Ordination Plot')
            plt.tight_layout()
            plots['ordination_plot'] = fig
        
        return plots
    
    def _create_static_clustering_plots(self, clustering_results: Dict[str, Any]) -> Dict[str, plt.Figure]:
        """Create static clustering plots when Plotly is not available."""
        plots = {}
        
        if 'validation_metrics' in clustering_results and 'silhouette_scores' in clustering_results['validation_metrics']:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(clustering_results['validation_metrics']['silhouette_scores'], bins=20)
            ax.set_title('Silhouette Score Distribution')
            ax.set_xlabel('Silhouette Score')
            ax.set_ylabel('Frequency')
            plt.tight_layout()
            plots['silhouette_hist'] = fig
        
        return plots
    
    def save_dashboard(self, dashboard: Dict[str, Any], 
                      filename: str, 
                      format: str = 'html') -> str:
        """
        Save dashboard to file.
        
        Parameters
        ----------
        dashboard : Dict[str, Any]
            Dashboard components
        filename : str
            Output filename
        format : str, optional
            Output format ('html', 'png'), by default 'html'
            
        Returns
        -------
        str
            Path to saved file
        """
        if not PLOTLY_AVAILABLE and format == 'html':
            warnings.warn("Plotly not available, cannot save HTML dashboard")
            return None
        
        if format == 'html':
# Copyright (c) 2025 Mohamed Z. Hatim
            html_parts = []
            html_parts.append('<html><head><title>VegZ Dashboard</title></head><body>')
            html_parts.append('<h1>VegZ Analysis Dashboard</h1>')
            
            for plot_name, plot_obj in dashboard.items():
                if hasattr(plot_obj, 'to_html'):
                    html_parts.append(f'<h2>{plot_name.replace("_", " ").title()}</h2>')
                    html_parts.append(plot_obj.to_html(include_plotlyjs='cdn', div_id=plot_name))
            
            html_parts.append('</body></html>')
            
            with open(filename, 'w') as f:
                f.write('\n'.join(html_parts))
            
            return filename
        
        return None


class ReportGenerator:
    """
    Automated report generator for vegetation analysis results.
    """
    
    def __init__(self):
        """Initialize the ReportGenerator."""
        self.template_cache = {}
        
    def generate_analysis_report(self, 
                               results: Dict[str, Any],
                               data_summary: Dict[str, Any] = None,
                               output_format: str = 'html') -> str:
        """
        Generate comprehensive analysis report.
        
        Parameters
        ----------
        results : Dict[str, Any]
            Analysis results from various VegZ methods
        data_summary : Dict[str, Any], optional
            Summary statistics of the input data
        output_format : str, optional
            Output format ('html', 'markdown'), by default 'html'
            
        Returns
        -------
        str
            Generated report content
        """
        if not JINJA2_AVAILABLE:
            return self._generate_simple_report(results, data_summary)
        
        template_content = self._get_report_template(output_format)
        template = Template(template_content)
        
# Copyright (c) 2025 Mohamed Z. Hatim
        context = {
            'results': results,
            'data_summary': data_summary or {},
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
# Copyright (c) 2025 Mohamed Z. Hatim
        context['summary_stats'] = self._calculate_summary_stats(results)
        
        return template.render(**context)
    
    def _get_report_template(self, format: str) -> str:
        """Get report template for specified format."""
        if format == 'html':
            return """
<!DOCTYPE html>
<html>
<head>
    <title>VegZ Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #2E7D32; border-bottom: 2px solid #2E7D32; }
        h2 { color: #388E3C; border-bottom: 1px solid #388E3C; }
        .summary { background-color: #E8F5E8; padding: 15px; border-radius: 5px; }
        .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #F1F8E9; border-radius: 3px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
    </style>
</head>
<body>
    <h1>VegZ Vegetation Analysis Report</h1>
    <p><strong>Generated:</strong> {{ timestamp }}</p>
    
    {% if data_summary %}
    <div class="summary">
        <h2>Data Summary</h2>
        <div class="metric"><strong>Sites:</strong> {{ data_summary.get('n_sites', 'N/A') }}</div>
        <div class="metric"><strong>Species:</strong> {{ data_summary.get('n_species', 'N/A') }}</div>
        <div class="metric"><strong>Total Observations:</strong> {{ data_summary.get('total_observations', 'N/A') }}</div>
    </div>
    {% endif %}
    
    {% if summary_stats %}
    <h2>Analysis Summary</h2>
    <ul>
    {% for stat in summary_stats %}
        <li>{{ stat }}</li>
    {% endfor %}
    </ul>
    {% endif %}
    
    {% for analysis_type, analysis_results in results.items() %}
    <h2>{{ analysis_type.replace('_', ' ').title() }}</h2>
    
    {% if analysis_results %}
        <p>Analysis completed successfully.</p>
        {% if analysis_results is mapping %}
            <ul>
            {% for key, value in analysis_results.items() %}
                <li><strong>{{ key }}:</strong> 
                {% if value is number %}
                    {{ "%.3f"|format(value) }}
                {% else %}
                    {{ value|string|truncate(100) }}
                {% endif %}
                </li>
            {% endfor %}
            </ul>
        {% endif %}
    {% else %}
        <p>No results available.</p>
    {% endif %}
    
    {% endfor %}
    
    <hr>
    <p><em>Report generated by VegZ - A comprehensive Python package for vegetation data analysis</em></p>
</body>
</html>
            """
        
        elif format == 'markdown':
            return """
# Copyright (c) 2025 Mohamed Z. Hatim

**Generated:** {{ timestamp }}

{% if data_summary %}
# Copyright (c) 2025 Mohamed Z. Hatim

- **Sites:** {{ data_summary.get('n_sites', 'N/A') }}
- **Species:** {{ data_summary.get('n_species', 'N/A') }}  
- **Total Observations:** {{ data_summary.get('total_observations', 'N/A') }}
{% endif %}

{% if summary_stats %}
# Copyright (c) 2025 Mohamed Z. Hatim

{% for stat in summary_stats %}
- {{ stat }}
{% endfor %}
{% endif %}

{% for analysis_type, analysis_results in results.items() %}
# Copyright (c) 2025 Mohamed Z. Hatim

{% if analysis_type == 'diversity' %}
# Copyright (c) 2025 Mohamed Z. Hatim
- Analysis completed with {{ analysis_results.diversity_indices|length if analysis_results.diversity_indices else 0 }} sites
{% endif %}

{% endfor %}

---
*Report generated by VegZ - A comprehensive Python package for vegetation data analysis*
            """
        
        return ""
    
    def _generate_simple_report(self, results: Dict[str, Any], 
                              data_summary: Dict[str, Any] = None) -> str:
        """Generate simple text report when Jinja2 is not available."""
        report_lines = []
        report_lines.append("VegZ Vegetation Analysis Report")
        report_lines.append("=" * 40)
        report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        if data_summary:
            report_lines.append("Data Summary:")
            report_lines.append(f"- Sites: {data_summary.get('n_sites', 'N/A')}")
            report_lines.append(f"- Species: {data_summary.get('n_species', 'N/A')}")
            report_lines.append(f"- Total Observations: {data_summary.get('total_observations', 'N/A')}")
            report_lines.append("")
        
        for analysis_type, analysis_results in results.items():
            report_lines.append(f"{analysis_type.replace('_', ' ').title()}:")
            report_lines.append("-" * 20)
            
            if isinstance(analysis_results, dict):
                for key, value in analysis_results.items():
                    if isinstance(value, (int, float)):
                        report_lines.append(f"- {key}: {value:.3f}")
                    else:
                        report_lines.append(f"- {key}: {type(value).__name__}")
            
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def _calculate_summary_stats(self, results: Dict[str, Any]) -> List[str]:
        """Calculate summary statistics from results."""
        stats = []
        
        for analysis_type, analysis_results in results.items():
            if analysis_type == 'diversity' and isinstance(analysis_results, dict):
                if 'diversity_indices' in analysis_results:
                    n_sites = len(analysis_results['diversity_indices'])
                    stats.append(f"Diversity analysis completed for {n_sites} sites")
            
            elif analysis_type == 'ordination' and isinstance(analysis_results, dict):
                if 'explained_variance' in analysis_results:
                    total_var = sum(analysis_results['explained_variance'][:2])
                    stats.append(f"First two ordination axes explain {total_var:.1%} of variance")
            
            elif analysis_type == 'clustering' and isinstance(analysis_results, dict):
                if 'n_clusters' in analysis_results:
                    stats.append(f"Identified {analysis_results['n_clusters']} vegetation clusters")
        
        return stats
    
    def save_report(self, report_content: str, filename: str, format: str = 'html') -> str:
        """
        Save report to file.
        
        Parameters
        ----------
        report_content : str
            Generated report content
        filename : str
            Output filename
        format : str, optional
            Output format, by default 'html'
            
        Returns
        -------
        str
            Path to saved file
        """
        if not filename.endswith(f'.{format}'):
            filename = f"{filename}.{format}"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return filename


# Copyright (c) 2025 Mohamed Z. Hatim
def quick_diversity_dashboard(diversity_results: Dict[str, Any], 
                            data: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Quick interactive diversity dashboard.
    
    Parameters
    ----------
    diversity_results : Dict[str, Any]
        Results from diversity analysis
    data : pd.DataFrame, optional
        Original community data
        
    Returns
    -------
    Dict[str, Any]
        Dashboard plots
    """
    visualizer = InteractiveVisualizer()
    return visualizer.create_diversity_dashboard(diversity_results, data)


def quick_analysis_report(results: Dict[str, Any], 
                        data_summary: Dict[str, Any] = None,
                        filename: str = 'vegz_report',
                        format: str = 'html') -> str:
    """
    Quick analysis report generation.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Analysis results from VegZ methods
    data_summary : Dict[str, Any], optional
        Data summary statistics
    filename : str, optional
        Output filename, by default 'vegz_report'
    format : str, optional
        Output format, by default 'html'
        
    Returns
    -------
    str
        Path to generated report
    """
    generator = ReportGenerator()
    report_content = generator.generate_analysis_report(results, data_summary, format)
    return generator.save_report(report_content, filename, format)