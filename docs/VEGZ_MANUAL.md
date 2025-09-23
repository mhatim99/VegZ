# VegZ: Complete User Manual

**Version 1.1.0**
**Author: Mohamed Z. Hatim**
**Date: September 2025**

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core VegZ Class](#core-vegz-class)
5. [Data Management](#data-management)
6. [Diversity Analysis](#diversity-analysis)
7. [Multivariate Analysis](#multivariate-analysis)
8. [Clustering Methods](#clustering-methods)
9. [Statistical Analysis](#statistical-analysis)
10. [Environmental Modeling](#environmental-modeling)
11. [Temporal Analysis](#temporal-analysis)
12. [Spatial Analysis](#spatial-analysis)
13. [Functional Traits](#functional-traits)
14. [Specialized Methods](#specialized-methods)
15. [Machine Learning](#machine-learning)
16. [Visualization](#visualization)
17. [Interactive Features](#interactive-features)
18. [Data Quality and Validation](#data-quality-and-validation)
19. [Enhanced Species Name Error Detection](#enhanced-species-name-error-detection)
20. [Quick Functions](#quick-functions)
21. [Best Practices](#best-practices)
22. [Troubleshooting](#troubleshooting)
23. [API Reference](#api-reference)

## Introduction

VegZ is a comprehensive Python package designed for vegetation data analysis and environmental modeling. It provides a complete suite of tools for ecologists, environmental scientists, and researchers working with biodiversity and vegetation data.

### Key Features

- **Data Management**: Load, validate, and preprocess vegetation data
- **Enhanced Species Name Error Detection** (New in v1.1.0): Comprehensive validation with 10+ error categories
- **Diversity Analysis**: 15+ diversity indices and rarefaction curves
- **Multivariate Analysis**: Complete ordination suite (PCA, NMDS, CCA, RDA, etc.)
- **Advanced Clustering**: TWINSPAN, elbow analysis, hierarchical clustering
- **Statistical Analysis**: PERMANOVA, ANOSIM, Mantel tests
- **Environmental Modeling**: GAMs, species response curves
- **Temporal & Spatial Analysis**: Time series, interpolation, landscape metrics
- **Machine Learning**: Species distribution modeling, predictive modeling
- **Visualization**: Comprehensive plotting and interactive dashboards

## Installation

### Basic Installation

```bash
pip install VegZ
```

### Extended Installation

```bash
# With spatial analysis support
pip install VegZ[spatial]

# With remote sensing capabilities
pip install VegZ[remote-sensing]

# Complete installation with all features
pip install VegZ[spatial,remote-sensing,fuzzy,interactive]
```

### Development Installation

```bash
pip install git+https://github.com/mhatim99/VegZ.git
```

## Quick Start

### Basic Usage

```python
import pandas as pd
import numpy as np
from VegZ import VegZ

# Initialize VegZ
veg = VegZ()

# Create sample data
np.random.seed(42)
n_sites = 50
n_species = 20

# Generate synthetic vegetation data
data = pd.DataFrame(
    np.random.poisson(3, (n_sites, n_species)),
    columns=[f'Species_{i+1}' for i in range(n_species)],
    index=[f'Site_{i+1}' for i in range(n_sites)]
)

# Load data
veg.data = data
veg.species_matrix = data

# Quick diversity analysis
diversity = veg.calculate_diversity(['shannon', 'simpson', 'richness'])
print("Diversity indices calculated for", len(diversity), "sites")

# Quick ordination
pca_results = veg.pca_analysis()
print("PCA completed - explained variance:", 
      pca_results['explained_variance_ratio'][:2])
```

## Core VegZ Class

The main `VegZ` class provides the primary interface for vegetation data analysis.

### Initialization

```python
from VegZ import VegZ

# Basic initialization
veg = VegZ()

# Initialize with data
veg = VegZ()
veg.load_data('vegetation_data.csv')
```

### Core Attributes

- `data`: Main dataset (DataFrame)
- `species_matrix`: Species abundance matrix
- `environmental_data`: Environmental variables
- `metadata`: Additional information

## Data Management

### Loading Data

#### From CSV Files

```python
# Load CSV data
data = veg.load_data('vegetation_data.csv', format_type='csv')

# Specify species columns
data = veg.load_data(
    'data.csv', 
    species_cols=['Sp1', 'Sp2', 'Sp3']
)
```

#### From Excel Files

```python
# Load Excel data
data = veg.load_data('vegetation_data.xlsx', format_type='excel')

# Specify sheet
data = veg.load_data(
    'data.xlsx', 
    format_type='excel',
    sheet_name='VegetationData'
)
```

#### Data Format Requirements

VegZ expects data in **site-by-species matrix format**:

```
Site_ID  | Species_1 | Species_2 | Species_3 | ...
---------|-----------|-----------|-----------|----
Site_001 |    25     |    18     |    12     | ...
Site_002 |    32     |    22     |    16     | ...
Site_003 |    19     |    15     |     8     | ...
```

### Data Transformations

```python
# Hellinger transformation
transformed_data = veg.transform_data(method='hellinger')

# Log transformation
log_data = veg.transform_data(method='log')

# Square root transformation
sqrt_data = veg.transform_data(method='sqrt')

# Wisconsin double standardization
wisconsin_data = veg.transform_data(method='wisconsin')

# Chord transformation
chord_data = veg.transform_data(method='chord')

# Standardization (z-score)
standardized_data = veg.transform_data(method='standardize')
```

## Diversity Analysis

### Basic Diversity Indices

```python
from VegZ import DiversityAnalyzer

diversity = DiversityAnalyzer()

# Calculate multiple indices
indices = diversity.calculate_diversity(
    data, 
    indices=['shannon', 'simpson', 'richness', 'evenness']
)

# Results structure
print(indices.keys())  # ['shannon', 'simpson', 'richness', 'evenness']
```

### Advanced Diversity Indices

```python
# Fisher's alpha
fisher_alpha = diversity.calculate_diversity(data, indices=['fisher_alpha'])

# Berger-Parker dominance
berger_parker = diversity.calculate_diversity(data, indices=['berger_parker'])

# McIntosh diversity
mcintosh = diversity.calculate_diversity(data, indices=['mcintosh'])

# Brillouin diversity
brillouin = diversity.calculate_diversity(data, indices=['brillouin'])

# Menhinick and Margalef indices
additional = diversity.calculate_diversity(
    data, 
    indices=['menhinick', 'margalef']
)
```

### Richness Estimators

```python
# Chao1 estimator
chao1 = diversity.richness_estimators(data, methods=['chao1'])

# ACE estimator
ace = diversity.richness_estimators(data, methods=['ace'])

# Jackknife estimators
jackknife = diversity.richness_estimators(
    data, 
    methods=['jackknife1', 'jackknife2']
)

# All estimators
all_estimators = diversity.richness_estimators(
    data,
    methods=['chao1', 'ace', 'jackknife1', 'jackknife2']
)
```

### Hill Numbers

```python
# Calculate Hill numbers for multiple orders
hill_numbers = diversity.hill_numbers(
    data,
    orders=[0, 0.5, 1, 1.5, 2]  # q values
)

# Hill numbers interpretation:
# q=0: Species richness
# q=1: Shannon diversity (exponential)
# q=2: Simpson diversity (inverse)
```

### Beta Diversity

```python
# Beta diversity analysis
beta_div = diversity.beta_diversity(
    data,
    method='whittaker'  # 'whittaker', 'sorensen', 'jaccard'
)

# Pairwise beta diversity
pairwise_beta = diversity.pairwise_beta_diversity(data)

print("Beta diversity components:")
print(f"Turnover: {beta_div['turnover']}")
print(f"Nestedness: {beta_div['nestedness']}")
print(f"Total beta: {beta_div['total_beta']}")
```

### Rarefaction Analysis

```python
# Rarefaction curves
rarefaction = diversity.rarefaction_analysis(
    data,
    sample_sizes=range(1, 101, 5),  # Sample sizes to test
    n_iterations=100  # Number of iterations
)

# Plot rarefaction curves
diversity.plot_rarefaction(rarefaction)

# Species accumulation curves
accumulation = diversity.species_accumulation_curves(
    data,
    n_permutations=100
)
```

## Multivariate Analysis

### Principal Component Analysis (PCA)

```python
from VegZ import MultivariateAnalyzer

multivar = MultivariateAnalyzer()

# Basic PCA
pca_results = multivar.pca_analysis(
    data,
    n_components=5,
    transform='hellinger'  # Data transformation
)

# Access results
print("Explained variance ratio:", pca_results['explained_variance_ratio'])
print("Site scores shape:", pca_results['site_scores'].shape)
print("Species loadings shape:", pca_results['species_loadings'].shape)

# Biplot
multivar.plot_pca_biplot(
    pca_results,
    show_species=True,
    show_sites=True
)
```

### Correspondence Analysis (CA)

```python
# Correspondence Analysis
ca_results = multivar.correspondence_analysis(
    data,
    scaling=1  # 1 for sites, 2 for species
)

print("CA Eigenvalues:", ca_results['eigenvalues'])
print("CA Inertia:", ca_results['total_inertia'])
```

### Detrended Correspondence Analysis (DCA)

```python
# DCA with detrending
dca_results = multivar.dca_analysis(
    data,
    detrending='polynomial',  # 'polynomial' or 'linear'
    segments=26  # Number of segments for detrending
)

print("DCA gradient lengths:", dca_results['gradient_lengths'])
```

### Canonical Correspondence Analysis (CCA)

```python
# Environmental data required
env_data = pd.DataFrame({
    'pH': np.random.normal(6.5, 0.8, n_sites),
    'Moisture': np.random.normal(50, 15, n_sites),
    'Temperature': np.random.normal(15, 5, n_sites)
})

# CCA with environmental constraints
cca_results = multivar.cca_analysis(
    species_data=data,
    environmental_data=env_data,
    formula=None  # Use all environmental variables
)

print("CCA Eigenvalues:", cca_results['eigenvalues'])
print("Environmental fit:", cca_results['environmental_fit'])
```

### Redundancy Analysis (RDA)

```python
# RDA for linear relationships
rda_results = multivar.rda_analysis(
    species_data=data,
    environmental_data=env_data,
    transform='hellinger'
)

print("RDA Eigenvalues:", rda_results['eigenvalues'])
print("Constrained variance:", rda_results['constrained_variance'])
```

### Non-metric Multidimensional Scaling (NMDS)

```python
# NMDS analysis
nmds_results = multivar.nmds_analysis(
    data,
    distance_metric='bray_curtis',
    n_dimensions=2,
    max_iterations=200
)

print("NMDS Stress:", nmds_results['stress'])
print("NMDS Converged:", nmds_results['converged'])

# Stress plot
multivar.plot_nmds_stress(nmds_results)
```

### Principal Coordinates Analysis (PCoA)

```python
# PCoA (metric MDS)
pcoa_results = multivar.pcoa_analysis(
    data,
    distance_metric='bray_curtis',
    n_dimensions=5
)

print("PCoA Eigenvalues:", pcoa_results['eigenvalues'])
print("Explained variance:", pcoa_results['explained_variance'])
```

### Environmental Vector Fitting

```python
# Fit environmental vectors to ordination
vector_fit = multivar.environmental_fit(
    ordination_scores=nmds_results['site_scores'],
    environmental_data=env_data,
    permutations=999
)

print("Environmental vectors:")
for var in vector_fit:
    print(f"{var}: R² = {vector_fit[var]['r_squared']:.3f}, "
          f"p = {vector_fit[var]['p_value']:.3f}")
```

## Clustering Methods

### TWINSPAN Analysis

```python
from VegZ import VegetationClustering

clustering = VegetationClustering()

# Two-Way Indicator Species Analysis
twinspan_results = clustering.twinspan(
    data,
    cut_levels=[0, 2, 5, 10, 20],  # Abundance cut levels
    max_divisions=6,
    min_group_size=5
)

print("Site classification:", twinspan_results['site_classification'])
print("Number of groups:", len(np.unique(twinspan_results['site_classification'])))

# Access classification tree
tree = twinspan_results['classification_tree']
print("Indicator species:", tree['indicator_species'])
```

### Elbow Analysis

```python
# Comprehensive elbow analysis
elbow_results = clustering.comprehensive_elbow_analysis(
    data,
    k_range=range(2, 16),
    methods=[
        'knee_locator',      # Kneedle algorithm
        'derivative',        # Second derivative maximum
        'variance_explained', # <10% additional variance
        'distortion_jump',   # Jump detection
        'l_method'          # L-method
    ],
    transform='hellinger',
    plot_results=True
)

# Get consensus recommendation
optimal_k = elbow_results['recommendations']['consensus']
confidence = elbow_results['recommendations']['confidence']

print(f"Optimal clusters: {optimal_k} (confidence: {confidence:.2f})")

# Individual method results
for method in elbow_results['individual_recommendations']:
    k_rec = elbow_results['individual_recommendations'][method]
    print(f"{method}: {k_rec} clusters")
```

### Hierarchical Clustering

```python
# Hierarchical clustering with ecological distances
hier_results = clustering.hierarchical_clustering(
    data,
    n_clusters=4,
    distance_metric='bray_curtis',
    linkage_method='ward'
)

print("Cluster labels:", hier_results['cluster_labels'])
print("Cophenetic correlation:", hier_results['cophenetic_correlation'])

# Plot dendrogram
clustering.plot_dendrogram(
    hier_results,
    show_leaf_labels=True,
    color_threshold=0.7
)
```

### K-means Clustering

```python
# K-means clustering
kmeans_results = clustering.kmeans_clustering(
    data,
    n_clusters=4,
    n_init=10,
    transform='hellinger'
)

print("Cluster centers shape:", kmeans_results['cluster_centers'].shape)
print("Within-cluster sum of squares:", kmeans_results['inertia'])
```

### Fuzzy C-means Clustering

```python
# Fuzzy clustering for gradient boundaries
fuzzy_results = clustering.fuzzy_cmeans(
    data,
    n_clusters=4,
    fuzziness=2.0,
    max_iterations=100
)

print("Fuzzy membership matrix shape:", fuzzy_results['membership'].shape)
print("Partition coefficient:", fuzzy_results['partition_coefficient'])
```

### DBSCAN Clustering

```python
# Density-based clustering
dbscan_results = clustering.dbscan_clustering(
    data,
    eps=0.5,
    min_samples=5,
    distance_metric='bray_curtis'
)

print("Number of clusters:", len(np.unique(dbscan_results['cluster_labels'])))
print("Number of noise points:", np.sum(dbscan_results['cluster_labels'] == -1))
```

### Clustering Validation

```python
# Silhouette analysis
silhouette = clustering.silhouette_analysis(
    data,
    cluster_labels=kmeans_results['cluster_labels'],
    distance_metric='bray_curtis'
)

print("Average silhouette score:", silhouette['average_score'])

# Gap statistic
gap_stat = clustering.gap_statistic(
    data,
    k_range=range(2, 11),
    n_references=50
)

print("Gap statistic optimal k:", gap_stat['optimal_k'])
```

## Statistical Analysis

### PERMANOVA

```python
from VegZ import EcologicalStatistics

stats = EcologicalStatistics()

# Create grouping variable
groups = ['Group_A'] * 25 + ['Group_B'] * 25

# PERMANOVA test
permanova_results = stats.permanova(
    distance_matrix=None,  # Will be calculated
    species_data=data,
    groups=groups,
    distance_metric='bray_curtis',
    permutations=999
)

print(f"PERMANOVA F-statistic: {permanova_results['F_statistic']:.4f}")
print(f"p-value: {permanova_results['p_value']:.4f}")
print(f"R²: {permanova_results['R_squared']:.4f}")
```

### ANOSIM

```python
# Analysis of Similarities
anosim_results = stats.anosim(
    distance_matrix=None,
    species_data=data,
    groups=groups,
    distance_metric='bray_curtis',
    permutations=999
)

print(f"ANOSIM R-statistic: {anosim_results['R_statistic']:.4f}")
print(f"p-value: {anosim_results['p_value']:.4f}")

# R-statistic interpretation:
# R = 1: Groups are completely separated
# R = 0: Groups are not separated
# R < 0: Within-group distances > between-group distances
```

### MRPP (Multi-Response Permutation Procedures)

```python
# MRPP analysis
mrpp_results = stats.mrpp(
    distance_matrix=None,
    species_data=data,
    groups=groups,
    distance_metric='bray_curtis',
    permutations=999
)

print(f"MRPP delta: {mrpp_results['delta']:.4f}")
print(f"MRPP A: {mrpp_results['A']:.4f}")
print(f"p-value: {mrpp_results['p_value']:.4f}")
```

### Mantel Tests

```python
# Create second matrix (e.g., environmental distances)
env_distances = pdist(env_data, metric='euclidean')
species_distances = pdist(data, metric='braycurtis')

# Mantel test
mantel_results = stats.mantel_test(
    matrix1=species_distances,
    matrix2=env_distances,
    permutations=999
)

print(f"Mantel correlation: {mantel_results['correlation']:.4f}")
print(f"p-value: {mantel_results['p_value']:.4f}")

# Partial Mantel test (controlling for spatial distances)
spatial_data = pd.DataFrame({
    'X': np.random.uniform(0, 100, n_sites),
    'Y': np.random.uniform(0, 100, n_sites)
})
spatial_distances = pdist(spatial_data, metric='euclidean')

partial_mantel = stats.partial_mantel_test(
    matrix1=species_distances,
    matrix2=env_distances,
    matrix3=spatial_distances,
    permutations=999
)

print(f"Partial Mantel correlation: {partial_mantel['correlation']:.4f}")
```

### Indicator Species Analysis

```python
# Indicator Species Analysis (IndVal)
cluster_labels = kmeans_results['cluster_labels']

indval_results = stats.indicator_species_analysis(
    species_data=data,
    groups=cluster_labels,
    permutations=999
)

# Display significant indicators
significant_indicators = indval_results[indval_results['p_value'] < 0.05]
print("Significant indicator species:")
print(significant_indicators[['indicator_value', 'p_value']].head())
```

### SIMPER Analysis

```python
# Similarity Percentages
simper_results = stats.simper_analysis(
    species_data=data,
    groups=groups,
    distance_metric='bray_curtis'
)

print("Species contributing to group differences:")
print(simper_results['between_groups'].head())
```

## Environmental Modeling

### Generalized Additive Models (GAMs)

```python
from VegZ import EnvironmentalModeler

env_model = EnvironmentalModeler()

# Fit GAM for species response
species_col = 'Species_1'
environmental_var = 'pH'

gam_results = env_model.fit_gam(
    species_data=data[species_col],
    environmental_data=env_data[environmental_var],
    smoother='spline',  # 'spline', 'lowess', 'polynomial'
    smoothing_parameter=0.5
)

print(f"GAM R²: {gam_results['r_squared']:.4f}")
print(f"AIC: {gam_results['aic']:.2f}")

# Plot response curve
env_model.plot_species_response(
    gam_results,
    species_name=species_col,
    env_var_name=environmental_var
)
```

### Species Response Curves

```python
# Fit different response curve models
response_models = [
    'gaussian',
    'skewed_gaussian', 
    'beta',
    'linear',
    'threshold',
    'unimodal'
]

best_models = {}
for species in data.columns[:5]:  # First 5 species
    species_responses = env_model.fit_response_curves(
        species_data=data[species],
        environmental_data=env_data['pH'],
        models=response_models
    )
    
    # Select best model based on AIC
    best_model = min(species_responses.items(), 
                    key=lambda x: x[1]['aic'])
    best_models[species] = best_model
    
    print(f"{species}: Best model = {best_model[0]}, "
          f"AIC = {best_model[1]['aic']:.2f}")
```

### Environmental Gradient Analysis

```python
# Gradient analysis
gradient_results = env_model.gradient_analysis(
    species_data=data,
    environmental_data=env_data,
    ordination_method='cca'
)

print("Environmental gradients:")
for i, gradient in enumerate(gradient_results['gradients']):
    print(f"Gradient {i+1}: {gradient['interpretation']}")
    print(f"  Explained variance: {gradient['explained_variance']:.2%}")
```

### Environmental Niche Modeling

```python
# Niche modeling for species
species_niches = env_model.niche_modeling(
    species_data=data,
    environmental_data=env_data,
    method='hypervolume'  # 'hypervolume', 'convex_hull', 'ellipsoid'
)

for species in list(species_niches.keys())[:3]:
    niche = species_niches[species]
    print(f"{species} niche:")
    print(f"  Volume: {niche['volume']:.4f}")
    print(f"  Centroid: pH={niche['centroid']['pH']:.2f}")
```

## Temporal Analysis

### Phenology Modeling

```python
from VegZ import TemporalAnalyzer
import datetime

temporal = TemporalAnalyzer()

# Create temporal data
dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
phenology_data = pd.DataFrame({
    'date': dates,
    'abundance': np.sin(2 * np.pi * np.arange(len(dates)) / 365) + 
                np.random.normal(0, 0.1, len(dates))
})

# Fit phenology models
phenology_results = temporal.phenology_modeling(
    temporal_data=phenology_data,
    date_column='date',
    value_column='abundance',
    models=['gaussian', 'double_gaussian', 'polynomial']
)

best_model = min(phenology_results.items(), 
                key=lambda x: x[1]['aic'])
print(f"Best phenology model: {best_model[0]}")
print(f"Peak date: {best_model[1]['peak_date']}")
```

### Trend Detection

```python
# Mann-Kendall trend test
trend_results = temporal.trend_detection(
    temporal_data=phenology_data['abundance'],
    method='mann_kendall'
)

print(f"Trend: {trend_results['trend']}")
print(f"p-value: {trend_results['p_value']:.4f}")
print(f"Slope: {trend_results['slope']:.6f}")
```

### Time Series Decomposition

```python
# Seasonal decomposition
decomposition = temporal.time_series_decomposition(
    temporal_data=phenology_data,
    date_column='date',
    value_column='abundance',
    method='seasonal_decompose',  # 'seasonal_decompose', 'stl'
    period=365
)

print("Decomposition components:")
print(f"  Trend strength: {decomposition['trend_strength']:.3f}")
print(f"  Seasonal strength: {decomposition['seasonal_strength']:.3f}")
```

## Spatial Analysis

### Spatial Interpolation

```python
from VegZ import SpatialAnalyzer

spatial = SpatialAnalyzer()

# Create spatial coordinates
coords = pd.DataFrame({
    'X': np.random.uniform(0, 100, n_sites),
    'Y': np.random.uniform(0, 100, n_sites)
})

# Species abundance for interpolation
abundance = data['Species_1'].values

# Inverse Distance Weighting
idw_results = spatial.spatial_interpolation(
    coordinates=coords[['X', 'Y']].values,
    values=abundance,
    method='idw',
    power=2,
    grid_resolution=50
)

print("IDW interpolation completed")
print(f"Grid shape: {idw_results['interpolated_grid'].shape}")

# Kriging interpolation
kriging_results = spatial.spatial_interpolation(
    coordinates=coords[['X', 'Y']].values,
    values=abundance,
    method='kriging',
    variogram_model='spherical'
)

print(f"Kriging variance: {kriging_results['variance'].mean():.4f}")
```

### Landscape Metrics

```python
# Calculate landscape metrics from presence/absence data
binary_data = (data > 0).astype(int)

landscape_metrics = spatial.landscape_metrics(
    species_data=binary_data,
    coordinates=coords,
    cell_size=1.0
)

print("Landscape metrics:")
print(f"  Patch density: {landscape_metrics['patch_density']:.4f}")
print(f"  Edge density: {landscape_metrics['edge_density']:.4f}")
print(f"  Contagion index: {landscape_metrics['contagion_index']:.4f}")
```

### Spatial Autocorrelation

```python
# Moran's I test
morans_i = spatial.spatial_autocorrelation(
    species_data=abundance,
    coordinates=coords[['X', 'Y']].values,
    method='morans_i',
    distance_threshold=10.0
)

print(f"Moran's I: {morans_i['statistic']:.4f}")
print(f"p-value: {morans_i['p_value']:.4f}")
print(f"Expected I: {morans_i['expected']:.4f}")

# Geary's C test
gearys_c = spatial.spatial_autocorrelation(
    species_data=abundance,
    coordinates=coords[['X', 'Y']].values,
    method='gearys_c',
    distance_threshold=10.0
)

print(f"Geary's C: {gearys_c['statistic']:.4f}")
```

## Functional Traits

### Trait Analysis

```python
from VegZ import FunctionalTraitAnalyzer

# Create trait data
trait_data = pd.DataFrame({
    'SLA': np.random.normal(20, 5, n_species),  # Specific Leaf Area
    'Height': np.random.lognormal(1, 0.5, n_species),  # Plant Height
    'SeedMass': np.random.lognormal(0, 1, n_species)  # Seed Mass
})

traits = FunctionalTraitAnalyzer()

# Community-weighted means
cwm_results = traits.community_weighted_means(
    species_data=data,
    trait_data=trait_data
)

print("Community-weighted means:")
print(cwm_results.head())
```

### Functional Diversity

```python
# Functional diversity indices
func_diversity = traits.functional_diversity(
    species_data=data,
    trait_data=trait_data,
    indices=['fric', 'feve', 'fdiv', 'rao']
)

print("Functional diversity indices:")
print(f"  FRic (Functional Richness): {func_diversity['fric'].mean():.3f}")
print(f"  FEve (Functional Evenness): {func_diversity['feve'].mean():.3f}")
print(f"  FDiv (Functional Divergence): {func_diversity['fdiv'].mean():.3f}")
print(f"  Rao's Quadratic Entropy: {func_diversity['rao'].mean():.3f}")
```

### Trait Syndromes

```python
# Identify trait syndromes
syndromes = traits.trait_syndromes(
    trait_data=trait_data,
    method='pca',
    n_components=2
)

print("Trait syndromes (PCA loadings):")
print(syndromes['loadings'])
```

### Fourth-Corner Analysis

```python
# Fourth-corner analysis (trait-environment relationships)
fourth_corner = traits.fourth_corner_analysis(
    species_data=data,
    trait_data=trait_data,
    environmental_data=env_data,
    permutations=999
)

print("Significant trait-environment associations:")
significant = fourth_corner[fourth_corner['p_value'] < 0.05]
print(significant[['trait', 'environment', 'correlation', 'p_value']])
```

## Specialized Methods

### Phylogenetic Diversity

```python
from VegZ import PhylogeneticDiversityAnalyzer

# Create mock phylogenetic tree (distances)
phylo_distances = np.random.exponential(1, (n_species, n_species))
phylo_distances = (phylo_distances + phylo_distances.T) / 2
np.fill_diagonal(phylo_distances, 0)

phylo = PhylogeneticDiversityAnalyzer()

# Faith's Phylogenetic Diversity
faith_pd = phylo.faith_pd(
    species_data=data,
    phylogenetic_distances=phylo_distances
)

print(f"Faith's PD range: {faith_pd.min():.2f} - {faith_pd.max():.2f}")

# Phylogenetic endemism
phylo_endemism = phylo.phylogenetic_endemism(
    species_data=data,
    phylogenetic_distances=phylo_distances
)

print(f"Phylogenetic endemism range: {phylo_endemism.min():.3f} - {phylo_endemism.max():.3f}")
```

### Metacommunity Analysis

```python
from VegZ import MetacommunityAnalyzer

metacommunity = MetacommunityAnalyzer()

# Elements of metacommunity structure
ems_results = metacommunity.elements_metacommunity_structure(
    species_data=data,
    site_coordinates=coords
)

print("Metacommunity structure:")
print(f"  Coherence: {ems_results['coherence']:.3f}")
print(f"  Turnover: {ems_results['turnover']:.3f}")
print(f"  Boundary clumping: {ems_results['boundary_clumping']:.3f}")
```

### Network Analysis

```python
from VegZ import NetworkAnalyzer

network = NetworkAnalyzer()

# Co-occurrence network
cooccurrence_network = network.cooccurrence_network(
    species_data=data,
    correlation_threshold=0.3,
    p_value_threshold=0.05
)

print("Co-occurrence network:")
print(f"  Number of nodes: {cooccurrence_network['n_nodes']}")
print(f"  Number of edges: {cooccurrence_network['n_edges']}")
print(f"  Network density: {cooccurrence_network['density']:.3f}")

# Modularity analysis
modularity = network.modularity_analysis(cooccurrence_network['adjacency_matrix'])
print(f"  Modularity: {modularity['modularity']:.3f}")
print(f"  Number of modules: {modularity['n_modules']}")
```

## Machine Learning

### Species Distribution Modeling

```python
from VegZ import MachineLearningAnalyzer

ml = MachineLearningAnalyzer()

# Prepare presence/absence data
presence_data = (data > 0).astype(int)

# Random Forest species distribution model
sdm_results = ml.species_distribution_modeling(
    species_data=presence_data['Species_1'],
    environmental_data=env_data,
    model_type='random_forest',
    test_size=0.3,
    cross_validation=True
)

print("Species Distribution Model Results:")
print(f"  Accuracy: {sdm_results['accuracy']:.3f}")
print(f"  AUC: {sdm_results['auc']:.3f}")
print(f"  Feature importance: {dict(zip(env_data.columns, sdm_results['feature_importance']))}")
```

### Vegetation Classification

```python
# Create vegetation types for classification
vegetation_types = ['Forest', 'Grassland', 'Shrubland', 'Wetland'] * (n_sites // 4)
if len(vegetation_types) < n_sites:
    vegetation_types += ['Forest'] * (n_sites - len(vegetation_types))

# Classification model
classification_results = ml.vegetation_classification(
    species_data=data,
    vegetation_types=vegetation_types,
    model_type='random_forest',
    test_size=0.3
)

print("Vegetation Classification Results:")
print(f"  Accuracy: {classification_results['accuracy']:.3f}")
print("  Classification Report:")
print(classification_results['classification_report'])
```

### Ensemble Modeling

```python
# Ensemble species distribution model
ensemble_results = ml.ensemble_modeling(
    species_data=presence_data['Species_1'],
    environmental_data=env_data,
    models=['random_forest', 'gradient_boosting', 'logistic_regression'],
    test_size=0.3
)

print("Ensemble Model Results:")
print(f"  Ensemble accuracy: {ensemble_results['ensemble_accuracy']:.3f}")
print("  Individual model accuracies:")
for model, accuracy in ensemble_results['individual_accuracies'].items():
    print(f"    {model}: {accuracy:.3f}")
```

## Visualization

### Diversity Plots

```python
import matplotlib.pyplot as plt

# Plot diversity indices
veg.plot_diversity(diversity, index='shannon')
plt.title('Shannon Diversity Index')
plt.show()

# Multiple diversity indices
veg.plot_multiple_diversity(
    diversity,
    indices=['shannon', 'simpson', 'richness'],
    plot_type='boxplot'
)
plt.show()
```

### Ordination Plots

```python
# PCA biplot
veg.plot_ordination(
    pca_results,
    ordination_type='pca',
    color_by=cluster_labels,
    show_species=True,
    show_sites=True
)
plt.title('PCA Biplot')
plt.show()

# NMDS plot with environmental vectors
veg.plot_ordination(
    nmds_results,
    ordination_type='nmds',
    color_by=groups,
    environmental_vectors=vector_fit
)
plt.title('NMDS with Environmental Vectors')
plt.show()
```

### Clustering Visualizations

```python
# Dendrogram
clustering.plot_dendrogram(hier_results)
plt.title('Hierarchical Clustering Dendrogram')
plt.show()

# Elbow analysis plot (4-panel layout)
clustering.plot_elbow_analysis(elbow_results)
plt.show()

# Silhouette plot
clustering.plot_silhouette(silhouette)
plt.title('Silhouette Analysis')
plt.show()
```

### Species Response Curves

```python
# Plot GAM response curve
env_model.plot_species_response(
    gam_results,
    species_name='Species_1',
    env_var_name='pH'
)
plt.title('Species Response to pH')
plt.show()

# Multiple species responses
env_model.plot_multiple_responses(
    species_data=data.iloc[:, :5],  # First 5 species
    environmental_data=env_data['pH'],
    models=['gaussian', 'linear']
)
plt.show()
```

## Interactive Features

### Interactive Dashboards

```python
from VegZ import InteractiveVisualizer

interactive = InteractiveVisualizer()

# Create interactive ordination plot
interactive.interactive_ordination(
    ordination_results=nmds_results,
    color_by=cluster_labels,
    hover_data=env_data
)

# Interactive diversity plot
interactive.interactive_diversity_plot(
    diversity_results=diversity,
    site_metadata=env_data
)

# Interactive species accumulation curves
interactive.interactive_accumulation_curves(
    accumulation_data=accumulation
)
```

### Report Generation

```python
from VegZ import ReportGenerator

report = ReportGenerator()

# Generate comprehensive analysis report
analysis_results = {
    'diversity': diversity,
    'ordination': pca_results,
    'clustering': kmeans_results,
    'statistics': permanova_results
}

# HTML report
report.generate_html_report(
    analysis_results=analysis_results,
    output_file='vegetation_analysis_report.html',
    include_plots=True
)

# PDF report
report.generate_pdf_report(
    analysis_results=analysis_results,
    output_file='vegetation_analysis_report.pdf'
)

print("Reports generated successfully!")
```

## Data Quality and Validation

### Spatial Validation

```python
from VegZ import data_quality

# Validate coordinates
coord_validation = data_quality.validate_coordinates(
    coordinates=coords,
    coordinate_system='geographic',  # 'geographic' or 'projected'
    country_boundaries=True
)

print("Coordinate validation results:")
print(f"  Valid coordinates: {coord_validation['n_valid']}")
print(f"  Invalid coordinates: {coord_validation['n_invalid']}")
print(f"  Outliers detected: {coord_validation['n_outliers']}")

# Precision assessment
precision_results = data_quality.coordinate_precision_assessment(
    coordinates=coords
)

print(f"  Average precision: {precision_results['average_precision']:.6f}")
```

### Temporal Validation

```python
# Validate dates
date_validation = data_quality.validate_temporal_data(
    dates=phenology_data['date'],
    expected_range=['2020-01-01', '2020-12-31']
)

print("Temporal validation results:")
print(f"  Valid dates: {date_validation['n_valid']}")
print(f"  Invalid dates: {date_validation['n_invalid']}")
print(f"  Date range: {date_validation['date_range']}")
```

### Species Data Validation

```python
# Validate species matrix
species_validation = data_quality.validate_species_matrix(
    species_data=data,
    check_negatives=True,
    check_zeros=True,
    abundance_range=[0, 1000]
)

print("Species data validation:")
print(f"  Matrix completeness: {species_validation['completeness']:.2%}")
print(f"  Negative values: {species_validation['n_negative']}")
print(f"  Zero-abundance sites: {species_validation['n_zero_sites']}")
print(f"  Zero-abundance species: {species_validation['n_zero_species']}")
```

## Enhanced Species Name Error Detection

### Overview

VegZ 1.1.0 introduces a comprehensive species name error detection system that identifies and classifies 10+ categories of taxonomic name errors. This feature helps ensure data quality and taxonomic consistency in vegetation datasets.

### Basic Usage

```python
from VegZ.data_management.standardization import SpeciesNameStandardizer

# Initialize the standardizer
standardizer = SpeciesNameStandardizer()

# Validate a single species name
result = standardizer.validate_species_name("Quercus alba L.")
print(f"Valid: {result['is_valid']}")
print(f"Errors: {result['errors']}")
print(f"Suggestions: {result['suggestions']}")
```

### Error Categories Detected

1. **Incomplete Binomial Names**
   - Genus-only names ("Quercus")
   - Species epithet-only names ("alba")
   - Missing components

2. **Formatting Issues**
   - Capitalization errors ("quercus alba", "Quercus Alba")
   - Multiple consecutive spaces
   - Leading/trailing whitespace

3. **Author Citations**
   - Various citation formats ("L.", "Linnaeus", "(L.) Sweet")
   - Abbreviated authors
   - Author with year citations

4. **Hybrid Markers**
   - Multiplication symbol (×)
   - Letter x hybrid markers
   - Text-based hybrid indicators
   - Malformed hybrid names

5. **Infraspecific Ranks**
   - Subspecies (subsp.), varieties (var.)
   - Forms (f.), cultivars (cv.)
   - Incorrect formatting

6. **Placeholder Names**
   - Species placeholders (sp., spec.)
   - Confer placeholders (cf., aff.)
   - Unknown/indeterminate markers

7. **Invalid Characters**
   - Numbers in names
   - Special symbols and punctuation
   - Non-standard Unicode characters

### Individual Name Validation

```python
# Detailed validation with error analysis
test_names = [
    "Quercus alba",           # Valid binomial
    "Quercus",               # Genus only
    "quercus alba",          # Capitalization error
    "Quercus alba L.",       # Author citation
    "Quercus × alba",        # Hybrid marker
    "Quercus sp.",           # Placeholder
    "Quercus alba!",         # Invalid character
]

for name in test_names:
    result = standardizer.validate_species_name(name)
    print(f"\n'{name}':")
    print(f"  Valid: {result['is_valid']}")
    print(f"  Error count: {result['error_count']}")
    print(f"  Severity: {result['severity']}")
    print(f"  Cleaned: '{result['cleaned_name']}'")

    if result['errors']:
        print("  Error categories:")
        for category, errors in result['errors'].items():
            if errors:  # Only show categories with errors
                print(f"    {category}: {errors}")
```

### Batch Validation

```python
import pandas as pd

# Create a DataFrame with species names
species_list = [
    "Quercus alba", "Pinus strobus", "quercus sp.",
    "Acer saccharum L.", "Unknown species", "Betula × nigra"
]

# Batch validation returns a comprehensive DataFrame
results_df = standardizer.batch_validate_names(species_list)

print("Batch validation results:")
print(f"Shape: {results_df.shape}")
print(f"Columns: {list(results_df.columns)}")

# Summary statistics
valid_count = results_df['is_valid'].sum()
total_count = len(results_df)
print(f"\nValidation summary:")
print(f"Valid names: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")

# Error distribution
error_columns = [col for col in results_df.columns if col.startswith('has_')]
print(f"\nError distribution:")
for col in error_columns:
    error_count = results_df[col].sum()
    error_type = col.replace('has_', '')
    print(f"  {error_type}: {error_count} names")
```

### DataFrame Integration

```python
# Enhanced DataFrame standardization with error detection
vegetation_df = pd.DataFrame({
    'site_id': ['site_001', 'site_002', 'site_003'],
    'species': ['Quercus alba', 'quercus sp.', 'Pinus strobus L.'],
    'abundance': [25, 12, 18]
})

# Standardize with full error detection (new default)
enhanced_df = standardizer.standardize_dataframe(
    vegetation_df,
    species_column='species'
)

print("Enhanced standardization:")
print(f"Original columns: {list(vegetation_df.columns)}")
print(f"Enhanced columns: {list(enhanced_df.columns)}")

# Access error detection results
error_summary = enhanced_df[['species', 'name_is_valid', 'name_error_count', 'name_severity']]
print("\nError summary:")
print(error_summary)

# Backward compatibility mode (minimal columns)
simple_df = standardizer.standardize_dataframe(
    vegetation_df,
    species_column='species',
    include_error_detection=False
)

print(f"\nBackward compatible columns: {list(simple_df.columns)}")
```

### Comprehensive Error Reports

```python
# Generate detailed error report for a dataset
df = pd.DataFrame({'species': species_list})

report = standardizer.generate_error_report(df, species_column='species')

print("=== COMPREHENSIVE ERROR REPORT ===")
print(f"\nDataset Summary:")
for key, value in report['summary'].items():
    print(f"  {key}: {value}")

print(f"\nError Statistics:")
for category, stats in report['error_statistics'].items():
    print(f"  {category}: {stats['count']} ({stats['percentage']:.1f}%)")

print(f"\nSeverity Distribution:")
for severity, stats in report['severity_distribution'].items():
    print(f"  {severity}: {stats['count']} ({stats['percentage']:.1f}%)")

print(f"\nRecommendations:")
for i, recommendation in enumerate(report['recommendations'], 1):
    print(f"  {i}. {recommendation}")

# Access detailed validation results
detailed_results = report['detailed_results']
critical_errors = detailed_results[detailed_results['severity'] == 'critical']
print(f"\nCritical errors found in {len(critical_errors)} names")
```

### Name Type Classification

```python
# Classify different types of taxonomic names
name_types = [
    "Quercus alba",              # binomial
    "Quercus",                   # genus_only
    "alba",                      # epithet_only
    "Quercus alba var. alba",    # infraspecific
    "Quercus × alba",            # hybrid
    "Quercus alba L.",           # binomial_with_author
    "Quercus sp.",               # placeholder
    "Unknown species",           # placeholder
]

for name in name_types:
    name_type = standardizer.classify_name_type(name)
    print(f"'{name}' -> {name_type}")
```

### Error Severity Levels

The system classifies errors into four severity levels:

- **Critical**: Names that cannot be used for analysis (missing genus, incomplete binomials)
- **High**: Names with significant issues (multiple error types, placeholders)
- **Medium**: Names with moderate issues (author citations, hybrid markers)
- **Low**: Names with minor issues (spacing, minor formatting)
- **None**: Valid names with no errors

```python
# Filter by severity level
severe_errors = results_df[results_df['severity'].isin(['critical', 'high'])]
print(f"Names requiring immediate attention: {len(severe_errors)}")

for idx, row in severe_errors.iterrows():
    print(f"  '{row['original_name']}' - {row['severity']} ({row['error_count']} errors)")
```

### Integration with Data Processing Workflows

```python
# Complete data processing workflow with error detection
def process_vegetation_data_with_validation(raw_data, species_column):
    """Process vegetation data with comprehensive species name validation."""

    # Initialize standardizer
    standardizer = SpeciesNameStandardizer()

    # Step 1: Standardize species names with error detection
    standardized_data = standardizer.standardize_dataframe(
        raw_data,
        species_column=species_column
    )

    # Step 2: Generate quality report
    error_report = standardizer.generate_error_report(
        raw_data,
        species_column=species_column
    )

    # Step 3: Filter data based on quality criteria
    high_quality_data = standardized_data[
        standardized_data['name_is_valid'] == True
    ]

    # Step 4: Identify problematic records
    problematic_data = standardized_data[
        standardized_data['name_severity'].isin(['critical', 'high'])
    ]

    print(f"Processing complete:")
    print(f"  Total records: {len(standardized_data)}")
    print(f"  High quality: {len(high_quality_data)}")
    print(f"  Problematic: {len(problematic_data)}")
    print(f"  Validity rate: {error_report['summary']['validity_percentage']:.1f}%")

    return {
        'standardized_data': standardized_data,
        'high_quality_data': high_quality_data,
        'problematic_data': problematic_data,
        'error_report': error_report
    }

# Example usage
results = process_vegetation_data_with_validation(vegetation_df, 'species')
```

## Quick Functions

VegZ provides quick functions for immediate results:

### Quick Diversity Analysis

```python
from VegZ import quick_diversity_analysis

# Instant diversity calculation
quick_diversity = quick_diversity_analysis(
    data=data,
    indices=['shannon', 'simpson', 'richness']
)

print("Quick diversity analysis completed")
print(f"Average Shannon diversity: {quick_diversity['shannon'].mean():.3f}")
```

### Quick Ordination

```python
from VegZ import quick_ordination

# Rapid PCA
quick_pca = quick_ordination(
    data=data,
    method='pca',
    transform='hellinger',
    n_components=3
)

print("Quick PCA completed")
print(f"Explained variance: {quick_pca['explained_variance_ratio']}")
```

### Quick Clustering

```python
from VegZ import quick_clustering

# Fast clustering
quick_clusters = quick_clustering(
    data=data,
    method='kmeans',
    n_clusters=4,
    transform='hellinger'
)

print("Quick clustering completed")
print(f"Cluster sizes: {np.bincount(quick_clusters['cluster_labels'])}")
```

### Quick Elbow Analysis

```python
from VegZ import quick_elbow_analysis

# Rapid optimal k determination
quick_elbow = quick_elbow_analysis(
    data=data,
    max_k=10,
    transform='hellinger',
    plot_results=True
)

print(f"Quick elbow analysis: optimal k = {quick_elbow['optimal_k']}")
```

## Best Practices

### Data Preparation

1. **Data Format**: Ensure data is in site-by-species matrix format
2. **Missing Values**: Handle missing values appropriately
3. **Data Transformation**: Choose appropriate transformation for your analysis
4. **Zero Values**: Consider the ecological meaning of zeros (true absence vs. not detected)

```python
# Example of proper data preparation
def prepare_vegetation_data(raw_data):
    """Prepare vegetation data for analysis."""
    
    # Remove sites with no species
    raw_data = raw_data.loc[raw_data.sum(axis=1) > 0]
    
    # Remove species not present in any site
    raw_data = raw_data.loc[:, raw_data.sum(axis=0) > 0]
    
    # Check for negative values
    if (raw_data < 0).any().any():
        print("Warning: Negative values detected")
    
    # Fill NaN with zeros (if appropriate for your data)
    raw_data = raw_data.fillna(0)
    
    return raw_data

# Use the function
clean_data = prepare_vegetation_data(data)
```

### Analysis Workflow

1. **Exploratory Analysis**: Start with diversity indices and basic ordination
2. **Data Transformation**: Test different transformations
3. **Method Selection**: Choose appropriate methods for your research questions
4. **Validation**: Use cross-validation and permutation tests
5. **Interpretation**: Consider ecological meaning of results

```python
# Recommended workflow
def vegetation_analysis_workflow(data, env_data=None):
    """Complete vegetation analysis workflow."""
    
    results = {}
    
    # Step 1: Diversity analysis
    veg = VegZ()
    veg.data = data
    veg.species_matrix = data
    
    diversity = veg.calculate_diversity(['shannon', 'simpson', 'richness'])
    results['diversity'] = diversity
    
    # Step 2: Ordination
    pca_results = veg.pca_analysis(transform='hellinger')
    results['ordination'] = pca_results
    
    # Step 3: Clustering with elbow analysis
    clustering = VegetationClustering()
    elbow_results = clustering.comprehensive_elbow_analysis(data)
    optimal_k = elbow_results['recommendations']['consensus']
    
    clusters = clustering.kmeans_clustering(data, n_clusters=optimal_k)
    results['clustering'] = clusters
    
    # Step 4: Statistical tests (if groups available)
    if env_data is not None:
        stats = EcologicalStatistics()
        groups = clusters['cluster_labels']
        
        permanova = stats.permanova(
            species_data=data, 
            groups=groups,
            permutations=999
        )
        results['statistics'] = permanova
    
    return results
```

### Performance Considerations

1. **Large Datasets**: Use appropriate algorithms for large datasets
2. **Memory Usage**: Monitor memory usage with large matrices
3. **Computation Time**: Use parallel processing where available

```python
# Example for large datasets
def handle_large_dataset(data, chunk_size=1000):
    """Handle large datasets efficiently."""
    
    if len(data) > chunk_size:
        # Process in chunks
        results = []
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i+chunk_size]
            chunk_results = quick_diversity_analysis(chunk)
            results.append(chunk_results)
        
        # Combine results
        return pd.concat(results)
    else:
        return quick_diversity_analysis(data)
```

## Troubleshooting

### Common Issues

1. **Import Errors**
```python
# Check if VegZ is properly installed
try:
    import VegZ
    print(f"VegZ version: {VegZ.__version__}")
except ImportError:
    print("VegZ not installed. Run: pip install VegZ")
```

2. **Data Format Issues**
```python
# Check data format
def check_data_format(data):
    """Check if data is in correct format."""
    
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame")
    
    if data.isnull().any().any():
        print("Warning: Missing values detected")
    
    if (data < 0).any().any():
        print("Warning: Negative values detected")
    
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtypes.unique()}")
    
    return True
```

3. **Memory Issues**
```python
# Monitor memory usage
import psutil

def check_memory():
    """Check available memory."""
    memory = psutil.virtual_memory()
    print(f"Available memory: {memory.available / (1024**3):.2f} GB")
    print(f"Memory usage: {memory.percent}%")
```

4. **Convergence Issues**
```python
# Handle non-convergent algorithms
def robust_nmds(data, max_attempts=5):
    """Robust NMDS with multiple attempts."""
    
    multivar = MultivariateAnalyzer()
    
    for attempt in range(max_attempts):
        try:
            nmds_results = multivar.nmds_analysis(
                data, 
                random_state=attempt
            )
            
            if nmds_results['converged']:
                return nmds_results
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
    
    raise RuntimeError("NMDS failed to converge after multiple attempts")
```

### Error Messages

**"Data contains NaN values"**
- Solution: Use `data.fillna(0)` or remove rows/columns with NaN

**"Insufficient data for analysis"**
- Solution: Ensure minimum sample size requirements

**"Algorithm did not converge"**
- Solution: Adjust parameters or try different starting points

**"Memory error"**
- Solution: Reduce data size or use chunking approach

## API Reference

### Core Classes

- `VegZ`: Main analysis class
- `DiversityAnalyzer`: Diversity calculations
- `MultivariateAnalyzer`: Ordination methods
- `VegetationClustering`: Clustering algorithms
- `EcologicalStatistics`: Statistical tests
- `EnvironmentalModeler`: Environmental modeling
- `TemporalAnalyzer`: Temporal analysis
- `SpatialAnalyzer`: Spatial analysis
- `FunctionalTraitAnalyzer`: Trait analysis
- `MachineLearningAnalyzer`: ML methods
- `InteractiveVisualizer`: Interactive plots
- `ReportGenerator`: Report creation

### Key Functions

**Data Management:**
- `load_data()`: Load vegetation data
- `transform_data()`: Data transformation
- `validate_data()`: Data validation

**Diversity:**
- `calculate_diversity()`: Diversity indices
- `hill_numbers()`: Hill numbers
- `beta_diversity()`: Beta diversity
- `rarefaction_analysis()`: Rarefaction curves

**Ordination:**
- `pca_analysis()`: Principal Component Analysis
- `nmds_analysis()`: Non-metric MDS
- `cca_analysis()`: Canonical Correspondence Analysis
- `environmental_fit()`: Environmental vector fitting

**Clustering:**
- `twinspan()`: TWINSPAN analysis
- `comprehensive_elbow_analysis()`: Elbow analysis
- `hierarchical_clustering()`: Hierarchical clustering
- `kmeans_clustering()`: K-means clustering

**Statistics:**
- `permanova()`: PERMANOVA test
- `anosim()`: ANOSIM test
- `mantel_test()`: Mantel test
- `indicator_species_analysis()`: IndVal analysis

### Quick Functions

- `quick_diversity_analysis()`
- `quick_ordination()`
- `quick_clustering()`
- `quick_elbow_analysis()`

---

**VegZ Manual Version 1.1.0**
**Copyright (c) 2025 Mohamed Z. Hatim**
**For support: https://github.com/mhatim99/VegZ/issues**