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
fisher_alpha = diversity.calculate_index(data, 'fisher_alpha')

# Berger-Parker dominance
berger_parker = diversity.calculate_index(data, 'berger_parker')

# McIntosh diversity
mcintosh = diversity.calculate_index(data, 'mcintosh')

# Brillouin diversity
brillouin = diversity.calculate_index(data, 'brillouin')

# Menhinick and Margalef indices
menhinick = diversity.calculate_index(data, 'menhinick')
margalef = diversity.calculate_index(data, 'margalef')
```

### Richness Estimators

```python
# Chao1 estimator
chao1 = diversity.chao1_estimator(data)

# ACE estimator
ace = diversity.ace_estimator(data)

# Jackknife estimators
jackknife1 = diversity.jackknife1_estimator(data)
jackknife2 = diversity.jackknife2_estimator(data)

# All estimators using calculate_all_indices
all_estimators = diversity.calculate_all_indices(data)
# Or calculate specific ones:
# chao1 = diversity.calculate_index(data, 'chao1')
# ace = diversity.calculate_index(data, 'ace')
# jack1 = diversity.calculate_index(data, 'jack1')
# jack2 = diversity.calculate_index(data, 'jack2')
```

### Hill Numbers

```python
# Calculate Hill numbers for multiple orders
hill_numbers = diversity.hill_numbers(
    data,
    q_values=[0, 0.5, 1, 1.5, 2]  # q values
)

# Hill numbers interpretation:
# q=0: Species richness
# q=1: Shannon diversity (exponential)
# q=2: Simpson diversity (inverse)
```

### Beta Diversity

```python
# Whittaker's beta diversity (returns single value)
whittaker_beta = diversity.beta_diversity(data, method='whittaker')
print(f"Whittaker's beta: {whittaker_beta:.4f}")

# Pairwise beta diversity matrices
sorensen_beta = diversity.beta_diversity(data, method='sorensen')
jaccard_beta = diversity.beta_diversity(data, method='jaccard')

print(f"Sørensen beta matrix shape: {sorensen_beta.shape}")
print(f"Jaccard beta matrix shape: {jaccard_beta.shape}")
print(f"Average Sørensen beta: {sorensen_beta.values[sorensen_beta.values > 0].mean():.4f}")
print(f"Average Jaccard beta: {jaccard_beta.values[jaccard_beta.values > 0].mean():.4f}")
```

### Rarefaction Analysis

```python
from VegZ import VegZ

# Initialize VegZ with data
veg = VegZ()
veg.data = data
veg.species_matrix = data

# Rarefaction curves
rarefaction = veg.rarefaction_curve(
    sample_sizes=range(1, 101, 5)  # Sample sizes to test
)

print(f"Rarefaction data shape: {rarefaction.shape}")
print(f"Columns: {list(rarefaction.columns)}")

# Plot species accumulation curves
fig = veg.plot_species_accumulation(rarefaction)
print("Species accumulation plot created")
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
dca_results = multivar.detrended_correspondence_analysis(
    data,
    segments=26  # Number of segments for detrending
)

print("DCA gradient lengths:", dca_results['gradient_lengths'])
print("DCA eigenvalues:", dca_results['eigenvalues'][:3])
```

### Canonical Correspondence Analysis (CCA)

```python
# Environmental data required (ensure no missing values and matching indices)
env_data = pd.DataFrame({
    'pH': np.random.normal(6.5, 0.8, n_sites),
    'Moisture': np.random.normal(50, 15, n_sites),
    'Temperature': np.random.normal(15, 5, n_sites)
}, index=data.index)  # Important: matching row indices

# CCA with environmental constraints
cca_results = multivar.cca_analysis(
    species_data=data,
    env_data=env_data,
    scaling=1  # 1 for sites, 2 for species
)

print("CCA Eigenvalues:", cca_results['eigenvalues'])
print("CCA keys:", list(cca_results.keys()))
```

### Redundancy Analysis (RDA)

```python
# RDA for linear relationships
rda_results = multivar.redundancy_analysis(
    species_data=data,
    env_data=env_data
)

print("RDA Eigenvalues:", rda_results['eigenvalues'])
print("RDA keys:", list(rda_results.keys()))
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
vector_fit = multivar.environmental_fitting(
    ordination_scores=pca_results['site_scores'],
    env_data=env_data,
    method='vector'  # 'vector' or other fitting methods
)

print("Environmental vectors:")
print(f"Method: {vector_fit['method']}")
print(f"R-squared values: {vector_fit['r_squared']}")
print(f"P-values: {vector_fit['p_values']}")
print(f"Environmental vectors keys: {list(vector_fit['environmental_vectors'].keys())}")
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
fuzzy_results = clustering.fuzzy_cmeans_clustering(
    data,
    n_clusters=4,
    fuzziness=2.0,
    max_iter=100
)

print("Fuzzy membership matrix shape:", fuzzy_results['membership_matrix'].shape)
print("Fuzziness parameter:", fuzzy_results['fuzziness_parameter'])
print("Method:", fuzzy_results['method'])
```

### DBSCAN Clustering

```python
# Density-based clustering
dbscan_results = clustering.dbscan_clustering(
    data,
    eps=5.0,  # Adjusted for vegetation data scale
    min_samples=2,
    distance_metric='euclidean'
)

print("Number of clusters:", dbscan_results['n_clusters'])
print("Number of noise points:", dbscan_results['n_noise_points'])
print("Core samples:", dbscan_results['core_samples'].sum())
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
# MRPP analysis - first calculate distance matrix
from scipy.spatial.distance import pdist, squareform

# Calculate Bray-Curtis distance matrix
distances = pdist(data.values, metric='braycurtis')
distance_matrix = squareform(distances)

mrpp_results = stats.mrpp(
    distance_matrix=distance_matrix,
    groups=groups,
    permutations=999
)

print(f"MRPP delta: {mrpp_results['delta']:.4f}")
print(f"MRPP A-statistic: {mrpp_results['a_statistic']:.4f}")
print(f"p-value: {mrpp_results['p_value']:.4f}")
```

### Mantel Tests

```python
# Create distance matrices (square matrices required)
env_distances = squareform(pdist(env_data.values, metric='euclidean'))
species_distances = squareform(pdist(data.values, metric='braycurtis'))

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
    groups=clusters,  # Use clusters variable
    permutations=999
)

# Display indicator results
print("Indicator species analysis results:")
print(f"Number of species analyzed: {len(indval_results)}")

# Display results for each species
for species in list(indval_results.keys())[:3]:  # Show first 3 species
    print(f"{species}: {indval_results[species]}")
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
from VegZ.environmental import EnvironmentalModeler

env_model = EnvironmentalModeler()

# Combine species and environmental data
combined_data = pd.concat([data, env_data], axis=1)

# Fit GAM for species response
gam_results = env_model.fit_gam(
    data=combined_data,
    response_col='Species_1',
    predictor_cols=['pH', 'Moisture'],
    family='poisson'  # 'gaussian', 'poisson', 'binomial'
)

print("GAM Results:")
print(f"Response variable: {gam_results['response_variable']}")
print(f"Predictor variables: {gam_results['predictor_variables']}")
print(f"Family: {gam_results['family']}")
print(f"Number of observations: {gam_results['n_observations']}")
print(f"Diagnostics keys: {list(gam_results['diagnostics'].keys())}")

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
    data=phenology_data,
    time_col='date',
    response_col='abundance',
    model_type='sigmoid'  # 'sigmoid', 'gaussian', 'polynomial'
)

print(f"Phenology model type: {phenology_results['model_type']}")
print(f"Results keys: {list(phenology_results['results'].keys())}")
print(f"Data summary: {phenology_results['data_summary']}")
```

### Trend Detection

```python
# Mann-Kendall trend test
trend_results = temporal.trend_detection(
    data=phenology_data,
    time_col='date',
    response_col='abundance',
    method='mann_kendall'
)

print(f"Trend: {trend_results['trend']}")
print(f"p-value: {trend_results['p_value']:.4f}")
print(f"Sens slope: {trend_results['sens_slope']:.6f}")
print(f"Significant: {trend_results['significant']}")
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

# Create spatial data with proper column names
spatial_data = pd.DataFrame({
    'longitude': np.random.uniform(0, 100, n_sites),
    'latitude': np.random.uniform(0, 100, n_sites),
    'response': data['Species_1'].values
})

# Inverse Distance Weighting
idw_results = spatial.spatial_interpolation(
    data=spatial_data,
    x_col='longitude',
    y_col='latitude',
    z_col='response',
    method='idw',
    grid_resolution=0.5
)

print("IDW interpolation completed")
print(f"Grid shape: {idw_results['Z_grid'].shape}")

# Kriging interpolation
kriging_results = spatial.spatial_interpolation(
    data=spatial_data,
    x_col='longitude',
    y_col='latitude',
    z_col='response',
    method='kriging',
    grid_resolution=0.5
)

print("Kriging interpolation completed")
print(f"Available results: {list(kriging_results.keys())}")
```

### Landscape Metrics

```python
# Calculate landscape metrics from presence/absence data
binary_data = (data > 0).astype(int).values
landscape_array = binary_data.copy()

# Access individual landscape metrics methods
print("Landscape metrics:")

# Available methods in spatial.landscape_metrics dictionary:
# 'patch_density', 'edge_density', 'mean_patch_size', 'patch_size_cv',
# 'largest_patch_index', 'landscape_shape_index', 'contagion',
# 'shannon_diversity', 'simpson_diversity', 'evenness'

print(f"Available metrics: {list(spatial.landscape_metrics.keys())}")
```

### Spatial Autocorrelation

```python
# Moran's I test
morans_i = spatial.spatial_autocorrelation(
    data=spatial_data,
    x_col='longitude',
    y_col='latitude',
    response_col='response',
    method='morans_i'
)

print(f"Moran's I: {morans_i['morans_i']:.4f}")
print(f"p-value: {morans_i['p_value']:.4f}")
print(f"Expected I: {morans_i['expected_i']:.4f}")

# Geary's C test
gearys_c = spatial.spatial_autocorrelation(
    data=spatial_data,
    x_col='longitude',
    y_col='latitude',
    response_col='response',
    method='gearys_c'
)

print(f"Geary's C: {gearys_c['gearys_c']:.4f}")
```

## Functional Traits

### Trait Analysis

```python
from VegZ import FunctionalTraitAnalyzer

# Create trait data with species as index
trait_data = pd.DataFrame({
    'SLA': np.random.normal(20, 5, n_species),  # Specific Leaf Area
    'Height': np.random.lognormal(1, 0.5, n_species),  # Plant Height
    'SeedMass': np.random.lognormal(0, 1, n_species)  # Seed Mass
}, index=[f'Species_{i+1}' for i in range(n_species)])

traits = FunctionalTraitAnalyzer()

# Load data into the analyzer
traits.load_trait_data(trait_data)
traits.abundance_data = data

# Calculate functional diversity
func_diversity = traits.calculate_functional_diversity()

print("Functional diversity results:")
print(f"Available indices: {list(func_diversity['site_diversity'].columns)}")
print("Site diversity shape:", func_diversity['site_diversity'].shape)
```

### Functional Diversity

```python
# Functional diversity indices are already calculated above
site_diversity = func_diversity['site_diversity']

print("Functional diversity indices:")
for col in site_diversity.columns:
    print(f"  {col}: {site_diversity[col].mean():.3f}")

# Calculate functional beta diversity
beta_diversity = traits.calculate_functional_beta_diversity()

print(f"\nFunctional beta diversity:")
print(f"  Gamma diversity: {beta_diversity['gamma_diversity']:.3f}")
print(f"  Mean alpha diversity: {beta_diversity['mean_alpha_diversity']:.3f}")
print(f"  Beta diversity: {beta_diversity['beta_diversity']:.3f}")
```

### Trait Syndromes

```python
# Identify functional groups (trait syndromes)
functional_groups = traits.identify_functional_groups(
    n_groups=3,
    method='hierarchical'
)

print("Functional groups (trait syndromes):")
print(f"Number of groups: {functional_groups['n_groups']}")
print("Group characteristics:")
print(functional_groups['group_characteristics'])
```

### Trait-Environment Relationships

```python
# Trait-environment relationships (fourth-corner analysis)
trait_env = traits.trait_environment_relationships(
    environmental_data=env_data
)

print("Trait-environment relationships:")
print("Correlations shape:", trait_env['correlations'].shape)
print("Significant correlations:")
print(trait_env['significant_correlations'])
print("\nCommunity-weighted trait means:")
print(trait_env['cwm_traits'].head())
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

### Habitat Suitability Modeling

```python
from VegZ import MachineLearningAnalyzer

ml = MachineLearningAnalyzer()

# Prepare combined dataset with species and environmental data
presence_data = (data > 2).astype(int)  # Binary presence/absence
ml_data = pd.concat([presence_data, env_data], axis=1)

# Habitat suitability modeling (note: method may have internal issues in current version)
print("Available ML methods:")
print(f"  Methods: {[method for method in dir(ml) if not method.startswith('_')]}")

# Alternative: Use community classification for vegetation types
community_results = ml.community_classification(
    data=ml_data,
    species_columns=[f'Species_{i+1}' for i in range(5)],
    n_communities=3,
    method='kmeans'
)

print("Community Classification Results:")
print(f"  Number of communities: {community_results['n_communities']}")
print(f"  Cluster centers shape: {community_results['cluster_centers'].shape}")
```

### Biomass Prediction

```python
# Add biomass data for prediction
ml_data['biomass'] = np.random.exponential(50, n_sites)

# Biomass prediction using species and environmental data
biomass_results = ml.biomass_prediction(
    data=ml_data,
    biomass_column='biomass',
    predictor_features=[f'Species_{i+1}' for i in range(3)] + ['Temperature', 'pH'],
    model_type='rf',
    optimize_hyperparameters=False
)

print("Biomass Prediction Results:")
print(f"  Model performance: {biomass_results['performance']}")
print("  Feature importance:")
for feature, importance in zip(biomass_results['feature_names'], biomass_results['feature_importance']):
    print(f"    {feature}: {importance:.3f}")
```

### Species Identification

```python
# Add morphological features for species identification
ml_data['leaf_length'] = np.random.normal(5, 1, n_sites)
ml_data['leaf_width'] = np.random.normal(2, 0.5, n_sites)

# Create species labels
species_labels = np.random.choice(['Species_A', 'Species_B', 'Species_C'], n_sites)
ml_data['species_label'] = species_labels

# Species identification based on morphological features
identification_results = ml.species_identification(
    data=ml_data,
    morphological_features=['leaf_length', 'leaf_width'],
    species_column='species_label',
    test_size=0.3
)

print("Species Identification Results:")
print(f"  Best model: {identification_results['best_model']}")
print("  Model performance:")
for model, performance in identification_results['performance'].items():
    print(f"    {model}: accuracy = {performance.get('accuracy', 'N/A')}")
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

# Create diversity dashboard
diversity_dashboard = interactive.create_diversity_dashboard(
    diversity_results={'results': diversity, 'indices': ['shannon', 'simpson', 'richness']},
    data=data
)

print("Diversity dashboard created")
print(f"Dashboard components: {list(diversity_dashboard.keys())}")

# Create ordination dashboard
ordination_dashboard = interactive.create_ordination_dashboard(
    ordination_results=pca_results,
    environmental_data=env_data
)

print("Ordination dashboard created")
print(f"Dashboard components: {list(ordination_dashboard.keys())}")

# Create clustering dashboard
clustering_dashboard = interactive.create_clustering_dashboard(
    clustering_results=kmeans_results,
    ordination_results=pca_results
)

print("Clustering dashboard created")
```

### Report Generation

```python
from VegZ import ReportGenerator

report_gen = ReportGenerator()

# Prepare analysis results (use summary data instead of DataFrames)
analysis_results = {
    'diversity_summary': {
        'mean_shannon': diversity['shannon'].mean(),
        'mean_simpson': diversity['simpson'].mean(),
        'total_species': data.shape[1],
        'total_sites': data.shape[0]
    },
    'ordination_summary': {
        'method': 'PCA',
        'variance_explained': pca_results.get('variance_explained', [0.35, 0.25])
    },
    'data_shape': data.shape
}

# Generate HTML report
report_content = report_gen.generate_analysis_report(
    results=analysis_results,
    output_format='html'
)

# Save report to file
output_file = report_gen.save_report(
    report_content=report_content,
    filename='vegetation_analysis_report.html',
    format='html'
)

print("Report generated successfully!")
print(f"Report saved to: {output_file}")
print(f"Report length: {len(report_content)} characters")
```

## Data Quality and Validation

### Spatial Validation

```python
from VegZ.data_quality import SpatialValidator

# Initialize spatial validator
spatial_val = SpatialValidator()

# Create coordinate dataframe
coords_df = pd.DataFrame({
    'latitude': np.random.uniform(30, 45, n_sites),
    'longitude': np.random.uniform(-120, -100, n_sites)
})

# Validate coordinates
coord_validation = spatial_val.validate_coordinates(
    coords_df,
    lat_col='latitude',
    lon_col='longitude'
)

print("Coordinate validation results:")
print(f"  Total records: {coord_validation['total_records']}")
print(f"  Valid coordinates: {coord_validation['valid_coordinates']}")
print(f"  Issues found: {len(coord_validation['issues_found'])}")
print(f"  Precision assessment available: {'precision_assessment' in coord_validation}")
```

### Temporal Validation

```python
from VegZ.data_quality import TemporalValidator

# Initialize temporal validator
temp_val = TemporalValidator()

# Create temporal data
temporal_df = pd.DataFrame({
    'date': ['2020-01-15', '2020-02-20', '2020-13-05', '2020-05-30', 'invalid-date'],
    'event_date': ['2020-01-15', '2020-02-20', '2020-03-05', '2020-05-30', '2020-06-15']
})

# Validate dates
date_validation = temp_val.validate_dates(
    temporal_df,
    date_cols='date',
    event_date_col='event_date'
)

print("Temporal validation results:")
print(f"  Total records: {date_validation['total_records']}")
print(f"  Valid dates: {date_validation['valid_dates']}")
print(f"  Issues found: {len(date_validation['issues_found'])}")
print(f"  Date columns analyzed: {date_validation['date_columns_analyzed']}")
```

### Species Data Validation

```python
# Basic species matrix validation
print("Species data validation:")
print(f"  Matrix shape: {data.shape}")
print(f"  Matrix completeness: {(data > 0).sum().sum() / (data.shape[0] * data.shape[1]):.2%}")
print(f"  Negative values: {(data < 0).sum().sum()}")
print(f"  Zero-abundance sites: {(data.sum(axis=1) == 0).sum()}")
print(f"  Zero-abundance species: {(data.sum(axis=0) == 0).sum()}")
print(f"  Value range: {data.min().min():.2f} - {data.max().max():.2f}")
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
    print("Step 1: Diversity analysis completed")

    # Step 2: Ordination
    pca_results = veg.pca_analysis(transform='hellinger')
    results['ordination'] = pca_results
    print("Step 2: PCA analysis completed")

    # Step 3: Clustering analysis
    clustering = VegetationClustering()

    # Use default k=3 if comprehensive elbow analysis is not available
    try:
        elbow_results = clustering.comprehensive_elbow_analysis(data)
        optimal_k = elbow_results.get('recommendations', {}).get('consensus', 3)
    except:
        optimal_k = 3

    clusters = clustering.kmeans_clustering(data, n_clusters=optimal_k)
    results['clustering'] = clusters
    print(f"Step 3: Clustering completed with k={optimal_k}")

    # Step 4: Statistical tests (if environmental data available)
    if env_data is not None:
        from scipy.spatial.distance import pdist, squareform

        stats = EcologicalStatistics()
        groups = clusters['cluster_labels']

        # Calculate distance matrix for PERMANOVA
        distances = pdist(data, metric='braycurtis')
        distance_matrix = squareform(distances)

        permanova = stats.permanova(
            distance_matrix=distance_matrix,
            groups=groups,
            permutations=199  # Reduced for faster execution
        )
        results['statistics'] = permanova
        print("Step 4: PERMANOVA analysis completed")

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