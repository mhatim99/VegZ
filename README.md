# VegZ: Comprehensive Vegetation Data Analysis Package

[![PyPI version](https://badge.fury.io/py/VegZ.svg)](https://badge.fury.io/py/VegZ)
[![Python versions](https://img.shields.io/pypi/pyversions/VegZ.svg)](https://pypi.org/project/VegZ/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**VegZ** is a comprehensive, professional-grade Python package designed specifically for vegetation data analysis and environmental modeling. It provides a complete suite of tools for ecologists, environmental scientists, and researchers working with biodiversity and vegetation data.

## Complete Feature List

### Data Management & Preprocessing
- Parse vegetation survey data from multiple formats (CSV, Excel, Turboveg)
- Integration with remote sensing APIs (Landsat, MODIS, Sentinel)
- Darwin Core biodiversity standards compliance
- Species name standardization with fuzzy matching
- Coordinate system transformations
- Multiple data transformation methods (Hellinger, chord, Wisconsin, log, sqrt, standardize)
- Automatic species matrix detection
- Support for heterogeneous data integration
- **Improved Ecological Terminology** (New in v1.2.0) - Domain-specific language:
  - Use of "sites" instead of generic "samples" for ecological sampling locations
  - Professional ecological nomenclature throughout the package

### Data Quality & Validation
- Comprehensive spatial coordinate validation
- Temporal data validation and date parsing
- Geographic outlier detection with country boundary checks
- Coordinate precision assessment
- Duplicate record identification
- Invalid coordinate range detection
- Transposed coordinate detection
- Country boundary consistency checks
- Automated quality reporting
- **Enhanced Species Name Error Detection** (Introduced in v1.1.0):
  - 10+ error categories: incomplete binomial, formatting issues, author citations
  - Hybrid marker detection and validation
  - Infraspecific rank validation (var., subsp., f., cv.)
  - Placeholder name detection (sp., cf., aff., indet.)
  - Invalid character identification
  - Comprehensive error reporting with actionable suggestions
  - Batch processing capabilities for large datasets

### Diversity Analysis (15+ Indices)
- **Basic indices**: Shannon, Simpson, Simpson inverse, richness, evenness
- **Advanced indices**: Fisher's alpha, Berger-Parker, McIntosh, Brillouin
- **Additional indices**: Menhinick, Margalef
- **Richness estimators**: Chao1, ACE, Jackknife1, Jackknife2
- **Hill numbers** for multiple diversity orders (q = 0, 0.5, 1, 1.5, 2, etc.)
- **Beta diversity** analysis (Whittaker, Sørensen, Jaccard methods)
- **Rarefaction curves** and extrapolation
- **Species accumulation curves**
- **Diversity profiles**

### Complete Multivariate Analysis Suite
- **PCA** - Principal Component Analysis with multiple transformations
- **CA** - Correspondence Analysis with scaling options
- **DCA** - Detrended Correspondence Analysis with segment control
- **CCA** - Canonical Correspondence Analysis with constraints
- **RDA** - Redundancy Analysis for linear relationships
- **NMDS** - Non-metric Multidimensional Scaling with stress assessment
- **PCoA** - Principal Coordinates Analysis (metric MDS)
- **Scientific Method Names** (New in v1.2.0) - Professional abbreviated method names:
  - `ca_analysis()` for Correspondence Analysis
  - `dca_analysis()` for Detrended Correspondence Analysis
  - `cca_analysis()` for Canonical Correspondence Analysis
  - `rda_analysis()` for Redundancy Analysis
  - `pcoa_analysis()` for Principal Coordinates Analysis
  - Full backward compatibility with existing method names
- **Environmental vector fitting** to ordination axes
- **Procrustes analysis** for ordination comparison
- **Goodness-of-fit diagnostics**
- **Multiple ecological distance matrices** (Bray-Curtis, Jaccard, Sørensen, Euclidean, Manhattan, Canberra, Chord, Hellinger)

### Advanced Clustering Methods
- **TWINSPAN** - Two-Way Indicator Species Analysis (vegetation classification gold standard)
  - Pseudospecies creation with customizable cut levels
  - Hierarchical divisive classification
  - Indicator species identification
  - Classification tree structure
- **Hierarchical clustering** with ecological distance matrices
- **Comprehensive Elbow Analysis** with 5 detection algorithms:
  - **Kneedle algorithm** (Satopaa et al., 2011) - automatic knee detection
  - **Second derivative maximum** - curvature-based detection
  - **Variance explained threshold** - <10% additional variance criterion
  - **Distortion jump method** (Sugar & James, 2003) - jump detection
  - **L-method** (Salvador & Chan, 2004) - piecewise linear fitting
- **Consensus recommendations** with confidence scores
- **K-means clustering** with multiple initializations
- **Fuzzy C-means** clustering for gradient boundaries
- **DBSCAN** for density-based core community detection
- **Gaussian Mixture Models** for probabilistic clustering
- **Clustering validation** metrics (silhouette, gap statistic, Calinski-Harabasz, Davies-Bouldin)
- **Optimal k determination** with multiple methods

### Statistical Analysis
- **PERMANOVA** - Permutational multivariate analysis of variance
- **ANOSIM** - Analysis of similarities
- **MRPP** - Multi-response permutation procedures
- **Mantel tests** and partial Mantel tests for matrix correlation
- **Indicator Species Analysis** (IndVal) for cluster characterization
- **SIMPER** - Similarity percentages for group comparisons
- **Cophenetic correlation** for hierarchical clustering validation

### Environmental Modeling
- **Generalized Additive Models (GAMs)** with multiple smoothers:
  - Spline smoothers
  - LOWESS smoothers  
  - Polynomial smoothers
  - Gaussian process smoothers
- **Species response curves** modeling:
  - Gaussian response curves
  - Skewed Gaussian curves
  - Beta response curves
  - Linear responses
  - Threshold responses
  - Unimodal responses
- **Environmental gradient analysis**
- **Environmental niche modeling**

### Temporal Analysis
- **Phenology modeling** with multiple curve types
- **Trend detection** using Mann-Kendall tests
- **Time series decomposition** (seasonal, trend, residual)
- **Seasonal pattern analysis**
- **Temporal autocorrelation analysis**
- **Change point detection**

### Spatial Analysis
- **Spatial interpolation** methods:
  - Inverse Distance Weighting (IDW)
  - Kriging (ordinary, universal)
  - Spline interpolation
- **Landscape metrics** calculation:
  - Patch density
  - Edge density
  - Contagion index
  - Shannon diversity index for landscapes
- **Spatial autocorrelation** analysis (Moran's I, Geary's C)
- **Point pattern analysis**
- **Spatial clustering** detection

### Specialized Methods
- **Phylogenetic diversity analysis**:
  - Faith's phylogenetic diversity
  - Phylogenetic endemism
  - Net Relatedness Index (NRI)
  - Nearest Taxon Index (NTI)
- **Metacommunity analysis**:
  - Elements of metacommunity structure
  - Coherence, turnover, and boundary clumping
- **Network analysis**:
  - Co-occurrence networks
  - Modularity analysis
  - Network centrality measures
- **Nestedness analysis** with null models:
  - NODF (Nestedness based on Overlap and Decreasing Fill)
  - Temperature calculator
  - Null model generation and testing

### Functional Trait Analysis
- **Trait syndrome** identification
- **Community-weighted means** (CWM)
- **Functional diversity** indices:
  - Functional richness (FRic)
  - Functional evenness (FEve)
  - Functional divergence (FDiv)
  - Rao's quadratic entropy
- **Trait-environment** relationships
- **Fourth-corner analysis**

### Machine Learning & Predictive Modeling
- **Species Distribution Modeling** (SDM):
  - MaxEnt-style modeling
  - Random Forest models
  - Gradient Boosting models
- **Classification algorithms** for vegetation types
- **Regression models** for abundance prediction
- **Model validation** and performance metrics
- **Variable importance** assessment
- **Ensemble modeling**

### Visualization & Reporting
- **Specialized ecological plots**:
  - Diversity bar charts and histograms
  - Species accumulation curves
  - Rarefaction plots
- **Ordination diagrams** with:
  - Site scores plotting
  - Species loading arrows
  - Environmental vector overlays
  - Convex hulls for groups
  - Stress plots for NMDS
- **Clustering visualizations**:
  - Dendrograms with customizable formatting
  - Silhouette plots
  - **Comprehensive elbow analysis plots** (4-panel layout)
  - Cluster validation plots
- **Interactive dashboards** using Plotly/Bokeh
- **Automated quality reports** with statistical summaries
- **Export functions** (HTML, PDF, PNG, SVG, CSV)

### Quick Analysis Functions
- **quick_diversity_analysis()** - Instant diversity calculations
- **quick_ordination()** - Rapid PCA or NMDS analysis
- **quick_clustering()** - Fast k-means or hierarchical clustering
- **quick_elbow_analysis()** - Optimal cluster number determination

## Quick Start

### Installation

```bash
pip install VegZ
```

For extended functionality:
```bash
# With spatial analysis support
pip install VegZ[spatial]

# With remote sensing capabilities
pip install VegZ[remote-sensing]

# Complete installation with all features
pip install VegZ[spatial,remote-sensing,fuzzy,interactive]
```

### Basic Usage

```python
import pandas as pd
from VegZ import VegZ

# Initialize VegZ
veg = VegZ()

# Load your vegetation data
data = veg.load_data('vegetation_data.csv')

# Quick diversity analysis
diversity = veg.calculate_diversity(['shannon', 'simpson', 'richness'])

# Multivariate analysis
pca_results = veg.pca_analysis(transform='hellinger')
nmds_results = veg.nmds_analysis(distance_metric='bray_curtis')

# Advanced elbow analysis for optimal clustering
elbow_results = veg.elbow_analysis(
    k_range=range(1, 15),
    methods=['knee_locator', 'derivative', 'variance_explained'],
    plot_results=True
)
optimal_k = elbow_results['recommendations']['consensus']

# Clustering with optimal k
clusters = veg.kmeans_clustering(n_clusters=optimal_k)
indicators = veg.indicator_species_analysis(clusters['cluster_labels'])

# Create visualizations
veg.plot_diversity(diversity, 'shannon')
veg.plot_ordination(pca_results, color_by=clusters['cluster_labels'])
```

### Quick Functions for Immediate Results

```python
from VegZ import quick_diversity_analysis, quick_ordination, quick_elbow_analysis

# Instant analyses
diversity = quick_diversity_analysis(data, species_cols=['sp1', 'sp2', 'sp3'])
ordination = quick_ordination(data, method='pca')
elbow_results = quick_elbow_analysis(data, max_k=10, plot_results=True)
```

### Advanced TWINSPAN Analysis

```python
from VegZ.clustering import VegetationClustering

clustering = VegetationClustering()

# Two-Way Indicator Species Analysis - the gold standard for vegetation classification
twinspan_results = clustering.twinspan(
    species_data,
    cut_levels=[0, 2, 5, 10, 20],
    max_divisions=6,
    min_group_size=5
)

print("Site classification:", twinspan_results['site_classification'])
print("Indicator species:", twinspan_results['classification_tree']['indicator_species'])
```

### Enhanced Species Name Error Detection (Introduced in v1.1.0)

```python
from VegZ.data_management.standardization import SpeciesNameStandardizer

standardizer = SpeciesNameStandardizer()

# Validate individual species names
result = standardizer.validate_species_name("Quercus alba L.")
print(f"Valid: {result['is_valid']}")
print(f"Errors: {result['errors']}")
print(f"Suggestions: {result['suggestions']}")

# Batch validation of species names
import pandas as pd
df = pd.DataFrame({'species': ['Quercus alba', 'quercus sp.', 'Pinus × strobus']})
validated_df = standardizer.batch_validate_names(df['species'].tolist())

# Generate comprehensive error report
report = standardizer.generate_error_report(df, species_column='species')
print(f"Validity rate: {report['summary']['validity_percentage']}%")
```

## Data Format Requirements

VegZ expects data in **site-by-species matrix format**:

```csv
site_id,Species1,Species2,Species3,...
SITE_001,25,18,12,...
SITE_002,32,22,16,...
```

Environmental data should have matching site IDs:
```csv
site_id,latitude,longitude,elevation,soil_ph,temperature,...
SITE_001,44.2619,-72.5806,850,6.2,18.5,...
```

## Target Applications

- **Vegetation community classification** and mapping
- **Biodiversity assessments** and monitoring  
- **Environmental impact studies**
- **Species distribution modeling**
- **Ecological restoration planning**
- **Academic research** in plant ecology and environmental science

## Requirements

**Required:**
- Python >= 3.8
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0
- scikit-learn >= 1.0.0
- Seaborn >= 0.11.0

**Optional (for extended functionality):**
- GeoPandas (spatial analysis)
- PyProj (coordinate transformations)
- Earth Engine API (remote sensing)
- FuzzyWuzzy (fuzzy string matching)
- Plotly/Bokeh (interactive visualizations)

**Tested with:**
- Python 3.8 - 3.13
- All major operating systems (Windows, macOS, Linux)

## Scientific Background

VegZ implements methods from key ecological and statistical literature:

- **TWINSPAN**: Hill, M.O. (1979) TWINSPAN - A FORTRAN Program for Arranging Multivariate Data
- **Elbow Analysis**: Multiple algorithms including Satopaa et al. (2011) "Finding a kneedle in a haystack"
- **Ordination**: Methods from Legendre & Legendre "Numerical Ecology"
- **Diversity**: Comprehensive indices from Magurran "Measuring Biological Diversity"
- **Statistical tests**: From Anderson (2001) PERMANOVA and related methods

## Contributing

We welcome contributions! Please see the [Contributing Guide](https://github.com/mhatim99/VegZ/blob/main/CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/mhatim99/VegZ/blob/main/LICENSE) file for details.

## Support

- **GitHub Issues**: Report bugs or request features
- **Documentation**: Full user guide and API reference  
- **Email**: For academic collaborations and consulting

## Citation

If you use VegZ in your research, please cite:

```bibtex
@software{VegZ,
    author = {Hatim, Mohamed Z.},
    title = {VegZ: A comprehensive Python package for vegetation data analysis and environmental modeling},
    year = {2025},
    version = {1.2.0},
    url = {https://github.com/mhatim99/VegZ}
}
```

---

**VegZ** - *Empowering ecological research with comprehensive vegetation analysis tools.*