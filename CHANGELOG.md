# Changelog

All notable changes to VegZ will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-09-12

### Added

#### Core Functionality
- Complete VegZ core class with comprehensive vegetation analysis tools
- Support for CSV, Excel, and Turboveg data formats
- Automatic species matrix detection and data loading
- Multiple data transformation methods (Hellinger, log, sqrt, standardize)

#### Diversity Analysis
- DiversityAnalyzer class with 15+ diversity indices:
  - Basic: Shannon, Simpson, Simpson inverse, richness, evenness
  - Advanced: Fisher's alpha, Berger-Parker, McIntosh, Brillouin
  - Richness estimators: Chao1, ACE, Jackknife1, Jackknife2
  - Menhinick and Margalef indices
- Hill numbers calculation for multiple diversity orders
- Beta diversity analysis (Whittaker, Sørensen, Jaccard methods)
- Rarefaction curves and species accumulation analysis

#### Multivariate Analysis
- Complete ordination suite in MultivariateAnalyzer:
  - PCA (Principal Component Analysis)
  - CA (Correspondence Analysis)
  - DCA (Detrended Correspondence Analysis)
  - CCA (Canonical Correspondence Analysis)
  - RDA (Redundancy Analysis)
  - NMDS (Non-metric Multidimensional Scaling)
  - PCoA (Principal Coordinates Analysis)
- Environmental vector fitting to ordination axes
- Multiple ecological distance matrices (Bray-Curtis, Jaccard, Sørensen, etc.)
- Procrustes analysis for ordination comparison

#### Advanced Clustering Methods
- VegetationClustering class with comprehensive clustering tools:
  - **TWINSPAN** (Two-Way Indicator Species Analysis) - the gold standard
  - Hierarchical clustering with ecological distance matrices
  - **Comprehensive Elbow Analysis** with 5 detection algorithms:
    - **Knee Locator** (Kneedle algorithm) - Satopaa et al. (2011)
    - **Derivative Method** - Second derivative maximum
    - **Variance Explained** - <10% additional variance threshold
    - **Distortion Jump** - Jump method (Sugar & James, 2003)
    - **L-Method** - Piecewise linear fitting (Salvador & Chan, 2004)
  - Fuzzy C-means clustering for gradient boundaries
  - DBSCAN for core community detection
  - Gaussian Mixture Models
  - Clustering validation metrics (silhouette, gap statistic, Calinski-Harabasz)

#### Statistical Analysis
- EcologicalStatistics class with comprehensive tests:
  - PERMANOVA (Permutational multivariate analysis of variance)
  - ANOSIM (Analysis of similarities)
  - MRPP (Multi-response permutation procedures)
  - Mantel tests and partial Mantel tests
  - Indicator Species Analysis (IndVal)
  - SIMPER (Similarity percentages)

#### Environmental Modeling
- EnvironmentalModeler class with GAMs and gradient analysis:
  - Generalized Additive Models with multiple smoothers
  - Species response curves (Gaussian, beta, threshold, unimodal)
  - Environmental gradient analysis
  - Niche modeling capabilities

#### Temporal Analysis
- TemporalAnalyzer class for time series analysis:
  - Phenology modeling with multiple curve types
  - Trend detection (Mann-Kendall tests)
  - Time series decomposition
  - Seasonal pattern analysis

#### Spatial Analysis
- SpatialAnalyzer class for spatial ecology:
  - Spatial interpolation methods (IDW, kriging)
  - Landscape metrics calculation
  - Spatial autocorrelation analysis
  - Point pattern analysis

#### Specialized Methods
- PhylogeneticDiversityAnalyzer for phylogenetic analysis
- MetacommunityAnalyzer for metacommunity ecology
- NetworkAnalyzer for ecological network analysis
- NestednessAnalyzer with null models

#### Data Management & Quality
- Comprehensive data parsers for multiple formats
- Darwin Core biodiversity standards compliance
- Species name standardization with fuzzy matching
- Remote sensing integration (Landsat, MODIS, Sentinel APIs)
- Coordinate system transformations
- Spatial and temporal data validation
- Geographic outlier detection
- Quality assessment and reporting

#### Visualization & Reporting
- Specialized ecological plots
- Ordination diagrams with environmental vectors
- Diversity profiles and accumulation curves
- **Comprehensive elbow analysis plots** with 4-panel layout
- Interactive dashboards and visualizations
- Automated quality reports
- Export functions (HTML, PDF, CSV)

#### Quick Functions
- `quick_diversity_analysis()` for immediate diversity calculations
- `quick_ordination()` for rapid ordination analysis
- `quick_clustering()` for fast clustering
- `quick_elbow_analysis()` for optimal cluster determination

#### Examples and Documentation
- Comprehensive user manual (VEGLIB_MANUAL.md)
- Complete elbow analysis example with synthetic data
- Example datasets for testing and learning
- Detailed API documentation with usage examples

### Technical Features
- Professional package structure following Python packaging standards
- Comprehensive test suite with pytest
- Type hints throughout the codebase
- Robust error handling and validation
- Support for Python 3.8+
- Optional dependencies for extended functionality
- Modular design allowing use of individual components

### Dependencies
- **Core**: NumPy, Pandas, SciPy, Matplotlib, scikit-learn, Seaborn
- **Optional**: GeoPandas, PyProj, Rasterio, Earth Engine API, FuzzyWuzzy, Plotly/Bokeh

### Performance
- Optimized algorithms for large datasets
- Efficient memory usage with data transformations
- Vectorized operations using NumPy and Pandas
- Parallel processing support where applicable

### Standards Compliance
- Implements Darwin Core biodiversity standards
- Follows ecological analysis best practices
- Based on peer-reviewed scientific literature
- Professional code quality with comprehensive testing