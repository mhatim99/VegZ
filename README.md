# VegZ: Comprehensive Vegetation Data Analysis Package

[![PyPI version](https://badge.fury.io/py/VegZ.svg)](https://badge.fury.io/py/VegZ)
[![Python versions](https://img.shields.io/pypi/pyversions/VegZ.svg)](https://pypi.org/project/VegZ/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**VegZ** is a comprehensive, professional-grade Python package designed specifically for vegetation data analysis and environmental modeling. It provides a complete suite of tools for ecologists, environmental scientists, and researchers working with biodiversity and vegetation data.

> ### 1.5.1
>
> A metadata and documentation fix release; no analysis code changed. The
> declared dependency floors were not installable - VegZ imports `QhullError`
> from `scipy.spatial`, which SciPy only exposed in 1.8, so `import VegZ`
> failed on the advertised `scipy>=1.7`. The floors are now the oldest versions
> CI actually runs the suite against. See the
> [CHANGELOG](https://github.com/mhatim99/VegZ/blob/main/CHANGELOG.md).
>
> ### New in 1.5.0
>
> 1.5.0 adds the methods most often reached for after an ordination -
> **PERMDISP/betadisper**, **multi-factor PERMANOVA (`adonis`)**,
> **`anova.cca`-style permutation tests** for constrained ordinations,
> **variance partitioning**, **forward selection**, **Baselga turnover /
> nestedness partitioning**, and **coverage-based rarefaction and extrapolation
> of Hill numbers** - plus **`VegData`**, a container that keeps species,
> environment, trait and phylogeny tables aligned.
>
> It also lands a full scientific audit of the existing code base, correcting
> methods that ran without error but returned wrong numbers - among them NMDS
> (which was silently running *metric* MDS), the PERMANOVA sum-of-squares
> decomposition, the ANOSIM R statistic, TWINSPAN, NODF, FEve, FDiv, Moran's I,
> Geary's C and the ACE richness estimator. Every corrected formula is pinned by
> a test that checks it against an analytically known answer. See the
> [CHANGELOG](https://github.com/mhatim99/VegZ/blob/main/CHANGELOG.md).

## Complete Feature List

### Data Management & Preprocessing
- **`VegData` aligned container** (New in v1.5.0) - holds a species matrix with
  its environmental, trait and phylogeny tables, intersects them once in a
  stable order, and reports exactly which sites and species each table lost.
  Misaligned tables are the most common source of a confident, wrong community
  analysis
- Parse vegetation survey data from multiple formats (CSV, Excel, Turboveg)
- Integration with remote sensing APIs (Landsat, MODIS, Sentinel)
- Darwin Core biodiversity standards compliance
- Species name standardization with fuzzy matching
- Coordinate system transformations
- Multiple data transformation methods (Hellinger, chord, Wisconsin, log, sqrt, standardize)
- Automatic species matrix detection
- Support for heterogeneous data integration
- **Online Taxonomic Name Resolution** (New in v1.3.0):
  - Validate and update species names against 5 online databases
  - WFO (World Flora Online), POWO (Kew), IPNI, ITIS, GBIF
  - File-based and DataFrame integration
  - Confidence scores and synonym retrieval
- **Improved Ecological Terminology** (v1.2.0) - Domain-specific language:
  - Use of "sites" instead of generic "samples" for ecological sampling locations
  - Professional ecological nomenclature throughout the package

### Data Quality & Validation
- Comprehensive spatial coordinate validation
- Temporal data validation and date parsing
- Geographic outlier detection with country boundary checks
- Coordinate precision assessment
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
- **Basic indices**: Shannon, Simpson (concentration), Gini-Simpson, Simpson inverse, richness, evenness
- **Advanced indices**: Fisher's alpha, Berger-Parker, McIntosh, Brillouin
- **Additional indices**: Menhinick, Margalef
- **Richness estimators**: Chao1, ACE, Jackknife1, Jackknife2
- **Hill numbers** for multiple diversity orders (q = 0, 0.5, 1, 1.5, 2, etc.)
- **Beta diversity** analysis (Whittaker, Sørensen, Jaccard methods) - returns a
  pairwise dissimilarity matrix; `whittaker_beta()` gives the whole-dataset scalar
- **Rarefaction curves** - exact Hurlbert expectation with variance, computed in
  log space so it is stable for large counts
- **Species accumulation curves** with permutation confidence bands
- **Beta diversity partitioning** (New in v1.5.0) - Baselga (2010, 2012)
  separation into turnover (beta_sim / beta_jtu) and nestedness-resultant
  (beta_sne / beta_jne) components, pairwise and multi-site
- **Coverage-based standardization** (New in v1.5.0) - Chao & Jost (2012)
  sample coverage, and diversity compared at equal completeness rather than
  equal sampling effort
- **Hill number rarefaction and extrapolation** (New in v1.5.0) - Chao et al.
  (2014) interpolation/extrapolation for q = 0, 1, 2
- **Diversity profiles**

### Complete Multivariate Analysis Suite
- **PCA** - Principal Component Analysis with multiple transformations
- **CA** - Correspondence Analysis with scaling options
- **DCA** - Detrended Correspondence Analysis with segment control
- **CCA** - Canonical Correspondence Analysis with constraints
- **RDA** - Redundancy Analysis for linear relationships
- **NMDS** - genuine Non-metric Multidimensional Scaling with stress assessment
- **PCoA** - Principal Coordinates Analysis with Lingoes/Cailliez corrections for
  negative eigenvalues
- **Scientific Method Names** (New in v1.2.0) - Professional abbreviated method names:
  - `ca_analysis()` for Correspondence Analysis
  - `dca_analysis()` for Detrended Correspondence Analysis
  - `cca_analysis()` for Canonical Correspondence Analysis
  - `rda_analysis()` for Redundancy Analysis
  - `pcoa_analysis()` for Principal Coordinates Analysis
  - Full backward compatibility with existing method names
- **Constrained-ordination significance tests** (New in v1.5.0) - `anova_cca()`
  and `anova_rda()` in `vegan`'s `anova.cca` idiom: overall, by axis, by term
  (sequential) and by margin, with partial ordinations via `conditioning`
- **Variance partitioning** (New in v1.5.0) - `varpart()` across two or three
  explanatory tables (Peres-Neto et al. 2006) with adjusted R-squared
- **Forward selection** (New in v1.5.0) - permutation-based, with the Blanchet
  et al. (2008) double stopping criterion
- **Environmental vector fitting** (`envfit`-style) with unit-length direction
  vectors and permutation p-values
- **Procrustes analysis** for ordination comparison, with a PROTEST permutation test
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
- **Reproducible by default** - every stochastic method accepts `random_state`

### Statistical Analysis
- **PERMANOVA** - Permutational multivariate analysis of variance
- **PERMDISP / betadisper** (New in v1.5.0) - Anderson (2006) test of
  homogeneity of multivariate dispersions, the assumption PERMANOVA depends on;
  centroid or spatial-median centring, with Holm-adjusted pairwise comparisons
- **adonis** (New in v1.5.0) - multi-factor PERMANOVA from a model formula:
  crossed and nested designs, interactions, sequential (Type I) and marginal
  (Type III) sums of squares, continuous terms, and restricted permutation
  within `strata`
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
- **Phenology modeling** with sigmoid, double-sigmoid (Zhang et al. 2003),
  Gaussian, beta and Weibull curves, fitted from data-driven starting values
- **Trend detection** - linear, polynomial, spline and LOWESS trends, plus the
  Mann-Kendall test with Sen's slope and a distribution-free confidence interval
- **Time series decomposition** (seasonal, trend, residual) - classical, STL and
  X-11, with a dependency-free fallback when `statsmodels` is absent
- **Seasonal pattern analysis** with automatic period detection
- **Climate-vegetation response** analysis across user-specified lags
- **Growth curve fitting** - logistic, Gompertz, von Bertalanffy, exponential
  and power models

### Spatial Analysis
- **Spatial interpolation** methods, with leave-one-out cross-validated RMSE:
  - Inverse Distance Weighting (IDW)
  - Simple kriging with exponential, Gaussian or spherical variograms
  - Radial basis functions (thin-plate spline, multiquadric, Gaussian, linear)
  - Nearest neighbour, linear and cubic
- **Landscape metrics** calculation:
  - Patch density, mean patch size, patch size coefficient of variation
  - Edge density and largest patch index
  - Landscape shape index and contagion (O'Neill et al. 1988)
  - Shannon, Simpson and evenness indices for landscapes
- **Spatial autocorrelation** analysis (Moran's I with randomisation-based
  significance test, Geary's C, empirical variogram)
- **Habitat suitability modeling** from point occurrences (Random Forest, GLM)

### Specialized Methods
- **Phylogenetic diversity analysis**:
  - Faith's phylogenetic diversity
  - Mean Pairwise Distance (MPD) and Mean Nearest Taxon Distance (MNTD)
  - Net Relatedness Index (NRI)
  - Nearest Taxon Index (NTI)
- **Metacommunity analysis**:
  - Elements of metacommunity structure
  - Coherence, turnover, and boundary clumping
- **Network analysis**:
  - Co-occurrence networks from correlation or Jaccard association
  - Betweenness and eigenvector centrality, clustering coefficient
  - Connected components and small-world sigma against a random-graph ensemble
- **Nestedness analysis** with null models:
  - NODF (Nestedness based on Overlap and Decreasing Fill)
  - Temperature calculator
  - Null model generation and testing

### Functional Trait Analysis
- **Trait syndrome** identification
- **Community-weighted means** (CWM)
- **Functional diversity** indices:
  - Functional richness (FRic) - convex hull volume
  - Functional evenness (FEve) - minimum spanning tree based
  - Functional divergence (FDiv)
  - Functional dispersion (FDis)
  - Rao's quadratic entropy
- **Functional beta diversity** between sites
- **Trait-environment** relationships
- **Functional group** identification (hierarchical or k-means)

### Machine Learning & Predictive Modeling
- **Species Distribution Modeling** (SDM):
  - Random Forest models
  - Gradient Boosting models (LightGBM when installed)
  - Logistic regression, including a simplified MaxEnt-style variant
- **Classification algorithms** for vegetation types
- **Regression models** for abundance and biomass prediction
- **Anomaly detection** (Isolation Forest, DBSCAN)
- **Dimensionality reduction** (PCA, t-SNE) for exploration
- **Model validation** and performance metrics
- **Variable importance** assessment

### Visualization & Reporting
- **Specialized ecological plots**:
  - Diversity bar charts and histograms
  - Species accumulation curves
  - Rarefaction plots
- **Ordination diagrams** with:
  - Site scores plotting, coloured by a continuous or categorical variable
  - Species loading arrows
  - Environmental vector overlays
- **Clustering visualizations**:
  - Dendrograms with customizable formatting
  - Silhouette plots
  - **Comprehensive elbow analysis plots** (4-panel layout)
  - Cluster validation plots
- **Interactive dashboards** using Plotly, with static matplotlib fallbacks when
  it is not installed
- **Automated quality reports** with statistical summaries
- **Export** to HTML (dashboards and reports) and CSV. Plotting functions return
  matplotlib `Figure` objects, so `fig.savefig(...)` covers PNG, PDF and SVG

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

### Online Taxonomic Name Resolution (New in v1.3.0)

```python
from VegZ import TaxonomicResolver, resolve_species_names

# Quick resolution with default source (World Flora Online)
results = resolve_species_names(['Quercus robur', 'Pinus sylvestris'])

# Using specific source (GBIF)
resolver = TaxonomicResolver(sources='gbif')
results = resolver.resolve_names(['Quercus robur', 'Pinus sylvestris'])

# Multiple sources with fallback
resolver = TaxonomicResolver(
    sources=['wfo', 'powo', 'gbif'],
    use_fallback=True
)
results = resolver.resolve_names(['Quercus robur', 'Pinus sylvestris'])

# Resolve from file
results = resolver.resolve_from_file('species_list.csv')

# Update species names in your data
import pandas as pd
df = pd.read_csv('vegetation_data.csv')
df_updated = resolver.resolve_dataframe(df, species_column='species')

# Export results
resolver.export_results(results, 'resolved_names.xlsx')
resolver.print_summary(results)
```

Supported databases: WFO (World Flora Online), POWO (Plants of the World Online - Kew), IPNI (International Plant Names Index), ITIS (Integrated Taxonomic Information System), GBIF (Global Biodiversity Information Facility).

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
- Python >= 3.9
- NumPy >= 1.22.4
- Pandas >= 2.2.0
- SciPy >= 1.8.0
- Matplotlib >= 3.5.0
- scikit-learn >= 1.0.0
- Seaborn >= 0.11.0
- Requests >= 2.25.0

These are the oldest versions the suite is run against in CI, not estimates.

**Optional (for extended functionality):**

`import VegZ` never requires any of these and never warns about them; each is
only needed when you call a feature that uses it.

| Extra | Provides | Needed for |
|---|---|---|
| `VegZ[excel]` | openpyxl, xlrd | Reading `.xlsx` / `.xls` files |
| `VegZ[spatial]` | geopandas, pyproj, shapely, rasterio | Coordinate transforms, geospatial validation |
| `VegZ[fuzzy]` | fuzzywuzzy, python-Levenshtein | Faster fuzzy name matching (falls back to `difflib`) |
| `VegZ[network]` | networkx | Advanced co-occurrence network metrics |
| `VegZ[timeseries]` | statsmodels | LOWESS smoothing, STL decomposition |
| `VegZ[interactive]` | plotly, bokeh | Interactive dashboards |
| `VegZ[remote-sensing]` | earthengine-api, geemap, xarray | Remote sensing integration |
| `VegZ[all]` | the common subset of the above | Everything except remote sensing |

**Tested with:**
- Python 3.9 - 3.13
- All major operating systems (Windows, macOS, Linux)

## Reproducibility

Every stochastic routine accepts a `random_state`, and VegZ never mutates
NumPy's global random state:

```python
from VegZ.statistics import EcologicalStatistics

stats = EcologicalStatistics()
result = stats.permanova(distance_matrix, groups, permutations=999, random_state=42)
# Re-running with the same seed reproduces the p-value exactly.
```

`random_state` is available on PERMANOVA, ANOSIM, MRPP, Mantel and partial
Mantel tests, indicator species analysis, Procrustes/PROTEST, environmental
vector fitting, fuzzy c-means, the gap statistic, null models and species
accumulation curves.

## Validation

The corrected statistics are pinned by regression tests that check them against
analytically known answers rather than against previously recorded output:

- PERMANOVA reproduces the classical one-way ANOVA F exactly on univariate
  Euclidean distances
- PCoA on Euclidean distances reproduces PCA scores exactly
- CA total inertia x N equals `scipy.stats.chi2_contingency`'s chi-square
- ANOSIM R equals 1 for perfectly separated groups
- NODF equals 100 for a perfectly nested matrix and 0 with no fill gradient
- FEve equals 1 for a perfectly even community
- Geary's C equals 1 under no spatial autocorrelation
- Landscape shape index equals 1 for a square patch
- Rarefaction at full sample size returns the observed richness with zero variance
- TWINSPAN recovers three known vegetation types exactly (adjusted Rand = 1.0)
- PERMDISP's F equals `scipy.stats.levene(center='mean')` to ten decimal places
  on univariate Euclidean data
- `adonis()` reproduces `permanova()` exactly for one factor, and classical
  sequential ANOVA sums of squares exactly for two
- `anova_rda()`'s pseudo-F equals the regression F to ten decimal places on
  univariate data
- Variance-partitioning fractions sum to exactly 1.0 for two and three tables
- Baselga turnover and nestedness sum to the total beta diversity exactly
- Hill rarefaction at q = 0 equals Hurlbert rarefaction exactly, and
  extrapolation is continuous with the observed value at the reference size

The suite runs on Python 3.9-3.13 across Linux, macOS and Windows in CI, with
additional jobs for the oldest declared dependency floors, a
no-optional-dependencies install, and pre-release dependencies.

Run them with:

```bash
pytest tests/
```

## Scientific Background

VegZ implements methods from key ecological and statistical literature:

- **TWINSPAN**: Hill, M.O. (1979) TWINSPAN - A FORTRAN Program for Arranging Multivariate Data
- **Elbow Analysis**: Multiple algorithms including Satopaa et al. (2011) "Finding a kneedle in a haystack"
- **Ordination**: Methods from Legendre & Legendre "Numerical Ecology"
- **Diversity**: Comprehensive indices from Magurran "Measuring Biological Diversity"
- **Statistical tests**: Anderson (2001) PERMANOVA; Clarke (1993) ANOSIM
- **Indicator species**: Dufrêne & Legendre (1997) IndVal
- **Nestedness**: Almeida-Neto et al. (2008) NODF; Atmar & Patterson (1993) temperature
- **Functional diversity**: Villéger et al. (2008) FRic/FEve/FDiv; Laliberté & Legendre (2010) FDis
- **Constrained ordination**: ter Braak (1986) CCA
- **Landscape metrics**: O'Neill et al. (1988) contagion; FRAGSTATS conventions
- **Metacommunity structure**: Leibold & Mikkelson (2002) elements of metacommunity structure

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
    year = {2026},
    version = {1.5.0},
    url = {https://github.com/mhatim99/VegZ}
}
```

---

**VegZ** - *Empowering ecological research with comprehensive vegetation analysis tools.*