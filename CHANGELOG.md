# Changelog

All notable changes to VegZ will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2025-01-16

### Added

#### Scientific Method Abbreviations for Professional Use
- **Abbreviated multivariate analysis method names** for professional ecological workflow:
  - `ca_analysis()` - Correspondence Analysis (abbreviated from `correspondence_analysis()`)
  - `dca_analysis()` - Detrended Correspondence Analysis (abbreviated from `detrended_correspondence_analysis()`)
  - `cca_analysis()` - Canonical Correspondence Analysis (abbreviated from `canonical_correspondence_analysis()`)
  - `rda_analysis()` - Redundancy Analysis (abbreviated from `redundancy_analysis()`)
  - `pcoa_analysis()` - Principal Coordinates Analysis (abbreviated from `principal_coordinates_analysis()`)
- **Full backward compatibility** - all existing method names continue to work unchanged
- **Professional ecological nomenclature** following scientific conventions

#### Enhanced Ecological Terminology
- **Domain-specific terminology improvements**:
  - Use of "sites" instead of generic "samples" for ecological sampling locations
  - Improved consistency in ecological terminology throughout package
  - Enhanced professional language in method documentation
- **Systematic terminology standardization** across all modules
- **Maintains backward compatibility** with existing data structures

### Enhanced

#### Documentation and Examples
- **Complete manual verification** - all documentation examples tested and verified to work correctly
- **Fixed example inconsistencies**:
  - Corrected key structure differences between VegZ and MultivariateAnalyzer classes
  - Fixed environmental fitting examples to use correct result keys
  - Updated goodness of fit examples with proper key names
  - Corrected cumulative variance calculations for MultivariateAnalyzer
- **Comprehensive systematic testing** of all manual examples
- **Professional method name updates** in documentation

#### Code Quality and Consistency
- **Systematic package review** for terminology consistency
- **Method name standardization** following scientific abbreviation conventions
- **Enhanced error handling** for method compatibility
- **Improved API consistency** across all analysis modules

### Fixed
- **Manual example corrections**:
  - Fixed PCA key structure differences between classes (VegZ uses 'scores', MultivariateAnalyzer uses 'site_scores')
  - Corrected environmental vector fitting result parsing
  - Fixed goodness of fit correlation key references
  - Updated cumulative variance calculations for MultivariateAnalyzer class
- **Method accessibility issues** resolved through proper alias implementation
- **Documentation accuracy** - all examples now work correctly

### Technical Improvements
- **Comprehensive method compatibility** with backward compatibility aliases
- **Enhanced class structure consistency** between VegZ and MultivariateAnalyzer
- **Improved error detection and reporting** for method calls
- **Systematic testing framework** for documentation examples

## [1.1.0] - 2025-09-23

### Added

#### Comprehensive Species Name Error Detection & Classification
- **Complete error detection system** for taxonomic names with 10+ error categories
- **SpeciesNameStandardizer enhancements**:
  - `validate_species_name()` - Individual name validation with detailed error reporting
  - `classify_name_type()` - Taxonomic name type classification (binomial, hybrid, placeholder, etc.)
  - `batch_validate_names()` - Efficient batch processing with pandas DataFrame output
  - `generate_error_report()` - Statistical analysis and recommendations for datasets
  - `detect_errors()` - Core error detection with comprehensive classification

#### Error Detection Capabilities
- **Incomplete binomial names**: Detects genus-only and species-only entries
- **Formatting issues**: Identifies capitalization, spacing, and special character problems
- **Author citations**: Flags and removes 12+ different author citation patterns
- **Hybrid markers**: Handles ×, x, and text hybrid markers with malformation detection
- **Infraspecific ranks**: Validates var., subsp., f., cv., etc. with proper formatting checks
- **Anonymous/placeholder names**: Detects sp., cf., aff., indet., unknown, and 11+ similar patterns
- **Invalid characters**: Identifies numbers, symbols, and non-standard Unicode characters
- **Missing components**: Flags names with missing genus or species epithets

#### Enhanced DataFrame Processing
- **Optional error detection columns** in `standardize_dataframe()` method
- **16+ new columns** with detailed validation results:
  - `name_is_valid`, `name_error_count`, `name_severity`, `name_type`
  - Individual error category flags (e.g., `name_has_placeholder_names`)
  - `name_errors_summary` and `name_suggestions` for actionable insights
- **Backward compatibility mode** preserving original functionality

#### Advanced Pattern Recognition
- **Enhanced author patterns**: 12+ regex patterns for various citation formats
- **Infraspecific validation**: Dictionary-based marker validation with proper formatting
- **Hybrid detection**: Multiple hybrid marker patterns with malformation detection
- **Placeholder recognition**: 11+ patterns for anonymous/placeholder names
- **Unicode-aware validation**: Handles international characters and symbols

#### Error Classification System
- **Multi-level severity assessment**: Critical, High, Medium, Low, None
- **Detailed error categorization** with specific error types within categories
- **Actionable suggestions** for fixing detected errors
- **Statistical reporting** with error distribution analysis

#### Quality Assurance
- **100% backward compatibility** maintained - all existing functionality preserved
- **Comprehensive testing**: 53+ test cases covering all error types and edge cases
- **Performance optimization**: Efficient batch processing for large datasets
- **Integration testing**: Verified compatibility with main VegZ package ecosystem

### Enhanced

#### Data Management
- **SpeciesNameStandardizer class** significantly enhanced with error detection capabilities
- **DataStandardizer integration** automatically uses enhanced species name validation
- **Improved pattern matching** with optimized regex patterns for better performance

#### Documentation
- **Comprehensive examples** demonstrating new error detection features
- **Integration guides** for using enhanced functionality with existing workflows
- **Performance benchmarks** and usage recommendations

### Technical Improvements
- **Optimized regex patterns** for efficient pattern matching
- **Memory-efficient processing** for large datasets
- **Vectorized operations** where possible for improved performance
- **Modular architecture** allowing for future enhancements

### Documentation Updates
- **README.md**: Enhanced with comprehensive v1.1.0 feature documentation
- **VEGZ_MANUAL.md**: Added complete section on Enhanced Species Name Error Detection
- **Code examples**: Updated with new error detection functionality demonstrations
- **API documentation**: Expanded to cover all new validation methods

### Code Quality & Maintenance
- **Copyright standardization**: Updated all Python file comments to standardized copyright notices
- **Consistent licensing**: Ensured uniform copyright attribution across all source files
- **Package integrity**: Verified all files contain proper copyright and licensing information

## [1.0.3] - 2025-09-15

### Fixed
- **README**: Fixed Contributing Guide and LICENSE links to point to GitHub URLs for proper display on PyPI

## [1.0.2] - 2025-09-15

### Fixed
- **Documentation**: Corrected all import statements from `from vegz import` to `from VegZ import` across all documentation files
- **Documentation**: Fixed installation commands from `pip install vegz` to `pip install VegZ` in all documentation
- **Examples**: Updated import statements in demo.py and elbow_analysis_example.py
- **Tests**: Corrected import statements in test_core.py
- **Package consistency**: Ensured all documentation matches the correct PyPI package name 'VegZ'

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