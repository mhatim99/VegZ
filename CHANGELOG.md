# Changelog

All notable changes to VegZ will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.0] - 2026-08-02

The largest release so far, and it does two things.

It **adds the methods that were missing**: the tests ecologists reach for after
an ordination, the beta-diversity and rarefaction machinery that has become
standard since 2010, and a container that stops sites and species silently
falling out of alignment.

It also lands a **full scientific and engineering audit** of the existing code
base. That audit found methods that ran without error but returned wrong
numbers; every one is corrected here and pinned by a regression test that checks
it against an analytically known answer rather than against previously recorded
output.

### Added

#### Statistics
- **`permdisp()`** (alias **`betadisper()`**) - Anderson's (2006) test of
  homogeneity of multivariate dispersions, the assumption PERMANOVA depends on.
  Distances to the group centroid are computed in principal-coordinate space,
  with negative eigenvalues subtracted as in `vegan`. Supports centroid or
  spatial-median (Weiszfeld) centring, a permutation test, and Holm-adjusted
  pairwise comparisons. On univariate Euclidean data the F statistic reproduces
  `scipy.stats.levene(center='mean')` to ten decimal places.
- **`adonis()`** - multi-factor PERMANOVA on a model formula. Handles crossed
  and nested designs, `a*b` interaction expansion, sequential (Type I) and
  marginal (Type III) sums of squares, continuous and categorical terms, and
  `strata` for restricted permutation. Reproduces `permanova()` exactly in the
  one-way case and classical sequential ANOVA sums of squares exactly on
  univariate data.

#### Ordination
- **`anova_cca()`** and **`anova_rda()`** - permutation significance tests for
  constrained ordinations, in `vegan`'s `anova.cca` idiom: overall test, by
  axis, by term (sequential) and by margin. On univariate data the pseudo-F
  matches the regression F to ten decimal places.
- **`varpart()`** - variance partitioning across two or three explanatory
  tables (Peres-Neto et al. 2006), reporting unique, shared and unexplained
  fractions with adjusted R-squared. The fractions sum to exactly 1.0.
- **`forward_selection()`** - permutation-based forward selection of
  explanatory variables with Blanchet et al. (2008) double stopping criterion.

#### Diversity
- **`beta_partition()`** and **`beta_partition_multisite()`** - Baselga (2010,
  2012) partitioning of beta diversity into turnover and nestedness-resultant
  components, for both Sorensen (beta_sim / beta_sne) and Jaccard
  (beta_jtu / beta_jne) families. The two components sum to the total exactly.
- **`hill_rarefaction()`** - Chao et al. (2014) rarefaction and extrapolation of
  Hill numbers for q = 0, 1, 2. Interpolation at q = 0 equals the Hurlbert
  rarefaction exactly; extrapolation is continuous with the observed value at
  the reference sample size and monotone beyond it.
- **`sample_coverage()`**, **`coverage_at_size()`** and
  **`coverage_standardized_diversity()`** - Chao & Jost (2012) coverage-based
  standardisation, which compares assemblages at equal completeness rather than
  equal sampling effort.

#### Data handling
- **`VegData`** - a container that keeps a species matrix, an environmental
  table, a trait table and a site-metadata table aligned to one common set of
  sites and species. Misaligned tables are the most common source of silently
  wrong community analyses; `VegData` intersects them once, in a stable order,
  and reports exactly which labels each table lost. Provides `subset()`,
  `drop_empty()`, `transform()`, `summary()`, `from_files()` and `to_vegz()`.
- `DiversityAnalyzer.rarefaction_curve()` - exact Hurlbert rarefaction computed
  in log space, with variance.
- `DiversityAnalyzer.species_accumulation_curve()` - permutation-based curve
  with confidence bands.
- `DiversityAnalyzer.gini_simpson()` and `whittaker_beta()`.
- `VegZ.species_accumulation_curve()`.
- `MultivariateAnalyzer.pcoa_analysis(correction=...)` - Lingoes and Cailliez
  corrections for negative eigenvalues.
- PROTEST permutation test in `procrustes_analysis()`.
- `_compat` module providing a scikit-learn MDS shim (1.8 renamed `metric` to
  `metric_mds` and `dissimilarity` to `metric`) and optional-import helpers.
- Top-level exports for `VegetationPlotter`, `DataTransformer`,
  `DataStandardizer`, `SpeciesNameStandardizer`, `SpatialValidator` and
  `TemporalValidator`.
- `excel`, `network`, `timeseries` and `all` optional-dependency extras.
- `tests/test_scientific_correctness.py` - regression tests pinning every
  corrected formula against a known analytical result.

### Fixed

#### Corrected formulas

Methods that ran cleanly and returned wrong numbers. Each is now pinned by a
test against an analytically known answer.

#### Ordination
- **`nmds_analysis()` performed *metric* MDS, not NMDS.** `VegZ.nmds_analysis()`
  constructed `sklearn.manifold.MDS` without `metric=False`, so despite the name
  it ran classical metric scaling. It now performs genuine non-metric MDS, with
  `n_init`, `max_iter` and `random_state` exposed.
- **`procrustes_analysis()` always raised `ImportError`** - `procrustes` was
  imported from `scipy.spatial.distance` instead of `scipy.spatial`. It now
  works, and adds a PROTEST permutation test.
- **Correspondence analysis eigenvalues were inflated by the grand total.**
  CA now uses the standard proportion-based formulation, so eigenvalues are
  bounded by 1 and the total inertia equals chi-square / N (verified against
  `scipy.stats.chi2_contingency`). Species scores now carry the missing
  column-mass weighting.
- **DCA gradient lengths were always exactly 4.0.** Axes were standardised to
  unit SD and the gradient length then computed as `4 * SD`, which is 4 by
  construction. DCA now rescales axes to species-turnover SD units (correcting
  for weighted-averaging shrinkage), so gradient lengths track real beta
  diversity and the conventional 4-SD rule of thumb applies.
- **CCA was not a canonical correspondence analysis.** Replaced with the
  standard weighted-regression formulation (ter Braak 1986), producing LC and WA
  site scores, environmental biplot scores and a constrained/unconstrained
  inertia partition.
- **RDA reported raw squared singular values as eigenvalues** and returned an
  environment score matrix of the wrong shape. Eigenvalues are now variances,
  environmental scores are axis correlations, and the constrained/unconstrained
  variance partition plus an adjusted R-squared are reported.
- **PCoA** now uses a symmetric eigensolver, reports the negative-eigenvalue
  fraction, supports Lingoes/Cailliez corrections, and accepts `manhattan`
  (previously an error).
- **`environmental_fitting()`** returned unnormalised regression coefficients as
  "direction cosines" and a placeholder p-value on the fallback path. Vectors
  are now unit length scaled by sqrt(R-squared), with permutation p-values.
- **`goodness_of_fit_test()`** compared distances on incompatible scales; stress
  is now computed after an optimal linear rescaling.

#### Statistics
- **PERMANOVA's sum-of-squares decomposition was wrong.** Within-group sums of
  squares were multiplied by a spurious `n_g / N` weight, so the pseudo-F
  statistic, R-squared and p-value were all incorrect. The corrected
  implementation reproduces the classical one-way ANOVA F exactly on univariate
  Euclidean distances.
- **ANOSIM's R statistic used a non-standard denominator** and was not bounded
  by 1. It now uses Clarke's `M / 2` denominator; perfectly separated groups
  give R = 1.
- **Indicator Species Analysis (IndVal) used summed rather than mean
  abundances**, making the specificity component sensitive to unequal group
  sizes. Now the Dufrene-Legendre definition. `VegZ.indicator_species_analysis()`
  was a two-group approximation and now delegates to the real statistic.
- **SIMPER** gained contribution percentages, cumulative percentages, standard
  deviations and consistency ratios, and no longer silently mislabels a
  Euclidean decomposition.

#### Diversity
- **The ACE estimator used `sum(i * f_i)` instead of `sum(i * (i-1) * f_i)`** in
  its coefficient-of-variation correction, which reduces to `N_rare` and
  collapses the estimator.
- **`beta_diversity()` returned a scalar for `whittaker`** and a matrix for the
  other methods. It now returns a dissimilarity matrix for every method;
  `whittaker_beta()` provides the whole-dataset scalar.
- McIntosh diversity no longer divides by zero for single-individual samples.

#### Clustering
- **TWINSPAN never divided anything.** The reciprocal-averaging power iteration
  converged on the trivial constant eigenvector, so every division was rejected
  and all sites were returned in a single group. The divisive step now uses the
  first non-trivial correspondence-analysis axis with indicator-species
  refinement; on synthetic data with three known vegetation types it recovers
  them exactly.
- **The elbow curve's k=1 point was a mean variance, not a sum of squares**, so
  it was not on the same scale as `KMeans.inertia_` and the curve was not
  monotone decreasing - undermining every elbow detector.
- Elbow analysis, its plots and the best-k recommendations assumed `k_range`
  started at 1 and returned wrong answers otherwise.
- **Cophenetic correlation was computed against Euclidean distances regardless
  of the metric used to build the tree**, and called `cophenet()` with one
  argument. It now uses the distances the linkage was actually built from.
- `_distortion_jump_method()` computed a plain second difference; it now
  implements the actual Sugar & James (2003) transformed-distortion jump.
- `optimal_clusters_analysis()` had an off-by-one in the elbow index.
- TWINSPAN's site classification was built from an uninitialised integer Series.

#### Nestedness
- **NODF divided only by the pairs that happened to have decreasing fill**,
  inflating the metric (badly so with many tied marginal totals). It now
  averages over all pairs, giving exactly 100 for a perfectly nested matrix and
  0 when no fill gradient exists.
- **Matrix temperature was an O(n^2 m^2) loop unrelated to the published
  metric.** Replaced with a vectorised Atmar & Patterson isocline calculation.
- Permutation p-values now use `(count + 1) / (n + 1)` and are capped at 1.

#### Functional traits
- **FEve was negative by construction** - its numerator `sum(min(x) - x)` is
  always <= 0. Replaced with Villeger et al. (2008) minimum-spanning-tree
  evenness, which gives exactly 1 for a perfectly even community.
- **FDiv was a coefficient of variation, not functional divergence**, and was
  unbounded. Now the Villeger definition, bounded in [0, 1].
- Rao's quadratic entropy vectorised; FRic falls back more gracefully.

#### Spatial
- **Moran's I omitted the `n / S0` normalisation** and used a placeholder
  standard error of `sqrt(1/n)`. It now uses the correct normalisation and the
  randomisation-assumption variance.
- **Geary's C used `n` where the definition requires `n - 1`**, so its expected
  value was not 1. Now correct, with a z-score and p-value.
- **The contagion index used the wrong normalising constant and raw joint
  adjacency proportions.** Now the FRAGSTATS formulation: 100 for a
  single-class landscape, near 0 for a random one.
- Edge density, landscape shape index and patch compactness counted cells with a
  non-zero image gradient rather than measuring perimeter length; LSI now
  returns exactly 1.0 for a square.

#### Temporal
- **Mann-Kendall did not sort by time**, so its S statistic depended on row
  order in the input table. Also gained Sen's slope confidence intervals.
- **The double-sigmoid phenology model was monotonically increasing** - the
  senescence term was written as a decreasing logistic and then subtracted, so
  it could not represent a growing season at all.
- Phenology models are now given data-driven starting values; previously
  `curve_fit` defaulted to all-ones, which is infeasible against the day-of-year
  bounds. The Weibull amplitude bound was too tight to ever fit day-of-year data.
- Seasonality auto-detection matched frequency aliases by substring, classifying
  quarterly data (`QE-DEC`) as daily.

#### Environmental modelling
- **Spline smoothers required sorted input** and crashed or misfitted on
  unsorted predictors; duplicates are now averaged.
- **Fitted values inherited the predictor's dtype**, silently truncating to
  integers for integer predictors.
- Spline smoothing selection maximised R-squared, which always picks zero
  smoothing (interpolating the noise); it now uses generalised cross-validation.
  The Gaussian-process smoother gained a noise term for the same reason.
- An unrecognised GAM `family` raised `UnboundLocalError`; it now raises a clear
  `ValueError`.
- Species response curves now get data-driven initial values and bounds and
  recover known parameters.

#### Machine learning
- **`habitat_suitability_modeling()` raised `TypeError`** - `cross_val_score()`
  has no `random_state` parameter. Reproducibility now comes from a seeded CV
  splitter.
- Automatic cluster-count selection had an off-by-one that raised `IndexError`
  on small data sets.

#### Data management and quality
- **The chord and log-chord transforms raised `ValueError` for DataFrames**
  (multi-dimensional indexing of a Series) and the chi-square transform raised
  "truth value of a Series is ambiguous". Both now work; the chi-square
  transform also implements the Legendre & Gallagher formula, so Euclidean
  distances on the transformed table equal chi-square distances.
- **The z-score outlier method raised `NameError`** - `scipy.stats` was never
  imported in `spatial_validation`.
- **Country derivation was gated behind GeoPandas** although its implementation
  is pure NumPy.
- CSV/Excel parsers raised "got multiple values for keyword" when `sep`,
  `encoding` or `sheet_name` were passed through `**kwargs`.
- Turboveg parsers mutated `df.columns.values` in place, which pandas treats as
  undefined behaviour.
- `generate_spatial_quality_report()` raised `KeyError` when coordinate columns
  were absent.
- `integrate_datasets()` added a `source_dataset` column to the caller's
  DataFrames.

#### Reproducibility
- **Constructing `NestednessAnalyzer`, `NullModels` or
  `CommunityAssemblyAnalyzer` called `np.random.seed()`**, silently resetting
  NumPy's global random state for the rest of the user's program. All modules
  now use private `numpy.random.Generator` instances.
- Species and site alignment used `set(a) & set(b)`, whose iteration order
  varies between runs under Python's string-hash randomisation, making trait and
  phylogenetic results non-reproducible. Alignment is now order-preserving.
- `random_state` added to PERMANOVA, ANOSIM, MRPP, Mantel, partial Mantel,
  IndVal, fuzzy c-means, the gap statistic, envfit and Procrustes.

#### Defects found while writing the new tests

- **Taxonomic resolution ignored `use_fallback` on the no-match path.** The flag
  was only honoured after an exception, so a source that simply returned no
  match still fell through to the next database. With `use_fallback=False` the
  resolver now stops at the first source, as documented.
- **Country derivation from coordinates silently returned all-NaN.** The output
  column was created from `np.nan`, making it float64; assigning country names
  into a float column raises in pandas >= 2.2, and the exception was swallowed.
  The column is now created as object dtype.
- **An unrecognised geographic-outlier method returned "no outliers".** A typo
  in `method=` produced a clean, empty, wrong result. It now raises.
- **`generate_temporal_quality_report()` crashed on date ranges over ~292
  years.** Subtracting two `Timestamp`s further apart than the `Timedelta`
  limit overflows; the span is now computed via `datetime.datetime`.
- **Temporal cross-validation flags used a deprecated pandas assignment path.**
  Masked assignment of a subset-indexed Series triggered a dtype upcast that
  pandas will make an error; flags are now assigned by label.
- **A custom `fill` value for vegetation indices was clipped away.** `_finish()`
  substituted the fill before clipping, so a sentinel such as `-999` came back
  as `-1`. Clipping now runs first, leaving sentinels distinguishable from real
  measurements.
- **`plot_ordination(color_by=...)` crashed on a categorical grouping** and, for
  a string argument, silently coloured points by row order. Categorical values
  now get discrete colours and a legend; a bare string raises, since there is no
  data frame in scope to resolve a column name against.
- **Interactive dashboards returned an empty dict in silence** when the results
  passed in used different key names. All four now warn and name the keys they
  looked for. The trait dashboard additionally gained the matplotlib fallback
  the other three already had.
- **Markdown and HTML reports crashed on DataFrame-valued results** - the Jinja
  templates tested a DataFrame for truthiness.
- **Small-world sigma divided by zero** whenever the random reference graph was
  triangle-free, aborting the whole network-property block. It is now estimated
  from a seeded ensemble of Erdos-Renyi graphs, skipping degenerate draws, and
  returns `None` if no usable reference exists. This also makes it reproducible:
  the previous single draw used the global RNG.
- **t-SNE failed on typical vegetation data sets** - sklearn's default
  perplexity of 30 requires more than 30 sites. It is now derived from the
  sample size, and `**kwargs` are forwarded to the reducer.
- **A late-binding closure in the MODIS extractor** captured the loop variable,
  so every point could be evaluated at the last coordinate.
- **Co-occurrence networks crashed on pandas 3.** The self-associations were
  zeroed by writing through `DataFrame.values`, which is a mutable view only by
  accident in pandas 2 and is read-only under pandas 3's copy-on-write. Species
  could therefore be joined to themselves depending on the pandas version.
- **Date parsing crashed on pandas 3** for dates outside the nanosecond range.
  pandas 3 parses them at second resolution instead of coercing to `NaT`, and
  writing those into the nanosecond accumulator raised `OutOfBoundsDatetime`.
  Dates outside 1677-2262 are now reported as unparseable on every pandas
  version, rather than the behaviour depending on which one is installed.

### Changed

- `machine_learning.prepare_data()` one-hot encodes categorical predictors
  instead of dropping them (`encode_categorical=True`).
- `habitat_suitability_modeling()` chooses a classifier or a regressor from the
  target data rather than assuming classification, accepts `class_weight`, and
  raises a clear error on a single-class target.
- `VegetationIndexCalculator` was rewritten around an `available_indices`
  registry and a `compute()` dispatcher, adding EVI2, GNDVI, NDRE and ARVI.
  Mismatched band shapes now raise instead of broadcasting.
- `calculate_distances()` documents that all three methods return **metres**.
- Python 3.8 is no longer supported (end of life); `requires-python` is now
  `>=3.9`.
- **`mantel_test()` and `partial_mantel_test()` now default to
  `alternative='greater'`**, the one-sided ecological convention (matching
  vegan's `mantel`). Pass `alternative='two-sided'` for the previous behaviour.
- `DiversityAnalyzer.calculate_all_indices()` no longer includes the
  dataset-level `jack1`/`jack2` estimators by default, since broadcasting a
  single number across every row reads as a per-sample estimate. Pass
  `include_dataset_level=True` to restore them.
- `simpson` remains Simpson's *concentration* `sum(p^2)`; the complementary
  Gini-Simpson index is now available as `gini_simpson`.
- `VegZ.nmds_analysis()` defaults to `transform='none'`. Bray-Curtis already
  relativises, so the previous Hellinger default changed what the distance
  measured.
- `filter_rare_species()` no longer prints; pass `verbose=True` or read
  `VegZ.metadata['filter_rare_species']`.
- `VegetationPlotter` no longer changes global matplotlib/seaborn style on
  import or construction. Pass `apply_style_globally=True` for the old behaviour.

### Removed

- Import-time warnings. `import VegZ` is now silent: optional dependencies are
  resolved quietly and warn only when a feature that needs them is used.
- Hard dependencies on `xlrd`, `openpyxl` and `fuzzywuzzy`, which were imported
  at module scope but declared only as optional extras (or not at all), so
  `import VegZ` failed without them. Excel support now raises an informative
  error on use; fuzzy matching falls back to the standard library's `difflib`.
- Deprecated pandas arguments (`infer_datetime_format`, `fillna(method=...)`).
- Use of `scipy.spatial.distance_matrix`, deprecated in SciPy 1.18 and scheduled
  for removal in 1.20; replaced with `scipy.spatial.distance.cdist`. Without
  this, spatial autocorrelation and variogram analysis would have broken
  outright on SciPy 1.20.
- The stale duplicate source tree that shipped inside `dist/`, and the
  `examples/data/` directory (a byte-identical copy of `test_data/`).

### Infrastructure

- **GitHub Actions CI** across Python 3.9-3.13 on Linux, macOS and Windows,
  with dedicated jobs for the oldest declared dependency floors, a
  no-optional-dependencies install (asserting the package imports silently and
  that the extras really are absent), pre-release dependencies, lint
  (ruff + mypy), and a build job that runs the tests from the installed sdist.
- **`py.typed`** - the package now ships its type information.
- **Test coverage raised from 40% to 81%**, with 647 tests. New suites cover
  functional traits, spatial analysis, specialized methods, temporal analysis,
  environmental modelling, nestedness, plotting, dashboards, remote sensing,
  coordinate systems and temporal validation.
- The suite passes on both **pandas 2.3 and pandas 3.0** (and NumPy 2.4/2.5,
  scikit-learn 1.7/1.9, SciPy 1.16/1.18).

### Documentation

- `docs/index.md` gains sections on `VegData`, PERMDISP, `adonis`, the
  constrained-ordination tests, variance partitioning, forward selection,
  beta partitioning and coverage standardisation. Every example in them was
  executed against the built wheel before release.
- `examples/demo.py` now runs the full inferential workflow: PERMANOVA with its
  PERMDISP assumption check, turnover/nestedness partitioning, RDA permutation
  tests and variance partitioning. The synthetic environment includes a pure
  noise variable, so you can see the marginal tests correctly reject it.
- `examples/` is now covered by the CI lint job; the scripts ship in the sdist
  and had accumulated an unused import and two mid-file imports.

## [1.3.0] - 2025-01-19

### Added

#### Taxonomic Name Resolution System
- **TaxonomicResolver class** - Complete online species name validation and resolution system
- **Five taxonomic database integrations**:
  - WFO (World Flora Online) - Default source, comprehensive plant checklist
  - POWO (Plants of the World Online) - Kew Gardens authoritative database
  - IPNI (International Plant Names Index) - Nomenclatural verification
  - ITIS (Integrated Taxonomic Information System) - Standardized classification
  - GBIF (Global Biodiversity Information Facility) - Catalogue of Life backbone
- **Flexible source selection**: Single source, multiple sources, or fallback chain
- **File-based resolution**: Direct processing of CSV, Excel, TSV files
- **DataFrame integration**: Seamless integration with data analysis workflows

#### New Methods and Functions
- `TaxonomicResolver.resolve_names()` - Resolve list of species names
- `TaxonomicResolver.resolve_from_file()` - Resolve names directly from files
- `TaxonomicResolver.resolve_dataframe()` - Update species names in DataFrames
- `TaxonomicResolver.export_results()` - Export to CSV, Excel, JSON, TSV, Parquet, HTML
- `TaxonomicResolver.get_summary()` - Generate resolution statistics
- `TaxonomicResolver.print_summary()` - Display formatted summary
- `resolve_species_names()` - Convenience function for quick resolution
- `resolve_species_from_file()` - Convenience function for file-based resolution
- `update_species_in_dataframe()` - Convenience function for DataFrame updates

#### Resolution Output Features
- Original and accepted species names
- Author citations
- Match confidence scores (0-100)
- Match type classification (exact, fuzzy, candidate, synonym)
- Taxonomic status (accepted, synonym, unresolved)
- Synonym lists from source databases
- Family and genus extraction
- Source database identification
- Direct URLs to source records

#### Auto-detection and Usability
- Automatic species column detection in files and DataFrames
- Smart column name matching (species, scientific_name, taxon, etc.)
- Configurable minimum score threshold for name updates
- Original names preserved in separate column
- Taxonomy columns added automatically (family, genus, match_score, status, source)

### Fixed

#### Diversity Index Calculations
- Fixed division by zero in Pielou's evenness when richness equals 0 or 1
- Fixed division by zero in Margalef index when total abundance equals 1
- Fixed division by zero in ACE estimator when N_rare equals 1
- Fixed empty row handling in Shannon diversity calculation
- Fixed empty row handling in Simpson diversity calculation

#### Statistical Methods
- Fixed SIMPER between-group contributions formula (removed incorrect multiplication)
- Fixed SIMPER within-group contributions formula (corrected dissimilarity calculation)
- Fixed IndVal p-value computation (now correctly computes max across all groups per permutation)
- Fixed partial Mantel test Spearman correlation implementation

#### Multivariate Analysis
- Fixed DCA axis range division by zero when all sites have identical scores

#### Code Quality
- Fixed Jackknife estimator return type mismatch in calculate_all_indices
- Fixed bare except clauses with proper exception types (ValueError, TypeError)
- Fixed nestedness matrix binarization (now uses > 0 instead of int truncation)
- Fixed Sorensen distance function encoding (removed special character)

### Changed
- Added `requests>=2.25.0` to core dependencies for API communication

### Technical Improvements
- API request caching to minimize redundant calls
- Configurable request delays to respect rate limits
- Session-based HTTP connections for efficiency
- Comprehensive error handling for network failures
- Progress reporting for batch processing

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