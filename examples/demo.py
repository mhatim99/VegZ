#!/usr/bin/env python3
"""
VegZ Demo Script

Demonstrates the main VegZ workflow on synthetic vegetation data: diversity,
elbow analysis for the number of clusters, clustering, indicator species,
ordination, and the hypothesis tests that make those results interpretable -
PERMANOVA with its PERMDISP assumption check, beta-diversity partitioning,
constrained-ordination permutation tests and variance partitioning.

The synthetic data has four known vegetation types and an environment in which
elevation and soil pH track those types while aspect is pure noise, so you can
check the methods recover what is actually there.

Run with: python examples/demo.py
"""

import numpy as np
import pandas as pd


def build_synthetic_data(seed: int = 42) -> pd.DataFrame:
    """Four vegetation types, each with its own characteristic species."""
    rng = np.random.default_rng(seed)
    n_sites_per_type, n_species = 15, 20

    blocks = []
    for type_index, (background, signal) in enumerate(
            [(0.4, 12), (0.3, 15), (0.5, 10), (0.2, 18)]):
        block = rng.poisson(background, (n_sites_per_type, n_species)).astype(float)
        # Give each vegetation type five species it is characterised by.
        start = type_index * 5
        block[:, start:start + 5] = rng.poisson(signal, (n_sites_per_type, 5))
        blocks.append(block)

    return pd.DataFrame(
        np.vstack(blocks),
        index=[f'SITE_{i + 1:03d}' for i in range(4 * n_sites_per_type)],
        columns=[f'Species_{i + 1:02d}' for i in range(n_species)],
    )


def build_synthetic_environment(sites, seed: int = 7) -> pd.DataFrame:
    """
    Environmental variables, two of which track the vegetation types.

    Elevation and pH follow the four blocks, so they should carry real
    explanatory power; aspect is noise, so it should not.
    """
    rng = np.random.default_rng(seed)
    n_per_type = len(sites) // 4

    return pd.DataFrame({
        'elevation': np.repeat([200, 700, 1300, 1900], n_per_type)
                     + rng.normal(0, 120, len(sites)),
        'soil_ph': np.repeat([4.5, 5.5, 6.5, 7.5], n_per_type)
                   + rng.normal(0, 0.3, len(sites)),
        'aspect': rng.uniform(0, 360, len(sites)),
    }, index=sites)


def main():
    """Main demo function."""
    print("=" * 62)
    print("VegZ Demonstration Script")
    print("Comprehensive Vegetation Data Analysis")
    print("=" * 62)

    try:
        import VegZ as vegz_package
        from VegZ import (DiversityAnalyzer, EcologicalStatistics,
                          MultivariateAnalyzer, VegZ)
        print(f"[OK] VegZ {vegz_package.__version__} imported successfully")
    except ImportError as e:
        print(f"[FAIL] Could not import VegZ: {e}")
        return

    print("\n1. Creating synthetic vegetation data...")
    vegetation_data = build_synthetic_data()
    print(f"   Created data: {vegetation_data.shape[0]} sites x "
          f"{vegetation_data.shape[1]} species")
    print("   True structure: 4 vegetation types, 15 sites each")

    print("\n2. Initializing VegZ...")
    veg = VegZ()
    veg.species_matrix = vegetation_data
    print("   [OK] VegZ initialized with synthetic data")

    print("\n3. Calculating diversity indices...")
    diversity = veg.calculate_diversity(['shannon', 'simpson', 'richness', 'evenness'])
    print(f"   Calculated diversity for {len(diversity)} sites")
    print(f"   Mean Shannon diversity : {diversity['shannon'].mean():.2f}")
    print(f"   Mean species richness  : {diversity['richness'].mean():.1f}")
    print(f"   Mean Pielou evenness   : {diversity['evenness'].mean():.2f}")

    print("\n4. Performing comprehensive elbow analysis...")
    elbow_results = veg.elbow_analysis(
        k_range=range(1, 12),
        methods=['knee_locator', 'derivative', 'variance_explained', 'distortion_jump'],
        plot_results=False,
    )

    print("   Elbow points detected by each method:")
    for method, k_value in elbow_results['elbow_points'].items():
        print(f"     {method:22s}: k = {k_value}")

    optimal_k = elbow_results['recommendations']['consensus']
    confidence = elbow_results['recommendations']['confidence']
    print(f"   Consensus recommendation : k = {optimal_k}")
    print(f"   Confidence score         : {confidence:.2f}")

    print(f"\n5. Performing k-means clustering with k = {optimal_k}...")
    clusters = veg.kmeans_clustering(n_clusters=optimal_k)
    print(f"   Clustering inertia: {clusters['inertia']:.1f}")
    print("   Cluster sizes:")
    for cluster_id, count in clusters['cluster_labels'].value_counts().sort_index().items():
        print(f"     Cluster {cluster_id}: {count} sites")

    print(f"\n6. Finding indicator species for {optimal_k} clusters...")
    indicators = veg.indicator_species_analysis(
        clusters['cluster_labels'], permutations=199, random_state=42)

    print("   Top indicator species per cluster (Dufrene-Legendre IndVal):")
    for cluster_id in sorted(indicators['cluster'].unique()):
        best = indicators[(indicators['cluster'] == cluster_id)
                          & indicators['is_max_group']].nlargest(2, 'indicator_value')
        print(f"     Cluster {cluster_id}:")
        for _, row in best.iterrows():
            print(f"       {row['species']:12s} IndVal = {row['indicator_value']:5.1f}"
                  f"  p = {row['p_value']:.3f}")

    print("\n7. Performing ordination analysis...")
    pca_results = veg.pca_analysis(transform='hellinger', n_components=4)
    explained_var = pca_results['explained_variance_ratio']
    print("   PCA explained variance:")
    for i, var in enumerate(explained_var):
        print(f"     PC{i + 1}: {var:.1%}")
    print(f"     Cumulative: {sum(explained_var):.1%}")

    # Genuine non-metric MDS.
    nmds_results = veg.nmds_analysis(distance_metric='bray_curtis',
                                     n_dimensions=2, random_state=42)
    print(f"\n   NMDS stress value: {nmds_results['stress']:.3f}")
    if nmds_results['stress'] < 0.2:
        print("   [OK] Good NMDS representation (stress < 0.2)")
    else:
        print("   [WARN] High NMDS stress - consider more dimensions")

    print("\n8. Testing whether clusters differ compositionally...")
    stats_engine = EcologicalStatistics()
    distances = stats_engine.calculate_distance_matrix(vegetation_data, 'bray_curtis')
    permanova = stats_engine.permanova(distances, clusters['cluster_labels'],
                                       permutations=999, random_state=42)
    print(f"   PERMANOVA: F = {permanova['f_statistic']:.2f}, "
          f"R2 = {permanova['r_squared']:.3f}, p = {permanova['p_value']:.3f}")

    # PERMANOVA assumes the groups have comparable multivariate dispersion. If
    # they do not, a significant result may reflect differing spread rather
    # than differing composition, so check before interpreting the above.
    dispersion = stats_engine.permdisp(distances, clusters['cluster_labels'],
                                       permutations=999, random_state=42)
    print(f"   PERMDISP : F = {dispersion['f_statistic']:.2f}, "
          f"p = {dispersion['p_value']:.3f}")
    if dispersion['p_value'] >= 0.05:
        print("   [OK] Dispersions are homogeneous, so the PERMANOVA result")
        print("        can be read as a genuine difference in composition")
    else:
        print("   [WARN] Dispersions differ between groups - the PERMANOVA")
        print("          result may reflect spread rather than location")

    print("\n9. Partitioning beta diversity into turnover and nestedness...")
    # Two site pairs can share a total beta diversity for opposite reasons:
    # species replacing one another, or one site being a depleted subset.
    diversity_engine = DiversityAnalyzer()
    partition = diversity_engine.beta_partition_multisite(vegetation_data,
                                                          family='sorensen')
    print(f"   Turnover   (beta_SIM): {partition['turnover']:.3f}")
    print(f"   Nestedness (beta_SNE): {partition['nestedness']:.3f}")
    print(f"   Total      (beta_SOR): {partition['total']:.3f}")
    if partition['turnover'] > partition['nestedness']:
        print("   Turnover dominates: the types hold different species,")
        print("   which is what four distinct vegetation types should give")
    else:
        print("   Nestedness dominates: the poorer sites are subsets of the")
        print("   richer ones rather than holding different species")

    print("\n10. Relating composition to the environment...")
    environment = build_synthetic_environment(vegetation_data.index)
    multivar = MultivariateAnalyzer()

    # Is the constrained ordination significant at all? Constrained inertia is
    # always positive, so the ordination alone does not tell you.
    overall = multivar.anova_rda(vegetation_data, environment,
                                 permutations=999, random_state=42)
    model_row = overall['table'].loc['Model']
    print(f"   RDA overall: F = {model_row['F']:.2f}, "
          f"p = {model_row['Pr(>F)']:.3f}, "
          f"{overall['proportion_constrained']:.1%} of variance constrained")

    # Which variables carry the signal, each tested against all the others?
    by_margin = multivar.anova_rda(vegetation_data, environment,
                                   by='margin', permutations=999,
                                   random_state=42)
    print("   Marginal tests (elevation and pH track the vegetation types,")
    print("   aspect is noise):")
    for term, row in by_margin['table'].iterrows():
        if term in ('Residual', 'Total') or pd.isna(row['Pr(>F)']):
            continue
        verdict = 'significant' if row['Pr(>F)'] < 0.05 else 'not significant'
        print(f"     {term:12s} F = {row['F']:5.2f}  p = {row['Pr(>F)']:.3f}"
              f"  ({verdict})")

    print("\n11. Partitioning variance between topography and soil...")
    topography = environment[['elevation', 'aspect']]
    soil = environment[['soil_ph']]
    partitioned = multivar.varpart(vegetation_data, topography, soil,
                                   table_names=['Topography', 'Soil'],
                                   permutations=999, random_state=42)
    fractions = partitioned['fractions']
    for fraction, row in fractions.iterrows():
        print(f"     {fraction:24s}: adj R2 = {row['adj_R2']:6.3f}")
    print(f"   Fractions sum to {fractions['adj_R2'].sum():.3f}")

    print("\n12. Dataset summary")
    summary = veg.summary_statistics()
    print(f"   Total sites          : {summary['n_sites']}")
    print(f"   Total species        : {summary['n_species']}")
    print(f"   Mean species per site: {summary['mean_species_per_site']:.1f}")
    print(f"   Total abundance      : {summary['total_abundance']:.0f}")

    print("\n" + "=" * 62)
    print("[DONE] VegZ demonstration completed successfully")
    print("Documentation: https://mhatim99.github.io/VegZ/")
    print("Report issues: https://github.com/mhatim99/VegZ/issues")
    print("=" * 62)


if __name__ == "__main__":
    main()
