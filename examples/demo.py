#!/usr/bin/env python3
"""
VegZ Demo Script

This script demonstrates the main functionality of VegZ including
the comprehensive elbow analysis for optimal cluster determination.

Run with: VegZ-demo
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    """Main demo function."""
    print("=" * 60)
    print("VegZ Demonstration Script")
    print("Comprehensive Vegetation Data Analysis")
    print("=" * 60)
    
    try:
        from VegZ import VegZ, quick_elbow_analysis
        print("✓ VegZ imported successfully")
    except ImportError as e:
        print(f"✗ Could not import VegZ: {e}")
        return
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    print("\n1. Creating synthetic vegetation data...")
    np.random.seed(42)
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    n_sites_per_type = 15
    n_species = 20
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    deciduous = np.random.poisson(5, (n_sites_per_type, n_species)) * \
                np.random.binomial(1, 0.4, (n_sites_per_type, n_species))
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    coniferous = np.random.poisson(8, (n_sites_per_type, n_species)) * \
                 np.random.binomial(1, 0.2, (n_sites_per_type, n_species))
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    mixed = np.random.poisson(6, (n_sites_per_type, n_species)) * \
            np.random.binomial(1, 0.3, (n_sites_per_type, n_species))
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    grassland = np.random.poisson(3, (n_sites_per_type, n_species)) * \
                np.random.binomial(1, 0.5, (n_sites_per_type, n_species))
    
    data = np.vstack([deciduous, coniferous, mixed, grassland])
    
    species_names = [f'Species_{i+1:02d}' for i in range(n_species)]
    site_names = [f'SITE_{i+1:03d}' for i in range(60)]
    
    vegetation_data = pd.DataFrame(data, index=site_names, columns=species_names)
    print(f"   Created data: {vegetation_data.shape[0]} sites × {vegetation_data.shape[1]} species")
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    print("\n2. Initializing VegZ...")
    veg = VegZ()
    veg.species_matrix = vegetation_data
    print("✓ VegZ initialized with synthetic data")
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    print("\n3. Calculating diversity indices...")
    diversity = veg.calculate_diversity(['shannon', 'simpson', 'richness', 'evenness'])
    print(f"   Calculated diversity for {len(diversity)} sites")
    print(f"   Mean Shannon diversity: {diversity['shannon'].mean():.2f}")
    print(f"   Mean species richness: {diversity['richness'].mean():.1f}")
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    print("\n4. Performing comprehensive elbow analysis...")
    elbow_results = veg.elbow_analysis(
        k_range=range(1, 12),
        methods=['knee_locator', 'derivative', 'variance_explained', 'distortion_jump'],
        plot_results=False  # Copyright (c) 2025 Mohamed Z. Hatim
    )
    
    print("   Elbow points detected by each method:")
    for method, k_value in elbow_results['elbow_points'].items():
        print(f"   • {method:20s}: k = {k_value}")
    
    optimal_k = elbow_results['recommendations']['consensus']
    confidence = elbow_results['recommendations']['confidence']
    print(f"\n   📊 Consensus recommendation: k = {optimal_k}")
    print(f"   📊 Confidence score: {confidence:.2f}")
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    print(f"\n5. Performing clustering with k = {optimal_k}...")
    clusters = veg.kmeans_clustering(n_clusters=optimal_k)
    print(f"   Clustering inertia: {clusters['inertia']:.1f}")
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    cluster_counts = clusters['cluster_labels'].value_counts().sort_index()
    print("   Cluster sizes:")
    for cluster_id, count in cluster_counts.items():
        print(f"   • Cluster {cluster_id}: {count} sites")
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    print(f"\n6. Finding indicator species for {optimal_k} clusters...")
    indicators = veg.indicator_species_analysis(clusters['cluster_labels'])
    
    print("   Top indicator species per cluster:")
    for cluster_id in sorted(indicators['cluster'].unique()):
        cluster_indicators = indicators[indicators['cluster'] == cluster_id].nlargest(2, 'indicator_value')
        print(f"   • Cluster {cluster_id}:")
        for _, row in cluster_indicators.iterrows():
            print(f"     - {row['species']}: {row['indicator_value']:.1f}")
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    print("\n7. Performing ordination analysis...")
    pca_results = veg.pca_analysis(transform='hellinger', n_components=4)
    explained_var = pca_results['explained_variance_ratio']
    print(f"   PCA explained variance:")
    for i, var in enumerate(explained_var):
        print(f"   • PC{i+1}: {var:.1%}")
    print(f"   • Cumulative: {sum(explained_var):.1%}")
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    nmds_results = veg.nmds_analysis(distance_metric='bray_curtis', n_dimensions=2)
    print(f"\n   NMDS stress value: {nmds_results['stress']:.3f}")
    if nmds_results['stress'] < 0.2:
        print("   ✓ Good NMDS representation (stress < 0.2)")
    else:
        print("   ⚠ High NMDS stress - consider more dimensions")
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    print("\n8. Demonstrating quick functions...")
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    quick_optimal_k = veg.quick_elbow_analysis(max_k=8)
    print(f"   Quick elbow analysis: k = {quick_optimal_k}")
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    stats = veg.summary_statistics()
    print(f"\n📈 Dataset Summary:")
    print(f"   • Total sites: {stats['n_sites']}")
    print(f"   • Total species: {stats['n_species']}")
    print(f"   • Mean species per site: {stats['mean_species_per_site']:.1f}")
    print(f"   • Total abundance: {stats['total_abundance']}")
    
    print("\n" + "=" * 60)
    print("✅ VegZ demonstration completed successfully!")
    print("📚 For more examples, see: https://vegz.readthedocs.io/")
    print("🐛 Report issues: https://github.com/mhatim99/VegZ/issues")
    print("=" * 60)


if __name__ == "__main__":
    main()