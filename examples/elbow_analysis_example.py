"""
Comprehensive Elbow Analysis Example for VegZ

This example demonstrates how to use the new elbow analysis functionality
to determine the optimal number of clusters for vegetation data.

Author: Mohamed Z. Hatim
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from VegZ import VegZ, quick_elbow_analysis, VegetationClustering

def demonstrate_elbow_analysis():
    """Demonstrate comprehensive elbow analysis functionality."""
    
    print("=== VegZ Comprehensive Elbow Analysis Demonstration ===\n")
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    try:
        data = pd.read_csv('species_abundance.csv', index_col=0)
        print(f"Loaded vegetation data: {data.shape[0]} sites, {data.shape[1]} species")
    except FileNotFoundError:
        print("Creating synthetic vegetation data for demonstration...")
        # Copyright (c) 2025 Mohamed Z. Hatim
        np.random.seed(42)
        n_sites = 50
        n_species = 20
        
        # Copyright (c) 2025 Mohamed Z. Hatim
        group1 = np.random.poisson(5, (12, n_species)) * np.random.binomial(1, 0.3, (12, n_species))
        group2 = np.random.poisson(3, (13, n_species)) * np.random.binomial(1, 0.4, (13, n_species))
        group3 = np.random.poisson(8, (12, n_species)) * np.random.binomial(1, 0.2, (12, n_species))
        group4 = np.random.poisson(2, (13, n_species)) * np.random.binomial(1, 0.5, (13, n_species))
        
        data = np.vstack([group1, group2, group3, group4])
        
        species_names = [f'Species_{i+1:02d}' for i in range(n_species)]
        site_names = [f'SITE_{i+1:03d}' for i in range(n_sites)]
        
        data = pd.DataFrame(data, index=site_names, columns=species_names)
        print(f"Created synthetic data: {data.shape[0]} sites, {data.shape[1]} species")
    
    print("\n" + "="*60)
    print("METHOD 1: Using VegZ Main Class")
    print("="*60)
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    veg = VegZ()
    veg.species_matrix = data
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    print("\n1. Running comprehensive elbow analysis...")
    elbow_results = veg.elbow_analysis(
        k_range=range(1, 12),
        methods=['knee_locator', 'derivative', 'variance_explained', 'distortion_jump'],
        transform='hellinger',
        plot_results=True
    )
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    print("\nElbow Analysis Results:")
    print("-" * 40)
    for method, k_value in elbow_results['elbow_points'].items():
        print(f"{method:20s}: k = {k_value}")
    
    print(f"\nConsensus recommendation: k = {elbow_results['recommendations']['consensus']}")
    print(f"Confidence score: {elbow_results['recommendations']['confidence']:.2f}")
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    print("\nAdditional recommendations:")
    print(f"Best Silhouette score: k = {elbow_results['recommendations'].get('silhouette_optimal', 'N/A')}")
    print(f"Best Calinski-Harabasz: k = {elbow_results['recommendations'].get('calinski_optimal', 'N/A')}")
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    print("\n2. Quick elbow analysis for rapid results...")
    optimal_k = veg.quick_elbow_analysis(max_k=10)
    print(f"Quick recommendation: k = {optimal_k}")
    
    print("\n" + "="*60)
    print("METHOD 2: Using VegetationClustering Class Directly")
    print("="*60)
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    clustering = VegetationClustering()
    
    print("\n3. Using specialized clustering class...")
    detailed_results = clustering.comprehensive_elbow_analysis(
        data=data,
        k_range=range(1, 10),
        methods=['knee_locator', 'derivative', 'l_method'],
        transform='hellinger',
        plot_results=False  # Copyright (c) 2025 Mohamed Z. Hatim
    )
    
    print("\nDetailed method information:")
    for method, details in detailed_results['method_details'].items():
        print(f"\n{method}:")
        print(f"  Description: {details['description']}")
        if 'reference' in details:
            print(f"  Reference: {details['reference']}")
    
    print("\n" + "="*60)
    print("METHOD 3: Quick Convenience Function")
    print("="*60)
    
    print("\n4. Using quick convenience function...")
    quick_results = quick_elbow_analysis(
        data=data,
        max_k=8,
        plot_results=False
    )
    
    print(f"Quick analysis recommendation: k = {quick_results['recommendations']['consensus']}")
    
    print("\n" + "="*60)
    print("METHOD 4: Practical Workflow Integration")
    print("="*60)
    
    print("\n5. Complete clustering workflow with elbow analysis...")
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    optimal_k = veg.quick_elbow_analysis(max_k=10)
    print(f"Optimal k determined: {optimal_k}")
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    print(f"\nPerforming k-means clustering with k={optimal_k}...")
    kmeans_results = veg.kmeans_clustering(n_clusters=optimal_k)
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    print(f"Clustering inertia: {kmeans_results['inertia']:.2f}")
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    print("\nFinding indicator species for each cluster...")
    indicators = veg.indicator_species_analysis(kmeans_results['cluster_labels'])
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    for cluster in sorted(indicators['cluster'].unique()):
        cluster_indicators = indicators[indicators['cluster'] == cluster].nlargest(3, 'indicator_value')
        print(f"\nCluster {cluster} top indicators:")
        for _, row in cluster_indicators.iterrows():
            print(f"  {row['species']}: {row['indicator_value']:.1f}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\nElbow Analysis Methods Available:")
    print("• knee_locator: Kneedle algorithm (Satopaa et al., 2011)")
    print("• derivative: Second derivative maximum")
    print("• variance_explained: <10% additional variance threshold")
    print("• distortion_jump: Jump method (Sugar & James, 2003)")
    print("• l_method: L-method (Salvador & Chan, 2004)")
    
    print("\nUsage Recommendations:")
    print("• For quick analysis: use quick_elbow_analysis() or veg.quick_elbow_analysis()")
    print("• For detailed analysis: use veg.elbow_analysis() with multiple methods")
    print("• For research: use VegetationClustering.comprehensive_elbow_analysis()")
    print("• Always validate results with ecological knowledge")
    
    print("\nOutput includes:")
    print("• Optimal k recommendations from each method")
    print("• Consensus recommendation with confidence score")
    print("• Comprehensive metrics (inertia, silhouette, Calinski-Harabasz)")
    print("• Visualization plots (optional)")
    print("• Method details and references")

if __name__ == "__main__":
    demonstrate_elbow_analysis()