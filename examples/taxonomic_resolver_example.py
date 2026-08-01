"""
TaxonomicResolver Example - VegZ v1.3.0

This script demonstrates the new TaxonomicResolver feature for
validating and standardizing plant species names against online
taxonomic databases.

Copyright (c) 2025 Mohamed Z. Hatim
"""

import os

import pandas as pd
import numpy as np

# Import TaxonomicResolver from VegZ
from VegZ import (DiversityAnalyzer, TaxonomicResolver, VegZ,
                  resolve_species_names)

# ============================================================
# EXAMPLE 1: Basic Name Resolution
# ============================================================
print("=" * 60)
print("EXAMPLE 1: Basic Name Resolution")
print("=" * 60)

# List of species names to resolve (including some with issues)
species_list = [
    'Quercus robur',           # Common oak - should resolve perfectly
    'Pinus sylvestris',        # Scots pine - should resolve perfectly
    'Betula pendula',          # Silver birch - should resolve perfectly
    'Fagus sylvatica',         # European beech - should resolve perfectly
    'Acer pseudoplatanus',     # Sycamore maple - should resolve perfectly
    'Quercus robor',           # Typo - should fuzzy match to Quercus robur
    'Pinus silvestris',        # Old spelling - should match
    'Picea abies',             # Norway spruce
    'Abies alba',              # Silver fir
    'Larix decidua',           # European larch
]

# Initialize resolver with default source (World Flora Online)
print("\nUsing default source (WFO - World Flora Online)...")
resolver = TaxonomicResolver()

# Resolve names
results = resolver.resolve_names(species_list)

# Display results
print("\nResolution Results:")
print(results[['original_name', 'accepted_name', 'match_score', 'family', 'source']].to_string())

# Print summary
resolver.print_summary(results)


# ============================================================
# EXAMPLE 2: Using Different Sources
# ============================================================
print("\n" + "=" * 60)
print("EXAMPLE 2: Using Different Sources")
print("=" * 60)

# Test with GBIF (usually has high confidence scores)
print("\nUsing GBIF (Global Biodiversity Information Facility)...")
resolver_gbif = TaxonomicResolver(sources='gbif')

test_species = ['Quercus robur', 'Pinus sylvestris', 'Betula pendula']
results_gbif = resolver_gbif.resolve_names(test_species, verbose=False)

print("\nGBIF Results:")
print(results_gbif[['original_name', 'accepted_name', 'match_score', 'match_type']].to_string())

# Test with POWO (Kew Gardens)
print("\nUsing POWO (Plants of the World Online - Kew)...")
resolver_powo = TaxonomicResolver(sources='powo')
results_powo = resolver_powo.resolve_names(test_species, verbose=False)

print("\nPOWO Results:")
print(results_powo[['original_name', 'accepted_name', 'match_score', 'family']].to_string())


# ============================================================
# EXAMPLE 3: Multiple Sources with Fallback
# ============================================================
print("\n" + "=" * 60)
print("EXAMPLE 3: Multiple Sources with Fallback")
print("=" * 60)

# Use multiple sources - if first fails, try next
resolver_multi = TaxonomicResolver(
    sources=['gbif', 'wfo', 'powo'],
    use_fallback=True
)

print("\nUsing multiple sources with fallback (GBIF -> WFO -> POWO)...")
results_multi = resolver_multi.resolve_names(species_list, verbose=False)

print("\nMulti-source Results:")
print(results_multi[['original_name', 'accepted_name', 'match_score', 'source']].to_string())


# ============================================================
# EXAMPLE 4: Quick Resolution Function
# ============================================================
print("\n" + "=" * 60)
print("EXAMPLE 4: Quick Resolution Function")
print("=" * 60)

# Use convenience function for quick resolution
quick_results = resolve_species_names(
    ['Quercus robur', 'Fagus sylvatica', 'Acer campestre'],
    sources='gbif',
    verbose=False
)

print("\nQuick resolution results:")
print(quick_results[['original_name', 'accepted_name', 'match_score']].to_string())


# ============================================================
# EXAMPLE 5: Working with DataFrames
# ============================================================
print("\n" + "=" * 60)
print("EXAMPLE 5: Working with DataFrames")
print("=" * 60)

# Create a sample vegetation survey DataFrame
vegetation_data = pd.DataFrame({
    'site_id': ['SITE_001', 'SITE_001', 'SITE_002', 'SITE_002', 'SITE_003'],
    'species': ['Quercus robur', 'Pinus silvestris', 'Fagus sylvatica', 'Betula pendual', 'Acer pseudoplatanus'],
    'abundance': [25, 18, 32, 12, 8],
    'cover_percent': [45, 30, 55, 20, 15]
})

print("\nOriginal vegetation data:")
print(vegetation_data.to_string())

# Resolve and update species names in the DataFrame
resolver = TaxonomicResolver(sources='gbif')
updated_data = resolver.resolve_dataframe(
    vegetation_data,
    species_column='species',
    update_names=True,
    add_taxonomy_columns=True,
    min_score_threshold=70,
    verbose=False
)

print("\nUpdated vegetation data with resolved names:")
print(updated_data[['site_id', 'species_original', 'species', 'taxon_family', 'taxon_match_score']].to_string())


# ============================================================
# EXAMPLE 6: Handling Problematic Names
# ============================================================
print("\n" + "=" * 60)
print("EXAMPLE 6: Handling Problematic Names")
print("=" * 60)

# Test with various problematic names
problematic_names = [
    'Quercus robur',           # Correct name
    'quercus robur',           # Lowercase genus
    'Quercus robor',           # Typo
    'Quercus alba L.',         # With author citation
    'Unknown species',         # Invalid name
    'Pinus sp.',               # Genus only
    'Betula x utilis',         # Hybrid
]

print("\nTesting problematic names...")
resolver = TaxonomicResolver(sources='gbif')
results_problematic = resolver.resolve_names(problematic_names, verbose=False)

print("\nResults for problematic names:")
for _, row in results_problematic.iterrows():
    status = "Resolved" if row['match_score'] > 0 else "UNRESOLVED"
    print(f"'{row['original_name']}' -> '{row['accepted_name']}' "
          f"(score: {row['match_score']}, {status})")


# ============================================================
# EXAMPLE 7: Export Results
# ============================================================
print("\n" + "=" * 60)
print("EXAMPLE 7: Export Results")
print("=" * 60)

# Create output directory path
output_dir = os.path.dirname(os.path.abspath(__file__))

# Export to different formats
csv_path = os.path.join(output_dir, 'resolved_species.csv')
resolver.export_results(results, csv_path)

# You can also export to other formats:
# resolver.export_results(results, 'resolved_species.xlsx')  # Excel
# resolver.export_results(results, 'resolved_species.json')  # JSON
# resolver.export_results(results, 'resolved_species.tsv')   # TSV

print(f"\nResults exported to: {csv_path}")


# ============================================================
# EXAMPLE 8: Get Detailed Summary Statistics
# ============================================================
print("\n" + "=" * 60)
print("EXAMPLE 8: Detailed Summary Statistics")
print("=" * 60)

# Get summary as dictionary
summary = resolver.get_summary(results)

print("\nSummary Statistics:")
print(f"Total names processed: {summary['total_names']}")
print(f"Successfully resolved: {summary['resolved']} ({summary['resolution_rate']}%)")
print(f"Unresolved: {summary['unresolved']}")
print(f"High confidence (>=90): {summary['high_confidence_matches']}")
print(f"Medium confidence (70-89): {summary['medium_confidence_matches']}")
print(f"Low confidence (<70): {summary['low_confidence_matches']}")
print(f"Average match score: {summary['average_match_score']}")
print(f"Unique families found: {summary['families_found']}")
print(f"Sources used: {', '.join(summary['sources_used'])}")


# ============================================================
# EXAMPLE 9: Compare Sources
# ============================================================
print("\n" + "=" * 60)
print("EXAMPLE 9: Compare Different Sources")
print("=" * 60)

comparison_species = ['Quercus robur', 'Pinus sylvestris', 'Betula pendula']
sources_to_compare = ['wfo', 'gbif', 'powo']

print("\nComparing resolution across different sources:\n")
print(f"{'Species':<25} {'WFO Score':<12} {'GBIF Score':<12} {'POWO Score':<12}")
print("-" * 60)

for species in comparison_species:
    scores = []
    for source in sources_to_compare:
        try:
            r = TaxonomicResolver(sources=source)
            result = r.resolve_names([species], verbose=False)
            scores.append(result['match_score'].iloc[0])
        except Exception:
            scores.append(0)

    print(f"{species:<25} {scores[0]:<12} {scores[1]:<12} {scores[2]:<12}")


# ============================================================
# EXAMPLE 10: Integration with VegZ Analysis
# ============================================================
print("\n" + "=" * 60)
print("EXAMPLE 10: Integration with VegZ Analysis Workflow")
print("=" * 60)

# Create sample species abundance data
np.random.seed(42)
n_sites = 10
species_names = ['Quercus robur', 'Pinus sylvestris', 'Betula pendula',
                 'Fagus sylvatica', 'Acer pseudoplatanus']

# Create abundance matrix
abundance_data = pd.DataFrame(
    np.random.poisson(5, (n_sites, len(species_names))),
    columns=species_names,
    index=[f'Site_{i+1}' for i in range(n_sites)]
)

print("\nOriginal abundance data:")
print(abundance_data.head())

# Step 1: Resolve species names
print("\nStep 1: Resolving species names...")
resolver = TaxonomicResolver(sources='gbif')
name_resolution = resolver.resolve_names(species_names, verbose=False)

# Create mapping of original to accepted names
name_mapping = dict(zip(
    name_resolution['original_name'],
    name_resolution['accepted_name']
))

# Rename columns with accepted names
abundance_data_clean = abundance_data.rename(columns=name_mapping)

print("\nSpecies name mapping:")
for orig, accepted in name_mapping.items():
    print(f"{orig} -> {accepted}")

# Step 2: Calculate diversity
print("\nStep 2: Calculating diversity indices...")
diversity = DiversityAnalyzer()
diversity_results = diversity.calculate_all_indices(abundance_data_clean)

print("\nDiversity results (first 5 sites):")
print(diversity_results[['shannon', 'simpson', 'richness']].head())

# Step 3: Perform ordination
print("\nStep 3: Performing PCA ordination...")
veg = VegZ()
veg.data = abundance_data_clean
veg.species_matrix = abundance_data_clean

pca_results = veg.pca_analysis(transform='hellinger')
print(f"PCA explained variance (first 2 axes): {pca_results['explained_variance_ratio'][:2]}")

print("\n" + "=" * 60)
print("All examples completed successfully!")
print("=" * 60)
