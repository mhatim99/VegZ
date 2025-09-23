"""
Pytest configuration file for VegZ tests.
"""

import pytest
import numpy as np
import pandas as pd
import warnings

# Copyright (c) 2025 Mohamed Z. Hatim
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


@pytest.fixture(scope="session")
def large_sample_data():
    """Create a large sample dataset for performance testing."""
    np.random.seed(42)
    n_sites = 100
    n_species = 50
    
# Copyright (c) 2025 Mohamed Z. Hatim
    groups = []
    for i in range(5):
# Copyright (c) 2025 Mohamed Z. Hatim
        group_size = 20
        lambda_param = [3, 5, 2, 7, 4][i]
        prob_param = [0.3, 0.4, 0.2, 0.5, 0.35][i]
        
        group_data = (np.random.poisson(lambda_param, (group_size, n_species)) * 
                     np.random.binomial(1, prob_param, (group_size, n_species)))
        groups.append(group_data)
    
    data = np.vstack(groups)
    
    species_names = [f'Species_{i+1:03d}' for i in range(n_species)]
    site_names = [f'SITE_{i+1:04d}' for i in range(n_sites)]
    
    return pd.DataFrame(data, index=site_names, columns=species_names)


@pytest.fixture
def environmental_data():
    """Create sample environmental data."""
    np.random.seed(42)
    n_sites = 30
    
    data = {
        'elevation': np.random.uniform(500, 2000, n_sites),
        'temperature': np.random.normal(15, 5, n_sites),
        'precipitation': np.random.uniform(800, 1500, n_sites),
        'soil_ph': np.random.normal(6.5, 1.0, n_sites),
        'slope': np.random.uniform(0, 45, n_sites)
    }
    
    site_names = [f'SITE_{i+1:03d}' for i in range(n_sites)]
    
    return pd.DataFrame(data, index=site_names)


@pytest.fixture
def traits_data():
    """Create sample species trait data."""
    np.random.seed(42)
    species = [f'Species_{i+1:02d}' for i in range(20)]
    
    growth_forms = np.random.choice(['tree', 'shrub', 'herb', 'fern'], size=20)
    heights = np.random.lognormal(1, 1, 20)  # Log-normal distribution for heights
    leaf_areas = np.random.gamma(2, 50, 20)  # Gamma distribution for leaf areas
    
    data = pd.DataFrame({
        'species': species,
        'growth_form': growth_forms,
        'max_height_m': heights,
        'leaf_area_cm2': leaf_areas,
        'shade_tolerance': np.random.choice(['low', 'intermediate', 'high'], size=20),
        'drought_tolerance': np.random.choice(['low', 'intermediate', 'high'], size=20)
    })
    
    return data


@pytest.fixture
def temporal_data():
    """Create sample temporal vegetation data."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='M')
    sites = [f'SITE_{i+1:02d}' for i in range(10)]
    species = [f'Species_{i+1:02d}' for i in range(15)]
    
    data = []
    for date in dates:
        for site in sites:
# Copyright (c) 2025 Mohamed Z. Hatim
            seasonal_factor = 0.5 + 0.5 * np.cos((date.month - 6) * 2 * np.pi / 12)
            
            abundances = np.random.poisson(5 * seasonal_factor, len(species))
            
            row = {'date': date, 'site': site}
            row.update({species[i]: abundances[i] for i in range(len(species))})
            data.append(row)
    
    return pd.DataFrame(data)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "plotting: marks tests that create plots"
    )