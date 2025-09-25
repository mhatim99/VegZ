"""
VegZ: A comprehensive Python package for vegetation data analysis and environmental modelling.

Copyright (c) 2025 Mohamed Z. Hatim

This package provides tools for:
- Data management and preprocessing
- Data quality and validation  
- Diversity and community analysis
- Multivariate analysis and ordination
- Clustering analyses
- Statistical analysis
- Temporal and spatial analysis
- Machine learning and predictive analysis
- Functional trait analysis
- Visualization and reporting
"""

__version__ = "1.2.0"
__author__ = "Mohamed Z. Hatim"
__email__ = "mhatim4040@gmail.com"
__copyright__ = "Copyright (c) 2025 Mohamed Z. Hatim"

# Copyright (c) 2025 Mohamed Z. Hatim
from .core import VegZ, quick_diversity_analysis, quick_ordination, quick_clustering, quick_elbow_analysis
from .diversity import DiversityAnalyzer
from .multivariate import MultivariateAnalyzer
from .clustering import VegetationClustering
from .statistics import EcologicalStatistics

# Copyright (c) 2025 Mohamed Z. Hatim
from .temporal import TemporalAnalyzer
from .spatial import SpatialAnalyzer
from .environmental import EnvironmentalModeler
from .machine_learning import MachineLearningAnalyzer, PredictiveModeling
from .functional_traits import FunctionalTraitAnalyzer, TraitSyndromes
from .nestedness import NestednessAnalyzer, NullModels, NestednessSignificance
from .specialized_methods import PhylogeneticDiversityAnalyzer, MetacommunityAnalyzer, NetworkAnalyzer
from .interactive_viz import InteractiveVisualizer, ReportGenerator

# Copyright (c) 2025 Mohamed Z. Hatim
from . import data_management
from . import data_quality
from . import diversity
from . import multivariate
from . import clustering
from . import statistics
from . import temporal
from . import spatial
from . import environmental
from . import machine_learning
from . import functional_traits
from . import nestedness
from . import specialized_methods
from . import interactive_viz
from . import visualization

__all__ = [
    # Copyright (c) 2025 Mohamed Z. Hatim
    'VegZ',
    'DiversityAnalyzer', 
    'MultivariateAnalyzer',
    'VegetationClustering',
    'EcologicalStatistics',
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    'TemporalAnalyzer',
    'SpatialAnalyzer', 
    'EnvironmentalModeler',
    'MachineLearningAnalyzer',
    'PredictiveModeling',
    'FunctionalTraitAnalyzer',
    'TraitSyndromes',
    'NestednessAnalyzer',
    'NullModels',
    'NestednessSignificance',
    'PhylogeneticDiversityAnalyzer',
    'MetacommunityAnalyzer',
    'NetworkAnalyzer',
    'InteractiveVisualizer',
    'ReportGenerator',
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    'quick_diversity_analysis',
    'quick_ordination', 
    'quick_clustering',
    'quick_elbow_analysis',
    
    # Copyright (c) 2025 Mohamed Z. Hatim
    'data_management',
    'data_quality',
    'diversity',
    'multivariate', 
    'clustering',
    'statistics',
    'temporal',
    'spatial',
    'environmental', 
    'machine_learning',
    'functional_traits',
    'nestedness',
    'specialized_methods',
    'interactive_viz',
    'visualization'
]

# Copyright (c) 2025 Mohamed Z. Hatim
DESCRIPTION = "A comprehensive Python package for vegetation data analysis and environmental modeling"
LONG_DESCRIPTION = """
VegZ is a professional-grade Python package designed specifically for vegetation data analysis 
and environmental modeling. It provides a comprehensive suite of tools for ecologists, 
environmental scientists, and researchers working with biodiversity and vegetation data.

Key Features:
=============

📊 Data Management & Preprocessing
- Parse vegetation survey data from CSV, Excel, Turboveg exports
- Integration with remote sensing APIs (Landsat, MODIS, Sentinel)
- Darwin Core standards for biodiversity data
- Species name standardization and fuzzy matching
- Coordinate system transformations
- Data transformation methods (Hellinger, chord, Wisconsin, etc.)

🔍 Data Quality & Validation  
- Comprehensive spatial coordinate validation
- Temporal data validation and date parsing
- Geographic outlier detection
- Country boundary consistency checks
- Coordinate precision assessment
- Duplicate record identification

📈 Diversity & Community Analysis
- 15+ diversity indices (Shannon, Simpson, Hill numbers, etc.)
- Species richness estimators (Chao1, ACE, Jackknife)
- Beta diversity analysis (turnover vs nestedness)
- Rarefaction and extrapolation curves
- Species accumulation curves

🎯 Multivariate Analysis & Ordination
- Complete ordination suite: PCA, CA, DCA, CCA, RDA, NMDS, PCoA
- Multiple ecological distance matrices
- Procrustes analysis for ordination comparison
- Environmental vector fitting
- Goodness-of-fit diagnostics

🌳 Advanced Clustering Methods
- TWINSPAN (Two-Way Indicator Species Analysis) 
- Hierarchical clustering with ecological distances
- Fuzzy C-means clustering
- DBSCAN for core community detection
- Gaussian Mixture Models
- Clustering validation (silhouette, gap statistic)

📊 Statistical Analysis
- PERMANOVA (Permutational MANOVA)
- ANOSIM (Analysis of Similarities) 
- MRPP (Multi-Response Permutation Procedures)
- Mantel tests and partial Mantel tests
- Indicator Species Analysis (IndVal)
- SIMPER (Similarity Percentages)

🎨 Visualization & Reporting
- Specialized ecological plots
- Ordination diagrams with environmental vectors
- Diversity profiles and accumulation curves
- Interactive dashboards
- Automated quality reports
- Export functions (HTML, PDF)

Example Usage:
==============

```python
import pandas as pd
from VegZ import VegZ

# Copyright (c) 2025 Mohamed Z. Hatim
veg = VegZ()

# Copyright (c) 2025 Mohamed Z. Hatim
data = veg.load_data('vegetation_data.csv')

# Copyright (c) 2025 Mohamed Z. Hatim
diversity = veg.calculate_diversity(['shannon', 'simpson', 'richness'])

# Copyright (c) 2025 Mohamed Z. Hatim
pca_results = veg.pca_analysis(transform='hellinger')
nmds_results = veg.nmds_analysis(distance_metric='bray_curtis')

# Copyright (c) 2025 Mohamed Z. Hatim
twinspan_results = veg.clustering.twinspan(data)
hierarchical_results = veg.hierarchical_clustering(n_clusters=5)

# Copyright (c) 2025 Mohamed Z. Hatim
permanova_results = veg.statistics.permanova(distance_matrix, groups)
mantel_results = veg.statistics.mantel_test(matrix1, matrix2)

# Copyright (c) 2025 Mohamed Z. Hatim
diversity_plot = veg.plot_diversity(diversity, 'shannon')
ordination_plot = veg.plot_ordination(pca_results)
```

Requirements:
=============
- Python >= 3.8
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0
- scikit-learn >= 1.0.0

Optional dependencies for extended functionality:
- GeoPandas (spatial analysis)
- PyProj (coordinate transformations)  
- Earth Engine API (remote sensing)
- Fuzzy matching libraries

Installation:
=============
```bash
pip install VegZ
```

For development version:
```bash
pip install git+https://github.com/mhatim99/VegZ.git
```

Documentation and Support:
==========================
- Documentation: https://mhatim99.github.io/VegZ/
- GitHub: https://github.com/mhatim99/VegZ
- Issues: https://github.com/mhatim99/VegZ/issues

License: MIT
Copyright (c) 2025 Mohamed Z. Hatim
"""

# Copyright (c) 2025 Mohamed Z. Hatim
VERSION_INFO = {
    'major': 1,
    'minor': 2,
    'patch': 0,
    'release': 'stable',
    'version': __version__
}

def get_version_info():
    """Return version information as a dictionary."""
    return VERSION_INFO.copy()

def show_versions():
    """Display version information for VegZ and dependencies."""
    print(f"VegZ version: {__version__}")
    print(f"Copyright: {__copyright__}")
    print()
    
# Copyright (c) 2025 Mohamed Z. Hatim
    dependencies = [
        'numpy', 'pandas', 'scipy', 'matplotlib', 'sklearn', 
        'geopandas', 'pyproj', 'fuzzywuzzy'
    ]
    
    print("Dependencies:")
    print("-" * 40)
    
    for dep in dependencies:
        try:
            if dep == 'sklearn':
                import sklearn
                version = sklearn.__version__
            else:
                module = __import__(dep)
                version = getattr(module, '__version__', 'unknown')
            
            print(f"{dep:15} : {version}")
        except ImportError:
            print(f"{dep:15} : not installed")

def citation():
    """Return citation information for VegZ."""
    return f"""
To cite VegZ in publications, please use:

Hatim, M.Z. (2025). VegZ: A comprehensive Python package for vegetation 
data analysis and environmental modeling. Version {__version__}.

BibTeX entry:
@software{{vegz2025,
    author = {{Hatim, Mohamed Z.}},
    title = {{VegZ: A comprehensive Python package for vegetation data analysis and environmental modeling}},
    year = {{2025}},
    version = {{{__version__}}},
    url = {{https://github.com/mhatim99/VegZ}}
}}
"""

# Copyright (c) 2025 Mohamed Z. Hatim
import warnings

# Copyright (c) 2025 Mohamed Z. Hatim
def configure_warnings():
    """Configure warning filters for VegZ."""
    warnings.filterwarnings('default', category=UserWarning, module='vegz')
    warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='scipy')

# Copyright (c) 2025 Mohamed Z. Hatim
configure_warnings()