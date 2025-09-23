"""
Comprehensive diversity analysis module for vegetation data.

Copyright (c) 2025 Mohamed Z. Hatim
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional, Dict, Any
from scipy import stats
from scipy.special import comb
import warnings


class DiversityAnalyzer:
    """Comprehensive diversity analysis for ecological communities."""
    
    def __init__(self):
        """Initialize diversity analyzer."""
        self.available_indices = [
            'shannon', 'simpson', 'simpson_inv', 'richness', 'evenness',
            'fisher_alpha', 'berger_parker', 'mcintosh', 'brillouin',
            'menhinick', 'margalef', 'chao1', 'ace', 'jack1', 'jack2'
        ]
    
    def calculate_all_indices(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all available diversity indices.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Species abundance matrix (samples x species)
            
        Returns:
        --------
        pd.DataFrame
            Diversity indices for each sample
        """
        results = pd.DataFrame(index=data.index)
        
        for index in self.available_indices:
            try:
                results[index] = self.calculate_index(data, index)
            except Exception as e:
                warnings.warn(f"Could not calculate {index}: {e}")
                results[index] = np.nan
        
        return results
    
    def calculate_index(self, data: pd.DataFrame, index: str) -> pd.Series:
        """
        Calculate specific diversity index.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Species abundance matrix
        index : str
            Name of diversity index
            
        Returns:
        --------
        pd.Series
            Diversity values for each sample
        """
        index_lower = index.lower()
        
        if index_lower == 'shannon':
            return self.shannon_diversity(data)
        elif index_lower == 'simpson':
            return self.simpson_diversity(data)
        elif index_lower == 'simpson_inv':
            return self.simpson_inverse(data)
        elif index_lower == 'richness':
            return self.species_richness(data)
        elif index_lower == 'evenness':
            return self.pielou_evenness(data)
        elif index_lower == 'fisher_alpha':
            return self.fisher_alpha(data)
        elif index_lower == 'berger_parker':
            return self.berger_parker(data)
        elif index_lower == 'mcintosh':
            return self.mcintosh_diversity(data)
        elif index_lower == 'brillouin':
            return self.brillouin_diversity(data)
        elif index_lower == 'menhinick':
            return self.menhinick_index(data)
        elif index_lower == 'margalef':
            return self.margalef_index(data)
        elif index_lower == 'chao1':
            return self.chao1_estimator(data)
        elif index_lower == 'ace':
            return self.ace_estimator(data)
        elif index_lower == 'jack1':
            return self.jackknife1_estimator(data)
        elif index_lower == 'jack2':
            return self.jackknife2_estimator(data)
        else:
            raise ValueError(f"Unknown diversity index: {index}")
    
    def shannon_diversity(self, data: pd.DataFrame) -> pd.Series:
        """Shannon diversity index (H')."""
        def calculate_shannon(row):
            abundances = row[row > 0]
            if len(abundances) == 0:
                return 0
            proportions = abundances / abundances.sum()
            return -np.sum(proportions * np.log(proportions))
        
        return data.apply(calculate_shannon, axis=1)
    
    def simpson_diversity(self, data: pd.DataFrame) -> pd.Series:
        """Simpson diversity index (D)."""
        def calculate_simpson(row):
            abundances = row[row > 0]
            if len(abundances) == 0:
                return 0
            proportions = abundances / abundances.sum()
            return np.sum(proportions ** 2)
        
        return data.apply(calculate_simpson, axis=1)
    
    def simpson_inverse(self, data: pd.DataFrame) -> pd.Series:
        """Inverse Simpson diversity (1/D)."""
        simpson = self.simpson_diversity(data)
        return 1 / simpson.replace(0, np.nan)
    
    def species_richness(self, data: pd.DataFrame) -> pd.Series:
        """Species richness (S)."""
        return (data > 0).sum(axis=1)
    
    def pielou_evenness(self, data: pd.DataFrame) -> pd.Series:
        """Pielou's evenness (J')."""
        shannon = self.shannon_diversity(data)
        richness = self.species_richness(data)
        return shannon / np.log(richness.replace(0, 1))
    
    def fisher_alpha(self, data: pd.DataFrame) -> pd.Series:
        """Fisher's alpha diversity."""
        def calculate_fisher(row):
            abundances = row[row > 0]
            if len(abundances) == 0:
                return 0
            
            S = len(abundances)  # Species richness
            N = abundances.sum()  # Total abundance
            
            if S <= 1 or N <= 0:
                return 0
            
# Copyright (c) 2025 Mohamed Z. Hatim
            alpha = S  # Initial guess
            for _ in range(100):  # Max iterations
                alpha_new = S / np.log(1 + N/alpha)
                if abs(alpha_new - alpha) < 1e-6:
                    break
                alpha = alpha_new
            
            return alpha
        
        return data.apply(calculate_fisher, axis=1)
    
    def berger_parker(self, data: pd.DataFrame) -> pd.Series:
        """Berger-Parker dominance index."""
        def calculate_bp(row):
            abundances = row[row > 0]
            if len(abundances) == 0:
                return 0
            return abundances.max() / abundances.sum()
        
        return data.apply(calculate_bp, axis=1)
    
    def mcintosh_diversity(self, data: pd.DataFrame) -> pd.Series:
        """McIntosh diversity index."""
        def calculate_mcintosh(row):
            abundances = row[row > 0]
            if len(abundances) == 0:
                return 0
            
            N = abundances.sum()
            U = np.sqrt(np.sum(abundances ** 2))
            
            if N == U:  # All individuals in one species
                return 0
            
            return (N - U) / (N - np.sqrt(N))
        
        return data.apply(calculate_mcintosh, axis=1)
    
    def brillouin_diversity(self, data: pd.DataFrame) -> pd.Series:
        """Brillouin diversity index."""
        def calculate_brillouin(row):
            abundances = row[row > 0].astype(int)
            if len(abundances) == 0:
                return 0
            
            N = abundances.sum()
            
# Copyright (c) 2025 Mohamed Z. Hatim
            log_factorial_sum = np.sum([self._log_factorial(n) for n in abundances])
            log_N_factorial = self._log_factorial(N)
            
            return (log_N_factorial - log_factorial_sum) / N
        
        return data.apply(calculate_brillouin, axis=1)
    
    def menhinick_index(self, data: pd.DataFrame) -> pd.Series:
        """Menhinick's richness index."""
        richness = self.species_richness(data)
        total_abundance = data.sum(axis=1)
        return richness / np.sqrt(total_abundance.replace(0, 1))
    
    def margalef_index(self, data: pd.DataFrame) -> pd.Series:
        """Margalef's richness index."""
        richness = self.species_richness(data)
        total_abundance = data.sum(axis=1)
        return (richness - 1) / np.log(total_abundance.replace(0, 1))
    
    def chao1_estimator(self, data: pd.DataFrame) -> pd.Series:
        """Chao1 species richness estimator."""
        def calculate_chao1(row):
            abundances = row[row > 0]
            if len(abundances) == 0:
                return 0
            
            S_obs = len(abundances)
            f1 = np.sum(abundances == 1)  # Singletons
            f2 = np.sum(abundances == 2)  # Doubletons
            
            if f2 == 0:
                return S_obs + (f1 * (f1 - 1)) / 2
            else:
                return S_obs + (f1 ** 2) / (2 * f2)
        
        return data.apply(calculate_chao1, axis=1)
    
    def ace_estimator(self, data: pd.DataFrame, rare_threshold: int = 10) -> pd.Series:
        """ACE (Abundance-based Coverage Estimator)."""
        def calculate_ace(row):
            abundances = row[row > 0]
            if len(abundances) == 0:
                return 0
            
# Copyright (c) 2025 Mohamed Z. Hatim
            rare = abundances[abundances <= rare_threshold]
            abundant = abundances[abundances > rare_threshold]
            
            S_rare = len(rare)
            S_abund = len(abundant)
            
            if S_rare == 0:
                return S_abund
            
            N_rare = rare.sum()
            f1 = np.sum(rare == 1)
            
# Copyright (c) 2025 Mohamed Z. Hatim
            C_ace = 1 - (f1 / N_rare) if N_rare > 0 else 1
            
            if C_ace == 0:
                return S_abund + S_rare
            
# Copyright (c) 2025 Mohamed Z. Hatim
            sum_i_fi = np.sum([i * np.sum(rare == i) for i in range(1, rare_threshold + 1)])
            gamma_ace = (S_rare / C_ace) * (sum_i_fi / (N_rare * (N_rare - 1))) - 1
            gamma_ace = max(gamma_ace, 0)
            
            return S_abund + (S_rare / C_ace) + ((f1 / C_ace) * gamma_ace)
        
        return data.apply(calculate_ace, axis=1)
    
    def jackknife1_estimator(self, data: pd.DataFrame) -> pd.Series:
        """First-order Jackknife estimator."""
        richness = self.species_richness(data)
        
        def calculate_jack1(row):
            abundances = row[row > 0]
            if len(abundances) == 0:
                return 0
            
            f1 = np.sum(abundances == 1)  # Singletons
            S_obs = len(abundances)
            
            return S_obs + f1
        
        return data.apply(calculate_jack1, axis=1)
    
    def jackknife2_estimator(self, data: pd.DataFrame) -> pd.Series:
        """Second-order Jackknife estimator."""
        def calculate_jack2(row):
            abundances = row[row > 0]
            if len(abundances) == 0:
                return 0
            
            f1 = np.sum(abundances == 1)  # Singletons
            f2 = np.sum(abundances == 2)  # Doubletons
            S_obs = len(abundances)
            
            return S_obs + f1 * (2 - 1) / 2 - f2 / 2
        
        return data.apply(calculate_jack2, axis=1)
    
    def beta_diversity(self, data: pd.DataFrame, 
                      method: str = 'whittaker') -> Union[float, pd.DataFrame]:
        """
        Calculate beta diversity between samples.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Species abundance matrix
        method : str
            Beta diversity method
            
        Returns:
        --------
        float or pd.DataFrame
            Beta diversity value(s)
        """
        if method.lower() == 'whittaker':
            return self._whittaker_beta(data)
        elif method.lower() == 'sorensen':
            return self._sorensen_beta(data)
        elif method.lower() == 'jaccard':
            return self._jaccard_beta(data)
        else:
            raise ValueError(f"Unknown beta diversity method: {method}")
    
    def _whittaker_beta(self, data: pd.DataFrame) -> float:
        """Whittaker's beta diversity."""
        gamma = self.species_richness(data.sum(axis=0).to_frame().T).iloc[0]
        alpha_mean = self.species_richness(data).mean()
        return gamma / alpha_mean
    
    def _sorensen_beta(self, data: pd.DataFrame) -> pd.DataFrame:
        """Sørensen beta diversity matrix."""
        n_samples = len(data)
        beta_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i, n_samples):
                if i == j:
                    beta_matrix[i, j] = 0
                else:
                    sample1 = data.iloc[i] > 0
                    sample2 = data.iloc[j] > 0
                    
                    shared = np.sum(sample1 & sample2)
                    total = np.sum(sample1 | sample2)
                    
                    beta = 1 - (2 * shared) / (np.sum(sample1) + np.sum(sample2))
                    beta_matrix[i, j] = beta
                    beta_matrix[j, i] = beta
        
        return pd.DataFrame(beta_matrix, index=data.index, columns=data.index)
    
    def _jaccard_beta(self, data: pd.DataFrame) -> pd.DataFrame:
        """Jaccard beta diversity matrix."""
        n_samples = len(data)
        beta_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i, n_samples):
                if i == j:
                    beta_matrix[i, j] = 0
                else:
                    sample1 = data.iloc[i] > 0
                    sample2 = data.iloc[j] > 0
                    
                    shared = np.sum(sample1 & sample2)
                    total = np.sum(sample1 | sample2)
                    
                    beta = 1 - shared / total if total > 0 else 0
                    beta_matrix[i, j] = beta
                    beta_matrix[j, i] = beta
        
        return pd.DataFrame(beta_matrix, index=data.index, columns=data.index)
    
    def hill_numbers(self, data: pd.DataFrame, 
                    q_values: List[float] = [0, 1, 2]) -> pd.DataFrame:
        """
        Calculate Hill numbers (diversity of order q).
        
        Parameters:
        -----------
        data : pd.DataFrame
            Species abundance matrix
        q_values : list
            Orders of diversity to calculate
            
        Returns:
        --------
        pd.DataFrame
            Hill numbers for each sample and order
        """
        results = pd.DataFrame(index=data.index)
        
        for q in q_values:
            results[f'Hill_q{q}'] = self._calculate_hill_number(data, q)
        
        return results
    
    def _calculate_hill_number(self, data: pd.DataFrame, q: float) -> pd.Series:
        """Calculate Hill number of order q."""
        def hill_q(row):
            abundances = row[row > 0]
            if len(abundances) == 0:
                return 0
            
            proportions = abundances / abundances.sum()
            
            if q == 0:
                return len(proportions)  # Species richness
            elif q == 1:
                return np.exp(-np.sum(proportions * np.log(proportions)))  # exp(Shannon)
            else:
                return (np.sum(proportions ** q)) ** (1 / (1 - q))
        
        return data.apply(hill_q, axis=1)
    
    def _log_factorial(self, n: int) -> float:
        """Calculate log factorial."""
        if n <= 1:
            return 0
        return np.sum(np.log(np.arange(1, n + 1)))