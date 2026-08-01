"""
Aligned container for a vegetation data set.

Copyright (c) 2025 Mohamed Z. Hatim

Most VegZ functions take loose DataFrames and align them ad hoc. That works
until two tables disagree about which sites or species they contain, at which
point every function has to re-derive the intersection - and they do not all do
it the same way. :class:`VegData` does the alignment once, deterministically,
and reports exactly what it dropped.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

__all__ = ['VegData']


def _ordered_intersection(primary: Iterable, other: Iterable) -> List:
    """
    Intersection of two label collections, in ``primary`` order.

    ``set(a) & set(b)`` returns an arbitrary order that varies between runs
    under Python's string-hash randomisation, which silently makes downstream
    results irreproducible.
    """
    other_set = set(other)
    return [item for item in primary if item in other_set]


def _check_unique(labels: pd.Index, what: str) -> None:
    if labels.has_duplicates:
        duplicates = labels[labels.duplicated()].unique().tolist()
        raise ValueError(
            f"Duplicate {what} labels are not allowed: {duplicates[:5]}"
            f"{' ...' if len(duplicates) > 5 else ''}"
        )


class VegData:
    """
    A vegetation data set with guaranteed-aligned components.

    Binds a site-by-species matrix to any combination of environmental
    variables (per site), functional traits (per species) and a phylogenetic
    distance matrix (species by species), and keeps their labels consistent.

    Parameters
    ----------
    species : pd.DataFrame
        Site-by-species abundance or incidence matrix. Must be numeric and
        non-negative.
    environment : pd.DataFrame, optional
        Environmental variables indexed by site.
    traits : pd.DataFrame, optional
        Species traits indexed by species. If a ``species_column`` is present it
        is used as the index.
    phylogeny : pd.DataFrame, optional
        Square phylogenetic distance matrix indexed by species.
    metadata : dict, optional
        Free-form information carried alongside the data.
    align : bool
        Align all components to their shared sites/species on construction
        (default). Set to False to keep the tables as given.
    species_column : str
        Column of ``traits`` holding species names, if it is not already the
        index.

    Attributes
    ----------
    alignment_report : dict
        What was dropped during alignment, so a silently shrinking data set is
        visible rather than mysterious.

    Examples
    --------
    >>> data = VegData(species=abundance, environment=env, traits=traits)
    >>> data.n_sites, data.n_species
    (50, 29)
    >>> forest = data.subset(sites=data.environment.query("habitat == 'forest'").index)
    """

    def __init__(self,
                 species: pd.DataFrame,
                 environment: Optional[pd.DataFrame] = None,
                 traits: Optional[pd.DataFrame] = None,
                 phylogeny: Optional[pd.DataFrame] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 align: bool = True,
                 species_column: str = 'species'):
        self.species = self._validate_species(species)
        self.environment = None if environment is None else environment.copy()
        self.traits = self._prepare_traits(traits, species_column)
        self.phylogeny = None if phylogeny is None else phylogeny.copy()
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.alignment_report: Dict[str, Any] = {}

        if align:
            self._align()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_species(species: pd.DataFrame) -> pd.DataFrame:
        """Check the species matrix is a usable numeric community table."""
        if not isinstance(species, pd.DataFrame):
            raise TypeError("species must be a pandas DataFrame")
        if species.shape[0] == 0 or species.shape[1] == 0:
            raise ValueError("species matrix is empty")

        _check_unique(species.index, 'site')
        _check_unique(species.columns, 'species')

        non_numeric = species.columns[~species.dtypes.apply(
            pd.api.types.is_numeric_dtype)]
        if len(non_numeric) > 0:
            raise ValueError(
                "Species matrix must be entirely numeric; non-numeric columns: "
                f"{list(non_numeric)[:5]}. Move identifiers into the index or "
                "into `environment`."
            )

        values = species.to_numpy(dtype=float)
        if np.isnan(values).any():
            warnings.warn(
                "Species matrix contains missing values; they are treated as "
                "absences (0). Fill them explicitly to silence this."
            )
            species = species.fillna(0)
        if (species.to_numpy(dtype=float) < 0).any():
            raise ValueError("Species abundances must be non-negative")

        return species.astype(float)

    @staticmethod
    def _prepare_traits(traits: Optional[pd.DataFrame],
                        species_column: str) -> Optional[pd.DataFrame]:
        """Index the trait table by species name."""
        if traits is None:
            return None
        if species_column in traits.columns:
            traits = traits.set_index(species_column)
        else:
            traits = traits.copy()
        _check_unique(traits.index, 'trait species')
        return traits

    def _align(self) -> None:
        """Restrict every component to the sites/species they all share."""
        # Work out the surviving labels first, then report each table's losses
        # against that final set. Reporting as we go would understate the loss,
        # because a label kept at one step can still be removed at the next.
        sites = list(self.species.index)
        if self.environment is not None:
            _check_unique(self.environment.index, 'environment site')
            sites = _ordered_intersection(sites, self.environment.index)
            if not sites:
                raise ValueError(
                    "Species matrix and environmental data share no site labels")

        species_names = list(self.species.columns)
        for name, labels in (
            ('traits', None if self.traits is None else self.traits.index),
            ('phylogeny', None if self.phylogeny is None else self.phylogeny.index),
        ):
            if labels is None:
                continue
            _check_unique(pd.Index(labels), f'{name} species')
            species_names = _ordered_intersection(species_names, labels)
            if not species_names:
                raise ValueError(
                    f"Species matrix and {name} share no species labels")

        kept_sites, kept_species = set(sites), set(species_names)
        report: Dict[str, Any] = {
            'sites_dropped_from_species_matrix': [
                s for s in self.species.index if s not in kept_sites],
            'species_dropped_from_species_matrix': [
                s for s in self.species.columns if s not in kept_species],
        }
        if self.environment is not None:
            report['sites_dropped_from_environment'] = [
                s for s in self.environment.index if s not in kept_sites]
        if self.traits is not None:
            report['species_dropped_from_traits'] = [
                s for s in self.traits.index if s not in kept_species]
        if self.phylogeny is not None:
            report['species_dropped_from_phylogeny'] = [
                s for s in self.phylogeny.index if s not in kept_species]

        self.species = self.species.loc[sites, species_names]
        if self.environment is not None:
            self.environment = self.environment.loc[sites]
        if self.traits is not None:
            self.traits = self.traits.loc[species_names]
        if self.phylogeny is not None:
            self.phylogeny = self.phylogeny.loc[species_names, species_names]

        self.alignment_report = report

        if any(v for v in report.values()):
            summary = ', '.join(
                f"{len(v)} {k.replace('_', ' ')}"
                for k, v in report.items() if v)
            warnings.warn(
                f"Aligning components dropped {summary}. "
                "Inspect `alignment_report` for the labels."
            )

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def n_sites(self) -> int:
        """Number of sites."""
        return int(self.species.shape[0])

    @property
    def n_species(self) -> int:
        """Number of species."""
        return int(self.species.shape[1])

    @property
    def site_ids(self) -> pd.Index:
        """Site labels."""
        return self.species.index

    @property
    def species_ids(self) -> pd.Index:
        """Species labels."""
        return self.species.columns

    @property
    def has_environment(self) -> bool:
        return self.environment is not None

    @property
    def has_traits(self) -> bool:
        return self.traits is not None

    @property
    def has_phylogeny(self) -> bool:
        return self.phylogeny is not None

    def __repr__(self) -> str:
        components = [f"{self.n_sites} sites x {self.n_species} species"]
        if self.environment is not None:
            components.append(f"{self.environment.shape[1]} env vars")
        if self.traits is not None:
            components.append(f"{self.traits.shape[1]} traits")
        if self.has_phylogeny:
            components.append("phylogeny")
        return f"VegData({', '.join(components)})"

    def __len__(self) -> int:
        return self.n_sites

    # ------------------------------------------------------------------
    # Subsetting and cleaning
    # ------------------------------------------------------------------

    def subset(self, sites: Optional[Sequence] = None,
               species: Optional[Sequence] = None) -> 'VegData':
        """
        Return a new :class:`VegData` restricted to the given sites/species.

        Every component is subset consistently, so the result is still aligned.

        Parameters
        ----------
        sites : sequence, optional
            Site labels, or a boolean mask the length of ``n_sites``.
        species : sequence, optional
            Species labels, or a boolean mask the length of ``n_species``.
        """
        selected_sites = self._resolve(sites, self.site_ids, 'sites')
        selected_species = self._resolve(species, self.species_ids, 'species')

        return VegData(
            species=self.species.loc[selected_sites, selected_species],
            environment=(None if self.environment is None
                         else self.environment.loc[selected_sites]),
            traits=(None if self.traits is None
                    else self.traits.loc[selected_species]),
            phylogeny=(None if self.phylogeny is None
                       else self.phylogeny.loc[selected_species, selected_species]),
            metadata=dict(self.metadata),
            align=False,
        )

    @staticmethod
    def _resolve(selection: Optional[Sequence], available: pd.Index,
                 what: str) -> List:
        """Turn labels or a boolean mask into a list of labels."""
        if selection is None:
            return list(available)

        selection = list(selection)
        if selection and all(isinstance(item, (bool, np.bool_)) for item in selection):
            if len(selection) != len(available):
                raise ValueError(
                    f"Boolean mask for {what} has length {len(selection)}, "
                    f"expected {len(available)}")
            return [label for label, keep in zip(available, selection) if keep]

        missing = [item for item in selection if item not in available]
        if missing:
            raise KeyError(f"Unknown {what}: {missing[:5]}")
        return selection

    def drop_empty(self, min_species: int = 1,
                   min_occurrences: int = 1) -> 'VegData':
        """
        Drop sites with too few species and species that are too rare.

        Parameters
        ----------
        min_species : int
            Keep sites containing at least this many species.
        min_occurrences : int
            Keep species occurring at at least this many sites.
        """
        keep_sites = (self.species > 0).sum(axis=1) >= min_species
        keep_species = (self.species > 0).sum(axis=0) >= min_occurrences

        return self.subset(sites=list(self.site_ids[keep_sites]),
                           species=list(self.species_ids[keep_species]))

    def transform(self, method: str = 'hellinger') -> 'VegData':
        """
        Return a copy with the species matrix transformed.

        Parameters
        ----------
        method : str
            Any method supported by
            :class:`~VegZ.data_management.transformations.DataTransformer`.
        """
        from .data_management.transformations import DataTransformer

        transformed = DataTransformer().transform(self.species, method)
        result = self.subset()
        result.species = pd.DataFrame(
            np.asarray(transformed, dtype=float),
            index=self.species.index, columns=self.species.columns)
        result.metadata['transform'] = method
        return result

    # ------------------------------------------------------------------
    # Interoperability
    # ------------------------------------------------------------------

    def to_vegz(self) -> Any:
        """
        Build a configured :class:`~VegZ.core.VegZ` instance from this data set.

        Returns
        -------
        VegZ
            With ``species_matrix`` and ``environmental_data`` already set, so
            no column guessing is needed.
        """
        from .core import VegZ

        analyzer = VegZ()
        analyzer.species_matrix = self.species
        analyzer.data = self.species
        analyzer.environmental_data = self.environment
        analyzer.metadata.update(self.metadata)
        return analyzer

    def summary(self) -> Dict[str, Any]:
        """A quick description of the data set's size, fill and components."""
        values = self.species.to_numpy(dtype=float)
        occupied = values > 0

        return {
            'n_sites': self.n_sites,
            'n_species': self.n_species,
            'total_abundance': float(values.sum()),
            'fill': float(occupied.mean()),
            'mean_richness': float(occupied.sum(axis=1).mean()),
            'min_richness': int(occupied.sum(axis=1).min()),
            'max_richness': int(occupied.sum(axis=1).max()),
            'empty_sites': int((~occupied.any(axis=1)).sum()),
            'absent_species': int((~occupied.any(axis=0)).sum()),
            'has_environment': self.has_environment,
            'has_traits': self.has_traits,
            'has_phylogeny': self.has_phylogeny,
        }

    @classmethod
    def from_files(cls, species_path: str,
                   environment_path: Optional[str] = None,
                   traits_path: Optional[str] = None,
                   phylogeny_path: Optional[str] = None,
                   index_col: Union[int, str] = 0,
                   species_column: str = 'species',
                   **read_kwargs: Any) -> 'VegData':
        """
        Load a data set from CSV/Excel files.

        Parameters
        ----------
        species_path : str
            Site-by-species matrix; the first column is used as the site index.
        environment_path, traits_path, phylogeny_path : str, optional
            Companion tables.
        index_col : int or str
            Index column for the species and environment tables.
        species_column : str
            Species-name column in the traits file.
        **read_kwargs
            Passed to the pandas reader.
        """
        def read(path: str, use_index: bool = True) -> pd.DataFrame:
            reader = (pd.read_excel if str(path).lower().endswith(('.xlsx', '.xls'))
                      else pd.read_csv)
            return reader(path, index_col=index_col if use_index else None,
                          **read_kwargs)

        def read_traits(path: str) -> pd.DataFrame:
            """
            Read a traits table, finding the species names wherever they are.

            They may be in a named ``species_column``, or - as produced by a
            plain ``DataFrame.to_csv()`` round trip - in an unnamed leading
            index column. Reading with ``index_col=None`` alone turns the
            latter into a data column, and alignment then fails with "share no
            species labels" on a file that looks perfectly reasonable.
            """
            traits = read(path, use_index=False)
            if species_column in traits.columns or traits.empty:
                return traits

            first = traits.columns[0]
            if not pd.api.types.is_numeric_dtype(traits[first]):
                return traits.set_index(first)
            return traits

        return cls(
            species=read(species_path),
            environment=None if environment_path is None else read(environment_path),
            traits=None if traits_path is None else read_traits(traits_path),
            phylogeny=None if phylogeny_path is None else read(phylogeny_path),
            species_column=species_column,
        )
