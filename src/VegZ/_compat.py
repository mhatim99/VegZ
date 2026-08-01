"""
Compatibility helpers for VegZ.

Isolates third-party API differences (scikit-learn versions) and provides a
single, consistent way to handle optional dependencies so that importing VegZ
never emits warnings for packages the user has not installed.

Copyright (c) 2025 Mohamed Z. Hatim
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, Optional

import numpy as np

__all__ = ['optional_import', 'require', 'make_mds', 'as_generator']


def optional_import(module_name: str) -> Optional[ModuleType]:
    """
    Import a module if it is available, otherwise return ``None``.

    Unlike a bare ``try/except ImportError`` at module scope this never emits a
    warning at import time; callers are expected to check for ``None`` and warn
    (or raise) only when the functionality is actually requested.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def require(module_name: str, feature: str,
            extra: Optional[str] = None) -> ModuleType:
    """
    Import a module, raising an informative :class:`ImportError` if missing.

    Parameters
    ----------
    module_name : str
        Importable module name, e.g. ``'openpyxl'``.
    feature : str
        Human-readable description of the feature that needs the module.
    extra : str, optional
        Name of the VegZ optional-dependency extra that provides the module.
    """
    module = optional_import(module_name)
    if module is None:
        install = f"pip install VegZ[{extra}]" if extra else f"pip install {module_name}"
        raise ImportError(
            f"{feature} requires the optional dependency '{module_name}'. "
            f"Install it with: {install}"
        )
    return module


def make_mds(n_components: int = 2,
             metric: bool = True,
             precomputed: bool = True,
             n_init: int = 4,
             max_iter: int = 300,
             random_state: Optional[int] = None,
             **kwargs: Any) -> Any:
    """
    Construct a :class:`sklearn.manifold.MDS` across scikit-learn versions.

    scikit-learn 1.8 renamed the constructor arguments: the boolean ``metric``
    flag became ``metric_mds`` and the ``dissimilarity`` string became
    ``metric``. This helper accepts the historical VegZ semantics
    (``metric=False`` means *non-metric* MDS) and maps them onto whichever API
    the installed scikit-learn exposes.

    Parameters
    ----------
    n_components : int
        Number of ordination dimensions.
    metric : bool
        ``False`` performs non-metric MDS (the correct choice for NMDS).
    precomputed : bool
        Whether the input is a precomputed dissimilarity matrix.
    n_init, max_iter, random_state
        Passed through to scikit-learn.
    """
    import inspect

    from sklearn.manifold import MDS

    params = set(inspect.signature(MDS.__init__).parameters)
    common = dict(n_components=n_components, n_init=n_init,
                  max_iter=max_iter, random_state=random_state, **kwargs)

    if 'metric_mds' in params:
        # scikit-learn >= 1.8
        common['metric_mds'] = metric
        common['metric'] = 'precomputed' if precomputed else 'euclidean'
        if 'init' in params:
            common.setdefault('init', 'random')
    else:
        common['metric'] = metric
        common['dissimilarity'] = 'precomputed' if precomputed else 'euclidean'

    return MDS(**common)


def as_generator(random_state: Any) -> np.random.Generator:
    """
    Coerce ``random_state`` into a :class:`numpy.random.Generator`.

    Accepts ``None`` (fresh entropy), an integer seed, an existing
    ``Generator``, or a legacy ``RandomState``. Using a local generator keeps
    VegZ from ever mutating NumPy's global random state.
    """
    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, np.random.RandomState):
        return np.random.default_rng(random_state.randint(0, 2 ** 32 - 1))
    return np.random.default_rng(random_state)
