"""Polynomial (Legendre) detrending, replacing the old `PolyDetrendMapper`.

That mapper supported confound regressors (`opt_regs`) and custom sample
coordinates (`inspace`) -- a codebase-wide grep found neither is ever passed
by any caller (`preprocessing/functions.py`'s `Detrender` only ever sets
`polyord`/`chunks_attr`), so this only implements evenly-spaced per-chunk
Legendre detrending, matching what's actually used. Like `zscore`, this is a
plain fit-and-apply function since callers always train and forward in the
same breath and reverse-mapping was never implemented.
"""
from __future__ import annotations

import numpy as np
from scipy.special import legendre

from sekupy.dataset.dataset import Dataset


def poly_detrend(ds: Dataset, chunks_attr: str | None = "chunks",
                  polyord: int = 1) -> Dataset:
    """Remove a per-chunk polynomial trend (of Legendre polynomials up to
    `polyord`, evaluated at evenly-spaced points in [-1, 1]) from
    `ds.samples`, chunk by chunk if `chunks_attr` is given, else globally."""
    samples = ds.samples
    if np.issubdtype(samples.dtype, np.integer):
        samples = samples.astype("float64")
    else:
        samples = samples.copy()

    if chunks_attr is None:
        chunk_masks = [np.ones(len(ds), dtype=bool)]
    else:
        chunks = ds.sa[chunks_attr]
        chunk_masks = [chunks == c for c in np.unique(chunks)]

    for mask in chunk_masks:
        n = int(mask.sum())
        coords = np.linspace(-1, 1, n)
        design = np.column_stack(
            [_legendre_regressor(order, coords) for order in range(polyord + 1)]
        )
        fit, *_ = np.linalg.lstsq(design, samples[mask], rcond=None)
        samples[mask] = samples[mask] - design @ fit

    out = ds.copy(deep=False)
    out.samples = samples
    return out


def _legendre_regressor(order: int, x: np.ndarray) -> np.ndarray:
    """`scipy.special.legendre(order)` evaluated at `x`, guarding against the
    +-1 boundary occasionally producing `inf` (see scipy issue referenced in
    the original `PolyDetrendMapper`)."""
    poly = legendre(order)
    values = poly(x)
    bad = np.isinf(values)
    if np.any(bad):
        values[bad] = poly(x[bad] + 1e-10)
    return values
