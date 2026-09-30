"""Z-scoring, replacing the old `ZScoreMapper`.

Its train/untrain state machine was never exploited: every call site
(`preprocessing/normalizers.py`) does `mapper.train(ds); mapper.forward(ds)`
back-to-back on the same dataset and discards the mapper afterwards, and
reverse-mapping was never implemented. So this is a plain, stateless
fit-and-apply function.
"""
from __future__ import annotations

import numpy as np

from sekupy.dataset.dataset import Dataset


def zscore(ds: Dataset, chunks_attr: str | None = "chunks", param_est=None,
           dtype: str = "float64") -> Dataset:
    """Z-score `ds.samples`, per chunk if `chunks_attr` is given, else
    globally. `param_est`, if given, is an `(attr_name, attr_values)` pair
    restricting which samples are used to *estimate* mean/std -- the
    transform is still applied to every sample in the (chunk of the)
    dataset, matching the old `ZScoreMapper` semantics."""
    samples = ds.samples
    if np.issubdtype(samples.dtype, np.integer):
        samples = samples.astype(dtype)
    else:
        samples = samples.copy()

    if param_est is not None:
        est_attr, est_values = param_est
        est_mask = np.isin(ds.sa[est_attr], est_values)
    else:
        est_mask = np.ones(len(ds), dtype=bool)

    if chunks_attr is not None:
        chunks = ds.sa[chunks_attr]
        for c in np.unique(chunks):
            chunk_mask = chunks == c
            est_idx = chunk_mask & est_mask
            samples[chunk_mask] = _apply_zscore(
                samples[chunk_mask],
                samples[est_idx].mean(axis=0),
                samples[est_idx].std(axis=0),
            )
    else:
        samples = _apply_zscore(samples, samples[est_mask].mean(axis=0),
                                 samples[est_mask].std(axis=0))

    out = ds.copy(deep=False)
    out.samples = samples
    return out


def _apply_zscore(samples: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    samples = samples - mean
    std_nz = std != 0
    samples[:, std_nz] = samples[:, std_nz] / std[std_nz]
    return samples
