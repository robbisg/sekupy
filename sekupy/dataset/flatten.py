"""Brain-volume <-> feature-vector conversion.

Replaces the old `FlattenMapper`/`ChainMapper`/`StaticFeatureSelection`
machinery, which existed to keep a *reversible* mapper alive across an
arbitrary chain of dataset transformations. In practice only one round trip
is ever used (flatten -> mask ->, much later, reverse a per-feature result
back into brain space, see `analysis/searchlight`), and masking is just an
ordinary feature selection. So instead of a reversible mapper object, we
simply carry each feature's original volume coordinates as a normal feature
attribute (`fa[space]`) -- plain `ds[:, mask]` indexing already keeps that
attribute correctly aligned, with no special-casing required.
"""
from __future__ import annotations

import numpy as np

from sekupy.dataset.dataset import Dataset


def flatten_dataset(ds: Dataset, space: str = "voxel_indices") -> Dataset:
    """Turn a Dataset whose samples are `(n_samples,) + volume_shape` into one
    with `(n_samples, n_voxels)` samples, recording each feature's original
    coordinates in `fa[space]` and the original shape in `a['flatten_shape']`
    so `unflatten_to_brain` can reverse it later."""
    orig_shape = ds.samples.shape[1:]
    flat_samples = ds.samples.reshape(ds.samples.shape[0], -1)

    fa = dict(ds.fa)
    if space is not None:
        fa[space] = np.array(list(np.ndindex(orig_shape)))

    a = dict(ds.a)
    a["flatten_shape"] = orig_shape
    a["flatten_space"] = space

    return Dataset(flat_samples, sa=dict(ds.sa), fa=fa, a=a)


def flatten_array(values: np.ndarray, ds: Dataset) -> np.ndarray:
    """Flatten a single volume-shaped array (e.g. a mask or an auxiliary ROI
    image) into the same 1D feature order `flatten_dataset` used, without
    wrapping it in a Dataset. Replaces `mapper.forward1(value)`."""
    orig_shape = ds.a["flatten_shape"]
    values = np.asarray(values)
    if values.shape != tuple(orig_shape):
        raise ValueError(
            f"expected an array of shape {tuple(orig_shape)}, got {values.shape}"
        )
    return values.reshape(-1)


def unflatten_samples_to_brain(samples: np.ndarray, ds: Dataset) -> np.ndarray:
    """Reverse a 2D `(n_samples, n_features)` array back into
    `(n_samples,) + original volume shape`, filling every voxel not covered
    by `ds` with 0. Replaces `.a.mapper.reverse()`."""
    space = ds.a["flatten_space"]
    orig_shape = tuple(ds.a["flatten_shape"])
    coords = ds.fa[space]

    samples = np.atleast_2d(np.asarray(samples))
    if samples.shape[-1] != len(coords):
        raise ValueError(
            f"expected {len(coords)} features per sample, got {samples.shape[-1]}"
        )

    brain = np.zeros((samples.shape[0],) + orig_shape, dtype=samples.dtype)
    brain[(slice(None),) + tuple(coords.T)] = samples
    return brain


def unflatten_to_brain(values: np.ndarray, ds: Dataset) -> np.ndarray:
    """Reverse a 1D per-feature statistic back into the dataset's original
    volume shape, filling every voxel not covered by `ds` with 0. Replaces
    `.a.mapper.reverse1()`."""
    return unflatten_samples_to_brain(np.asarray(values)[np.newaxis], ds)[0]
