"""The core Dataset container: samples plus named sample/feature/dataset
attributes.

This replaces a vendored PyMVPA object model (`AttrDataset` plus a
`Collectable -> Collection` class tower). A full-codebase usage audit found
that, outside `sekupy/dataset/` itself, nothing needs more than plain named
arrays: `sa`/`fa` are length-checked dicts of arrays, `a` is an unconstrained
bag, and `ds[...]`/`ds[:, ...]` indexing keeps everything aligned.
"""
from __future__ import annotations

import copy as _copy
from typing import Any, Iterable

import numpy as np


class AttrDict(dict):
    """A dict that also supports attribute-style access (`d.x` == `d['x']`).

    This is the entire replacement for PyMVPA's `Collectable` -> `Collection`
    class tower: no wrapper objects, no `.value` indirection, just a dict.
    """

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key) from None

    def copy(self) -> "AttrDict":
        # dict.copy() always returns a plain dict, even on a subclass --
        # overridden here so `.a.copy()`/`.sa.copy()` (used throughout
        # analysis/base.py's _store_info) keep attribute-style access.
        return type(self)(self)


class _LengthCheckedAttrDict(AttrDict):
    """`sa`/`fa`: every value must be an array of the same length.

    Replaces PyMVPA's `UniformLengthCollection`. The expected length is kept
    as a real instance attribute (via `object.__setattr__`, bypassing
    `AttrDict.__setattr__` above so it doesn't get stored as a dict entry).
    """

    def __init__(self, items: dict | None = None, length: int | None = None):
        object.__setattr__(self, "_length", length)
        super().__init__()
        if items:
            for k, v in items.items():
                self[k] = v

    def __setitem__(self, key: str, value: Any) -> None:
        value = np.asarray(value)
        length = self._length
        if length is not None and len(value) != length:
            raise ValueError(
                f"'{key}' has length {len(value)}, expected {length}"
            )
        super().__setitem__(key, value)

    def copy(self) -> "_LengthCheckedAttrDict":
        return type(self)(self, length=self._length)


class Dataset:
    """samples (n_samples x n_features) + named per-sample/per-feature/dataset
    attributes."""

    def __init__(self, samples, sa: dict | None = None, fa: dict | None = None,
                 a: dict | None = None):
        samples = np.asarray(samples)
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        self.samples = samples

        self.sa = _LengthCheckedAttrDict(sa, length=self.samples.shape[0])
        self.fa = _LengthCheckedAttrDict(fa, length=self.samples.shape[1])
        self.a = AttrDict(a or {})

    @property
    def nfeatures(self) -> int:
        return self.samples.shape[1]

    @property
    def shape(self):
        return self.samples.shape

    def __len__(self) -> int:
        return self.samples.shape[0]

    def __array__(self, dtype=None):
        """Lets `np.mean(ds, ...)`/`np.asarray(ds)` etc. work directly on a
        Dataset, matching several call sites that pass a Dataset where an
        array is expected."""
        return np.asarray(self.samples, dtype=dtype)

    @property
    def targets(self):
        """Shortcut for `sa['targets']`, widely used as `ds.targets`."""
        return self.sa["targets"]

    @targets.setter
    def targets(self, value):
        self.sa["targets"] = value

    @property
    def chunks(self):
        """Shortcut for `sa['chunks']`."""
        return self.sa["chunks"]

    @chunks.setter
    def chunks(self, value):
        self.sa["chunks"] = value

    def __getitem__(self, key) -> "Dataset":
        if isinstance(key, tuple):
            sample_sel, feature_sel = key
        else:
            sample_sel, feature_sel = key, slice(None)

        # a bare int would collapse that axis in plain numpy indexing (e.g.
        # `ds[0]` returning a 1D array); wrap it so a single sample/feature
        # selection still comes back as a Dataset with that axis intact --
        # this is also what makes `for sample_ds in ds` iterate one-sample
        # Datasets via the default sequence protocol (__len__ + __getitem__).
        if isinstance(sample_sel, (int, np.integer)):
            sample_sel = [sample_sel]
        if isinstance(feature_sel, (int, np.integer)):
            feature_sel = [feature_sel]

        if isinstance(sample_sel, slice) or isinstance(feature_sel, slice):
            samples = self.samples[sample_sel, feature_sel]
        else:
            samples = self.samples[np.ix_(np.asarray(sample_sel), np.asarray(feature_sel))]

        sa = {k: v[sample_sel] for k, v in self.sa.items()}
        fa = {k: v[feature_sel] for k, v in self.fa.items()}
        return Dataset(samples, sa=sa, fa=fa, a=_shallow_copy_values(self.a))

    def copy(self, deep: bool = True) -> "Dataset":
        # `a` is always shallow-copied *per value* (not just the outer dict),
        # regardless of `deep` -- otherwise two datasets derived from the
        # same source (e.g. a session-scoped fixture) would share mutable
        # `a` entries like the `prepro` history list, and a transform on one
        # would silently show up on the other.
        if deep:
            samples = self.samples.copy()
            sa = {k: v.copy() for k, v in self.sa.items()}
            fa = {k: v.copy() for k, v in self.fa.items()}
        else:
            samples = self.samples
            sa = dict(self.sa)
            fa = dict(self.fa)
        return Dataset(samples, sa=sa, fa=fa, a=_shallow_copy_values(self.a))

    def __repr__(self) -> str:
        return (f"Dataset(samples.shape={self.samples.shape}, "
                f"sa={list(self.sa.keys())}, fa={list(self.fa.keys())}, "
                f"a={list(self.a.keys())})")

    def summary(self) -> str:
        """A short, human-readable description -- used only for logging (in
        analysis/base.py's `_store_info`), not asserted on by any test."""
        lines = [repr(self)]
        if self.nfeatures:
            lines.append(
                "stats: mean=%g std=%g min=%g max=%g"
                % (np.mean(self.samples), np.std(self.samples),
                   np.min(self.samples), np.max(self.samples))
            )
        if "targets" in self.sa:
            u, c = np.unique(self.sa["targets"], return_counts=True)
            lines.append("targets: " + ", ".join(f"{t}={n}" for t, n in zip(u, c)))
        if "chunks" in self.sa:
            lines.append("chunks: " + ", ".join(str(c) for c in np.unique(self.sa["chunks"])))
        return "\n".join(lines)


def _shallow_copy_values(a: dict) -> dict:
    """A dict with the same keys as `a`, each value shallow-copied. Used
    whenever a Dataset is derived from another (indexing, `.copy()`) so
    mutable `a` entries (e.g. the `prepro` history list) don't end up
    shared between the source and the derived dataset."""
    return {k: _copy.copy(v) for k, v in a.items()}


def expand_attribute(attr, length: int, attr_name: str):
    """Expand a scalar (or a too-short sequence meant to be repeated) into an
    array of the given length; a sequence already of that length is passed
    through as an array."""
    try:
        if isinstance(attr, str):
            raise TypeError
        if len(attr) != length:
            raise ValueError(
                f"Length of attribute '{attr_name}' [{len(attr)}] has to be {length}."
            )
        return np.asanyarray(attr)
    except TypeError:
        return np.repeat(attr, length)


def _require_matching_keys(datasets: Iterable[Dataset], attr: str, label: str) -> set:
    key_sets = [set(getattr(ds, attr).keys()) for ds in datasets]
    if not all(ks == key_sets[0] for ks in key_sets[1:]):
        raise ValueError(
            f"{label} of datasets to be stacked have varying attributes."
        )
    return key_sets[0]


def _merge_equal_only(datasets: Iterable[Dataset], attr: str) -> dict:
    """Keep a key only if every dataset has it with an identical value
    (mirrors the old `vstack`/`hstack` default `'drop_nonunique'` behavior)."""
    datasets = list(datasets)
    key_sets = [set(getattr(ds, attr).keys()) for ds in datasets]
    common_keys = set.intersection(*key_sets) if key_sets else set()

    merged = {}
    for key in common_keys:
        values = [getattr(ds, attr)[key] for ds in datasets]
        if all(np.array_equal(values[0], v) for v in values[1:]):
            merged[key] = values[0]
    return merged


def vstack(datasets: list[Dataset]) -> Dataset:
    """Stack datasets vertically (append samples). All inputs must share the
    exact same set of `sa` keys. Matching `fa` values survive; `a` is always
    dropped (no call site outside `dataset/` relies on merging it)."""
    datasets = list(datasets)
    if not datasets:
        raise ValueError("concatenation of zero-length sequences is impossible")
    if len(datasets) == 1:
        return datasets[0]

    sa_keys = _require_matching_keys(datasets, "sa", "Sample attributes")
    samples = np.concatenate([ds.samples for ds in datasets], axis=0)
    sa = {k: np.concatenate([ds.sa[k] for ds in datasets], axis=0) for k in sa_keys}
    fa = _merge_equal_only(datasets, "fa")
    return Dataset(samples, sa=sa, fa=fa, a={})


def hstack(datasets: list[Dataset]) -> Dataset:
    """Stack datasets horizontally (append features). All inputs must share
    the exact same set of `fa` keys. Matching `sa` values survive; `a` is
    always dropped."""
    datasets = list(datasets)
    if not datasets:
        raise ValueError("concatenation of zero-length sequences is impossible")
    if len(datasets) == 1:
        return datasets[0]

    fa_keys = _require_matching_keys(datasets, "fa", "Feature attributes")
    samples = np.concatenate([ds.samples for ds in datasets], axis=1)
    fa = {k: np.concatenate([ds.fa[k] for ds in datasets], axis=0) for k in fa_keys}
    sa = _merge_equal_only(datasets, "sa")
    return Dataset(samples, sa=sa, fa=fa, a={})
