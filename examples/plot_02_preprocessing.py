"""
Building preprocessing pipelines
==================================

The :mod:`sekupy.preprocessing` module provides a collection of
:class:`~sekupy.preprocessing.base.Transformer` objects that follow a
``transform(ds) → ds`` interface compatible with the rest of the package.

This example shows:

* How to compose a custom :class:`~sekupy.preprocessing.pipelines.PreprocessingPipeline`.
* The effect of z-score normalisation on the data distribution.
* How to slice samples and features from the dataset.
* How transformation parameters are stored inside the dataset for provenance.
"""

# %%
# Helper: create a synthetic dataset
# ------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sekupy.dataset.base import Dataset


def make_dataset(n_subjects=4, n_trials=30, n_features=80, seed=0):
    rng = np.random.default_rng(seed)
    n_total = n_subjects * n_trials
    conditions = np.tile(["face", "object"], n_trials // 2 * n_subjects)[:n_total]
    # add a drift to make normalisation visible
    drift = np.linspace(0, 5, n_total)[:, None]
    samples = rng.normal(0, 1, (n_total, n_features)) + drift
    ds = Dataset(samples=samples)
    ds.sa["targets"] = conditions
    ds.sa["chunks"] = np.repeat(np.arange(n_subjects), n_trials)
    ds.sa["subject"] = np.repeat(
        [f"sub-{i+1:02d}" for i in range(n_subjects)], n_trials
    )
    roi_labels = np.tile(np.arange(1, 5), n_features // 4 + 1)[:n_features]
    ds.fa["roi"] = roi_labels
    return ds


ds_raw = make_dataset()
print(f"Raw dataset shape: {ds_raw.shape}")
print(f"Raw mean ± std : {ds_raw.samples.mean():.2f} ± {ds_raw.samples.std():.2f}")

# %%
# Compose a preprocessing pipeline
# ---------------------------------
#
# Transformers are chained inside a
# :class:`~sekupy.preprocessing.pipelines.PreprocessingPipeline`.
# The standard fMRI pipeline applies run-wise detrending then z-scoring.

from sekupy.preprocessing.pipelines import PreprocessingPipeline
from sekupy.preprocessing.normalizers import FeatureZNormalizer, SampleZNormalizer
from sekupy.preprocessing import Detrender

prepro = PreprocessingPipeline(
    nodes=[
        Detrender(chunks_attr="chunks"),  # per-subject / per-run detrend
        FeatureZNormalizer(),              # column-wise z-score
        SampleZNormalizer(),               # row-wise z-score
    ]
)

ds_prepro = prepro.transform(ds_raw.copy())
print(f"\nAfter preprocessing:")
print(f"  mean ± std : {ds_prepro.samples.mean():.3f} ± {ds_prepro.samples.std():.3f}")

# %%
# Visualise the normalisation effect
# ------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(ds_raw.samples.ravel(), bins=60, color="steelblue", alpha=0.8)
axes[0].set_title("Raw data distribution")
axes[0].set_xlabel("Activation (a.u.)")
axes[0].set_ylabel("Count")

axes[1].hist(ds_prepro.samples.ravel(), bins=60, color="coral", alpha=0.8)
axes[1].set_title("After detrend + z-score")
axes[1].set_xlabel("Activation (z-score)")

plt.tight_layout()
plt.show()

# %%
# Slicing samples and features
# -----------------------------
#
# :class:`~sekupy.preprocessing.slicers.SampleSlicer` and
# :class:`~sekupy.preprocessing.slicers.FeatureSlicer` select subsets
# of the dataset by matching attribute values.

from sekupy.preprocessing.slicers import SampleSlicer, FeatureSlicer

# Select only one subject and two conditions
ds_sub = SampleSlicer(subject=["sub-01"], targets=["face", "object"]).transform(ds_prepro)
print(f"\nSub-01 only: {ds_sub.shape}")

# Select only ROI 1 and 2
ds_roi = FeatureSlicer(roi=[1, 2]).transform(ds_prepro)
print(f"ROI 1+2 only: {ds_roi.shape}")

# %%
# Provenance: transformation info stored in ds.a
# -----------------------------------------------
#
# After each transform step, parameters are recorded in ``ds.a``
# so the full processing history is always accessible.

print("\nDataset-level attributes (ds.a) after preprocessing:")
for key in ds_prepro.a.keys():
    print(f"  {key}: {ds_prepro.a[key].value}")
