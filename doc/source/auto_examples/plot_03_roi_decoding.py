"""
ROI-based decoding
==================

:class:`~sekupy.analysis.decoding.roi_decoding.RoiDecoding` iterates over
brain regions encoded in ``dataset.fa`` and runs a cross-validated
classification pipeline for each region.

This example shows:

* How to configure a decoding analysis with
  :class:`~sekupy.analysis.configurator.AnalysisConfigurator`.
* How to run it with :class:`~sekupy.analysis.pipeline.AnalysisPipeline`.
* How to extract per-ROI accuracy scores and visualise them.
"""

# %%
# Build a synthetic dataset with discriminative signal
# -----------------------------------------------------
#
# We inject a condition-specific signal only in ROIs 1 and 2 so that
# decoding accuracy should exceed chance in those regions.

import numpy as np
import matplotlib.pyplot as plt
from sekupy.dataset.base import Dataset


def make_decoding_dataset(n_subjects=4, n_trials=40, n_features=60, n_rois=5, seed=7):
    rng = np.random.default_rng(seed)
    n_total = n_subjects * n_trials
    conditions = np.tile(["face", "object"], n_trials // 2 * n_subjects)[:n_total]
    samples = rng.normal(0, 1, (n_total, n_features))

    roi_labels = np.tile(np.arange(1, n_rois + 1), n_features // n_rois + 1)[:n_features]

    # Add discriminative signal only in ROI 1 and 2
    signal = np.where(conditions == "face", 1.5, -1.5)
    for roi_val in [1, 2]:
        mask = roi_labels == roi_val
        samples[:, mask] += signal[:, None]

    ds = Dataset(samples=samples)
    ds.sa["targets"] = conditions
    ds.sa["chunks"] = np.repeat(np.arange(n_subjects), n_trials)
    ds.sa["subject"] = np.repeat(
        [f"sub-{i+1:02d}" for i in range(n_subjects)], n_trials
    )
    ds.fa["roi"] = roi_labels
    return ds


ds = make_decoding_dataset()
print(f"Dataset: {ds.shape[0]} samples × {ds.shape[1]} features")
print(f"Targets : {sorted(set(ds.sa.targets))}")
print(f"ROIs    : {sorted(set(ds.fa.roi))}")

# %%
# Configure and run the decoding analysis
# ----------------------------------------

from sekupy.analysis.configurator import AnalysisConfigurator
from sekupy.analysis.pipeline import AnalysisPipeline
from sekupy.analysis.decoding.roi_decoding import RoiDecoding
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit

config = {
    # Preprocessing: select target attribute for classification
    "prepro": ["target_transformer"],
    "target_transformer__attr": "targets",

    # Estimator: linear SVM
    "estimator": [("clf", SVC(C=1, kernel="linear"))],

    # Cross-validation: stratified shuffle-split
    "cv": StratifiedShuffleSplit,
    "cv__n_splits": 10,
    "cv__test_size": 0.2,

    # Analysis class and ROI iteration strategy
    "analysis": RoiDecoding,
    "analysis__n_jobs": 1,
    "analysis__permutation": 0,

    # Iterate over all unique values of the 'roi' feature attribute
    "kwargs__roi": ["roi"],
    "kwargs__cv_attr": "chunks",

    "scores": ["accuracy"],
}

configurator = AnalysisConfigurator(**config)
pipeline = AnalysisPipeline(configurator, name="roi_decoding_example")
pipeline.fit(ds)

# %%
# Extract and visualise per-ROI accuracy
# ----------------------------------------

scores = pipeline._estimator.scores

roi_labels, mean_acc, se_acc = [], [], []
for key, fold_results in scores.items():
    acc_per_fold = fold_results[0]["test_accuracy"]
    roi_labels.append(key)
    mean_acc.append(np.mean(acc_per_fold))
    se_acc.append(np.std(acc_per_fold) / np.sqrt(len(acc_per_fold)))

order = np.argsort(mean_acc)[::-1]
roi_labels = [roi_labels[i] for i in order]
mean_acc = [mean_acc[i] for i in order]
se_acc = [se_acc[i] for i in order]

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(roi_labels))
ax.bar(x, mean_acc, yerr=se_acc, capsize=4, color="steelblue", alpha=0.8)
ax.axhline(0.5, color="red", linestyle="--", label="Chance (0.5)")
ax.set_xticks(x)
ax.set_xticklabels(roi_labels, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Decoding accuracy")
ax.set_title("ROI Decoding: Face vs. Object")
ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()

print("\nTop ROI results:")
for lbl, acc in zip(roi_labels[:3], mean_acc[:3]):
    print(f"  {lbl}: {acc:.3f}")
