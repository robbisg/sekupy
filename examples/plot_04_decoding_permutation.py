"""
Decoding with permutation testing
===================================

Permutation testing provides a non-parametric null distribution for decoding
accuracy, allowing inference without assumptions about the distribution of
scores.

Setting ``analysis__permutation`` to a positive integer runs the full
cross-validation ``N`` additional times with shuffled labels. The resulting
null distribution can be used to compute an empirical p-value.

This example shows:

* How to enable permutation testing in
  :class:`~sekupy.analysis.decoding.roi_decoding.RoiDecoding`.
* How to compute an empirical p-value from the null distribution.
* How to visualise the observed score against the null.
"""

# %%
# Dataset with moderate signal
# -----------------------------

import numpy as np
import matplotlib.pyplot as plt
from sekupy.dataset.base import Dataset


def make_dataset(n_subjects=4, n_trials=40, n_features=40, seed=3):
    rng = np.random.default_rng(seed)
    n_total = n_subjects * n_trials
    conditions = np.tile(["A", "B"], n_trials // 2 * n_subjects)[:n_total]
    signal = np.where(conditions == "A", 1.0, -1.0)
    samples = rng.normal(signal[:, None] * 0.8, 1.0, (n_total, n_features))
    ds = Dataset(samples=samples)
    ds.sa["targets"] = conditions
    ds.sa["chunks"] = np.repeat(np.arange(n_subjects), n_trials)
    roi_labels = np.ones(n_features, dtype=int)
    ds.fa["roi"] = roi_labels
    return ds


ds = make_dataset()

# %%
# Run decoding with permutations
# --------------------------------

from sekupy.analysis.configurator import AnalysisConfigurator
from sekupy.analysis.pipeline import AnalysisPipeline
from sekupy.analysis.decoding.roi_decoding import RoiDecoding
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit

N_PERMUTATIONS = 100  # increase to 1000 for publication-quality inference

config = {
    "prepro": ["target_transformer"],
    "target_transformer__attr": "targets",
    "estimator": [("clf", SVC(C=1, kernel="linear"))],
    "cv": StratifiedShuffleSplit,
    "cv__n_splits": 10,
    "cv__test_size": 0.2,
    "analysis": RoiDecoding,
    "analysis__n_jobs": 1,
    "analysis__permutation": N_PERMUTATIONS,
    "kwargs__roi": ["roi"],
    "kwargs__cv_attr": "chunks",
    "scores": ["accuracy"],
}

pipeline = AnalysisPipeline(
    AnalysisConfigurator(**config), name="permutation_test"
).fit(ds)

# %%
# Compute empirical p-value
# --------------------------

results = pipeline._estimator.scores["roi_value-1.0"]

# results[0] is the true run; results[1:] are permutations
true_acc = np.mean(results[0]["test_accuracy"])
null_dist = np.array([np.mean(r["test_accuracy"]) for r in results[1:]])
p_value = np.mean(null_dist >= true_acc)

print(f"True accuracy : {true_acc:.3f}")
print(f"Null mean     : {null_dist.mean():.3f} ± {null_dist.std():.3f}")
print(f"Empirical p   : {p_value:.3f}")

# %%
# Visualise the null distribution
# ---------------------------------

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(null_dist, bins=25, color="lightgrey", edgecolor="grey", label="Null distribution")
ax.axvline(true_acc, color="red", linewidth=2, label=f"Observed ({true_acc:.3f})")
ax.axvline(0.5, color="blue", linestyle="--", linewidth=1.5, label="Chance (0.5)")
ax.set_xlabel("Cross-validated accuracy")
ax.set_ylabel("Count")
ax.set_title(f"Permutation test  (p = {p_value:.3f}, N = {N_PERMUTATIONS})")
ax.legend()
plt.tight_layout()
plt.show()
