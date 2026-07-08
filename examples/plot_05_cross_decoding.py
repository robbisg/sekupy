"""
Cross-condition decoding (temporal generalisation)
====================================================

Cross-decoding tests whether a classifier trained on one condition (or time
point) generalises to a different condition, providing evidence for shared
neural representations across contexts.

:class:`~sekupy.analysis.decoding.cross_decoding.CrossDecoding` wraps
:class:`~sekupy.analysis.decoding.roi_decoding.RoiDecoding` and runs a
train-on-A / test-on-B design for every pair of conditions specified.

This example shows:

* How to configure and run a cross-decoding analysis.
* How to build a generalisation matrix and visualise it as a heatmap.
"""

# %%
# Synthetic dataset with shared representation across contexts
# ------------------------------------------------------------
#
# We create two task contexts (``ctx_1``, ``ctx_2``) and two stimulus classes
# (``face``, ``object``). The signal is consistent across contexts so that a
# cross-decoding design should reveal generalisation.

import numpy as np
import matplotlib.pyplot as plt
from sekupy.dataset.base import Dataset


def make_cross_dataset(n_subjects=4, n_trials=40, n_features=50, seed=11):
    rng = np.random.default_rng(seed)
    contexts = ["ctx_1", "ctx_2"]
    classes = ["face", "object"]
    rows = []
    for ctx in contexts:
        for sub in range(n_subjects):
            for _ in range(n_trials // 2):
                for cls in classes:
                    signal = 1.0 if cls == "face" else -1.0
                    x = rng.normal(signal * 0.9, 1.0, n_features)
                    rows.append((x, cls, ctx, sub, f"sub-{sub+1:02d}"))

    samples = np.array([r[0] for r in rows])
    targets = np.array([r[1] for r in rows])
    context = np.array([r[2] for r in rows])
    chunks = np.array([r[3] for r in rows])
    subjects = np.array([r[4] for r in rows])

    ds = Dataset(samples=samples)
    ds.sa["targets"] = targets
    ds.sa["context"] = context
    ds.sa["chunks"] = chunks
    ds.sa["subject"] = subjects
    ds.fa["roi"] = np.ones(n_features, dtype=int)
    return ds


ds = make_cross_dataset()
print(f"Dataset: {ds.shape}")
print(f"Contexts: {sorted(set(ds.sa.context))}")

# %%
# Configure and run CrossDecoding
# --------------------------------

from sekupy.analysis.configurator import AnalysisConfigurator
from sekupy.analysis.pipeline import AnalysisPipeline
from sekupy.analysis.decoding.cross_decoding import CrossDecoding
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut

config = {
    "prepro": ["target_transformer"],
    "target_transformer__attr": "targets",
    "estimator": [("clf", SVC(C=1, kernel="linear"))],
    "cv": LeaveOneGroupOut,
    "analysis": CrossDecoding,
    "analysis__n_jobs": 1,
    "analysis__permutation": 0,
    "kwargs__roi": ["roi"],
    "kwargs__cv_attr": "chunks",
    "kwargs__attr": "context",
    "scores": ["accuracy"],
}

pipeline = AnalysisPipeline(
    AnalysisConfigurator(**config), name="cross_decoding_example"
).fit(ds)

# %%
# Build the generalisation matrix
# ---------------------------------
#
# For each pair of (train context, test context) we collect the mean accuracy
# and arrange it in a matrix — the *generalisation matrix*.

contexts = sorted(set(ds.sa.context))
n_ctx = len(contexts)
gen_matrix = np.zeros((n_ctx, n_ctx))

scores = pipeline._estimator.scores
for key, fold_results in scores.items():
    acc = np.mean(fold_results[0]["test_accuracy"])
    # Key encodes train/test context; use index ordering
    for i, ctx_i in enumerate(contexts):
        for j, ctx_j in enumerate(contexts):
            if f"{ctx_i}" in key and f"{ctx_j}" in key:
                gen_matrix[i, j] = acc

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(gen_matrix, vmin=0.3, vmax=1.0, cmap="RdYlGn", origin="upper")
plt.colorbar(im, ax=ax, label="Accuracy")
ax.set_xticks(range(n_ctx))
ax.set_yticks(range(n_ctx))
ax.set_xticklabels([f"Test\n{c}" for c in contexts])
ax.set_yticklabels([f"Train {c}" for c in contexts])
ax.set_title("Cross-decoding generalisation matrix")
for i in range(n_ctx):
    for j in range(n_ctx):
        ax.text(j, i, f"{gen_matrix[i, j]:.2f}", ha="center", va="center",
                color="black", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.show()
