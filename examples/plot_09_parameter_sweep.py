"""
Parameter sweep with AnalysisIterator
=======================================

Sensitivity and robustness analyses require running the same analysis
multiple times with varying parameter settings.
:class:`~sekupy.analysis.iterator.AnalysisIterator` handles this cleanly,
without requiring nested ``for`` loops.

Three iteration modes are supported:

* ``combination``: Cartesian product of all parameter value lists.
* ``list``       : Element-wise pairing of equal-length lists.
* ``configuration``: A pre-built list of full configuration dictionaries.

This example shows:

* How to sweep over the SVM regularisation parameter ``C`` and the number of
  cross-validation splits using ``combination`` mode.
* How to collect accuracy scores across all parameter combinations.
* How to plot a sensitivity heatmap.
"""

# %%
# Build a synthetic dataset
# --------------------------

import numpy as np
import matplotlib.pyplot as plt
from sekupy.dataset.base import Dataset


def make_dataset(n_subjects=4, n_trials=40, n_features=50, seed=5):
    rng = np.random.default_rng(seed)
    n_total = n_subjects * n_trials
    conditions = np.tile(["A", "B"], n_trials // 2 * n_subjects)[:n_total]
    signal = np.where(conditions == "A", 1.0, -1.0)
    samples = rng.normal(signal[:, None] * 0.8, 1.0, (n_total, n_features))
    ds = Dataset(samples=samples)
    ds.sa["targets"] = conditions
    ds.sa["chunks"] = np.repeat(np.arange(n_subjects), n_trials)
    ds.fa["roi"] = np.ones(n_features, dtype=int)
    return ds


ds = make_dataset()

# %%
# Define base configuration and options grid
# -------------------------------------------

from sekupy.analysis.configurator import AnalysisConfigurator
from sekupy.analysis.iterator import AnalysisIterator
from sekupy.analysis.pipeline import AnalysisPipeline
from sekupy.analysis.decoding.roi_decoding import RoiDecoding
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit

base_config = {
    "prepro": ["target_transformer"],
    "target_transformer__attr": "targets",
    "estimator": [("clf", SVC(kernel="linear"))],
    "cv": StratifiedShuffleSplit,
    "cv__test_size": 0.2,
    "analysis": RoiDecoding,
    "analysis__n_jobs": 1,
    "analysis__permutation": 0,
    "kwargs__roi": ["roi"],
    "kwargs__cv_attr": "chunks",
    "scores": ["accuracy"],
}

# Grid: 4 values of C × 3 values of cv n_splits → 12 combinations
options = {
    "estimator__clf__C": [0.01, 0.1, 1.0, 10.0],
    "cv__n_splits": [5, 10, 20],
}

iterator = AnalysisIterator(
    options,
    AnalysisConfigurator,
    config_kwargs=base_config,
    kind="combination",
)

print(f"Total configurations to run: {len(list(AnalysisIterator(options, AnalysisConfigurator, config_kwargs=base_config, kind='combination')))}")

# %%
# Run all configurations and collect results
# -------------------------------------------

C_values = options["estimator__clf__C"]
split_values = options["cv__n_splits"]
acc_matrix = np.zeros((len(C_values), len(split_values)))

for conf in iterator:
    pipeline = AnalysisPipeline(conf, name="sensitivity").fit(ds)
    roi_key = list(pipeline._estimator.scores.keys())[0]
    acc = np.mean(pipeline._estimator.scores[roi_key][0]["test_accuracy"])

    # Recover the parameter values from the configurator
    C_val = conf._default_options.get("estimator__clf__C", 1.0)
    n_splits = conf._default_options.get("cv__n_splits", 10)

    i = C_values.index(C_val)
    j = split_values.index(n_splits)
    acc_matrix[i, j] = acc

# %%
# Plot the sensitivity heatmap
# -----------------------------

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(acc_matrix, cmap="YlOrRd", vmin=0.4, vmax=1.0, aspect="auto")
plt.colorbar(im, ax=ax, label="Mean accuracy")

ax.set_xticks(range(len(split_values)))
ax.set_yticks(range(len(C_values)))
ax.set_xticklabels([f"n_splits={s}" for s in split_values])
ax.set_yticklabels([f"C={c}" for c in C_values])
ax.set_title("Sensitivity analysis: SVM-C vs. CV splits")

for i in range(len(C_values)):
    for j in range(len(split_values)):
        ax.text(j, i, f"{acc_matrix[i, j]:.3f}", ha="center", va="center",
                fontsize=10, color="black")

plt.tight_layout()
plt.show()

print(f"\nBest accuracy {acc_matrix.max():.3f} at C={C_values[acc_matrix.argmax() // len(split_values)]}, "
      f"n_splits={split_values[acc_matrix.argmax() % len(split_values)]}")
