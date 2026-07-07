"""
Fingerprint (Identifiability) Analysis
========================================

The fingerprint or *identifiability* analysis (Finn et al., 2015) measures
how uniquely each individual can be re-identified from their neural data
when recorded in different conditions or sessions.

:class:`~sekupy.analysis.fingerprint.fingerprint.Identifiability` computes,
for every pair of conditions:

* **Self-similarity**: the diagonal of the between-condition correlation matrix.
* **Others-similarity**: the off-diagonal entries.
* **Identifiability score**: self − others.
* **Prediction accuracy**: fraction of subjects correctly identified.

References
----------
Finn, E. S. et al. (2015). Functional connectome fingerprinting: identifying
individuals using patterns of brain connectivity.
*Nature Neuroscience*, 18, 1664–1671.
"""

# %%
# Create a dataset with strong subject-specific fingerprints
# -----------------------------------------------------------
#
# Each subject has a unique brain "fingerprint" (a fixed offset vector)
# that is consistent across two conditions but idiosyncratic across subjects.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sekupy.dataset.base import Dataset

N_SUBJECTS = 8
N_FEATURES = 100
CONDITIONS = ["rest", "task"]
N_REPS = 10  # repetitions per condition per subject


def make_fingerprint_dataset(seed=42):
    rng = np.random.default_rng(seed)
    # Subject-specific fingerprints (strong individual signal)
    fingerprints = rng.normal(0, 2.0, (N_SUBJECTS, N_FEATURES))

    rows, targets, subjects = [], [], []
    for sub_i in range(N_SUBJECTS):
        for cond in CONDITIONS:
            for _ in range(N_REPS):
                noise = rng.normal(0, 0.3, N_FEATURES)
                # Mean pattern per subject is the fingerprint + condition noise
                cond_offset = rng.normal(0, 0.5, N_FEATURES)
                rows.append(fingerprints[sub_i] + cond_offset + noise)
                targets.append(f"sub-{sub_i+1:02d}")
                subjects.append(cond)  # store condition in 'subject' sa slot

    ds = Dataset(samples=np.array(rows))
    ds.sa["targets"] = np.array(targets)   # subject identity used for fingerprint
    ds.sa["condition"] = np.array(subjects)
    ds.fa["roi"] = np.ones(N_FEATURES, dtype=int)
    return ds


ds = make_fingerprint_dataset()
print(f"Dataset: {ds.shape}")
print(f"Unique identities: {sorted(set(ds.sa.targets))}")
print(f"Conditions       : {sorted(set(ds.sa.condition))}")

# %%
# Run the identifiability analysis
# ---------------------------------

from sekupy.analysis.fingerprint.fingerprint import Identifiability
from sekupy.analysis.configurator import AnalysisConfigurator
from sekupy.analysis.pipeline import AnalysisPipeline

config = {
    "prepro": ["none"],
    "analysis": Identifiability,
    "kwargs__attr": "targets",
}

pipeline = AnalysisPipeline(
    AnalysisConfigurator(**config), name="fingerprint_example"
).fit(ds)

# %%
# Visualise the identifiability matrix
# -------------------------------------

identifiability = pipeline._estimator.scores["matrix"]
accuracy = pipeline._estimator.scores["accuracy"]
subjects = pipeline._estimator.scores["vars"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.heatmap(
    identifiability,
    xticklabels=subjects,
    yticklabels=subjects,
    cmap="RdBu_r",
    center=0,
    annot=True,
    fmt=".2f",
    ax=axes[0],
)
axes[0].set_title("Identifiability matrix\n(self − others similarity)")
axes[0].tick_params(axis="x", rotation=45)

sns.heatmap(
    accuracy,
    xticklabels=subjects,
    yticklabels=subjects,
    cmap="Greens",
    vmin=0,
    vmax=1,
    annot=True,
    fmt=".2f",
    ax=axes[1],
)
axes[1].set_title("Prediction accuracy\n(fraction correctly identified)")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

# %%
# Summary statistics
# -------------------

diag_mean = np.mean(np.diag(identifiability))
offdiag = identifiability[~np.eye(len(subjects), dtype=bool)]
print(f"\nMean self-similarity   : {diag_mean:.3f}")
print(f"Mean cross-similarity  : {np.mean(offdiag):.3f}")
mean_acc = np.mean(np.diag(accuracy))
print(f"Mean prediction accuracy (diagonal): {mean_acc:.3f}")
