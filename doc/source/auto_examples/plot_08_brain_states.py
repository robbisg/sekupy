"""
Brain State Clustering
=======================

:class:`~sekupy.analysis.states.base.Clustering` groups time points (or
trials) of neuroimaging data into a discrete set of *brain states* using an
unsupervised clustering algorithm such as K-Means.

The analysis outputs:

* **labels**: the assigned cluster for each (subsampled) time point.
* **states**: cluster centroids in feature space.
* **dynamics**: the predicted state for *every* time point in the full dataset.

This example shows:

* How to configure and run K-Means brain-state segmentation.
* How to visualise state centroids and temporal dynamics.
* How to compute the *dwell time* distribution for each state.
"""

# %%
# Generate a synthetic multi-state time series
# --------------------------------------------
#
# We simulate 3 brain states, each defined by a distinct activation pattern,
# that alternate over 200 time points.

import numpy as np
import matplotlib.pyplot as plt
from sekupy.dataset.base import Dataset

N_STATES = 3
N_TIMEPOINTS = 200
N_FEATURES = 60
SEED = 99


def make_states_dataset(seed=SEED):
    rng = np.random.default_rng(seed)

    # One centroid per state
    centroids = rng.normal(0, 2.0, (N_STATES, N_FEATURES))

    # Generate a simple Markov sequence of states
    state_seq = [rng.integers(0, N_STATES)]
    for _ in range(N_TIMEPOINTS - 1):
        # Stay in same state with prob 0.85
        if rng.random() < 0.85:
            state_seq.append(state_seq[-1])
        else:
            options = [s for s in range(N_STATES) if s != state_seq[-1]]
            state_seq.append(rng.choice(options))
    state_seq = np.array(state_seq)

    samples = centroids[state_seq] + rng.normal(0, 0.5, (N_TIMEPOINTS, N_FEATURES))

    ds = Dataset(samples=samples)
    ds.sa["targets"] = np.array([f"state-{s}" for s in state_seq])
    ds.sa["chunks"] = np.zeros(N_TIMEPOINTS, dtype=int)
    ds.sa["time"] = np.arange(N_TIMEPOINTS)
    ds.fa["roi"] = np.ones(N_FEATURES, dtype=int)
    return ds, state_seq


ds, true_states = make_states_dataset()
print(f"Dataset shape: {ds.shape}  ({N_TIMEPOINTS} timepoints × {N_FEATURES} features)")

# %%
# Run K-Means clustering
# ----------------------

from sekupy.analysis.states.base import Clustering
from sekupy.analysis.configurator import AnalysisConfigurator
from sekupy.analysis.pipeline import AnalysisPipeline
from sklearn.cluster import KMeans

config = {
    "prepro": ["none"],
    "estimator": [("kmeans", KMeans(n_clusters=N_STATES, random_state=SEED, n_init=10))],
    "analysis": Clustering,
}

pipeline = AnalysisPipeline(
    AnalysisConfigurator(**config), name="brain_states_example"
).fit(ds)

scores = pipeline._estimator.scores
predicted_labels = scores.get("dynamics", scores.get("labels"))
if predicted_labels is None:
    predicted_labels = scores[list(scores.keys())[0]]
predicted_labels = np.asarray(predicted_labels).ravel()[:N_TIMEPOINTS]

# %%
# Visualise state centroids and temporal dynamics
# ------------------------------------------------

fig, axes = plt.subplots(3, 1, figsize=(12, 9))

# State centroids
centroids = scores.get("states")
if centroids is not None:
    centroids = np.asarray(centroids)
    im = axes[0].imshow(centroids, aspect="auto", cmap="RdBu_r")
    axes[0].set_yticks(range(N_STATES))
    axes[0].set_yticklabels([f"State {i}" for i in range(N_STATES)])
    axes[0].set_xlabel("Feature index")
    axes[0].set_title("State centroids")
    plt.colorbar(im, ax=axes[0])
else:
    axes[0].text(0.5, 0.5, "Centroids not available", ha="center", va="center",
                 transform=axes[0].transAxes)
    axes[0].set_title("State centroids")

# True state sequence
axes[1].plot(true_states, color="steelblue", linewidth=0.8)
axes[1].set_ylabel("True state")
axes[1].set_title("Ground-truth state sequence")
axes[1].set_ylim(-0.5, N_STATES - 0.5)
axes[1].set_yticks(range(N_STATES))

# Predicted state sequence
axes[2].plot(predicted_labels, color="coral", linewidth=0.8)
axes[2].set_ylabel("Predicted state")
axes[2].set_title("K-Means predicted state sequence")
axes[2].set_ylim(-0.5, N_STATES - 0.5)
axes[2].set_yticks(range(N_STATES))
axes[2].set_xlabel("Time point")

plt.tight_layout()
plt.show()

# %%
# Dwell-time distribution
# ------------------------

fig, ax = plt.subplots(figsize=(7, 3))
for state in range(N_STATES):
    dwells = []
    count = 0
    for lbl in predicted_labels:
        if lbl == state:
            count += 1
        elif count > 0:
            dwells.append(count)
            count = 0
    if count > 0:
        dwells.append(count)
    if dwells:
        ax.hist(dwells, bins=15, alpha=0.6, label=f"State {state}")

ax.set_xlabel("Dwell time (time points)")
ax.set_ylabel("Count")
ax.set_title("Dwell-time distribution per brain state")
ax.legend()
plt.tight_layout()
plt.show()
