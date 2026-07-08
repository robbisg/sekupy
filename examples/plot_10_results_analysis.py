"""
Results loading, statistics, and visualisation
===============================================

After running any sekupy analysis pipeline, the results can be saved to disk
and later loaded into a tidy :class:`pandas.DataFrame` using
:func:`~sekupy.results.bids.get_results_bids` (BIDS layout) or
:func:`~sekupy.results.base.get_results` (legacy layout).

This example shows the full post-analysis workflow:

1. Run a decoding analysis and save the results.
2. Reload the saved results as a :class:`pandas.DataFrame`.
3. Compute group-level statistics (one-sample t-test vs. chance).
4. Apply multiple-comparison correction (Bonferroni).
5. Visualise: bar + strip chart (accuracy per ROI) and fold distribution.
"""

# %%
# Step 1 – Run a multi-ROI decoding and save
# -------------------------------------------

import numpy as np
import tempfile
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sekupy.dataset.base import Dataset


def make_multi_roi_dataset(n_subjects=6, n_trials=40, n_features=75,
                           n_rois=5, seed=17):
    rng = np.random.default_rng(seed)
    n_total = n_subjects * n_trials
    conditions = np.tile(["face", "object"], n_trials // 2 * n_subjects)[:n_total]
    samples = rng.normal(0, 1, (n_total, n_features))

    roi_labels = np.tile(np.arange(1, n_rois + 1), n_features // n_rois + 1)[:n_features]

    # ROI 1 and 2 carry discriminative signal; 3-5 are noise
    signal = np.where(conditions == "face", 1.2, -1.2)
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


ds = make_multi_roi_dataset()

from sekupy.analysis.configurator import AnalysisConfigurator
from sekupy.analysis.pipeline import AnalysisPipeline
from sekupy.analysis.decoding.roi_decoding import RoiDecoding
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit

config = {
    "prepro": ["target_transformer"],
    "target_transformer__attr": "targets",
    "estimator": [("clf", SVC(C=1, kernel="linear"))],
    "cv": StratifiedShuffleSplit,
    "cv__n_splits": 15,
    "cv__test_size": 0.2,
    "analysis": RoiDecoding,
    "analysis__n_jobs": 1,
    "analysis__permutation": 0,
    "kwargs__roi": ["roi"],
    "kwargs__cv_attr": "chunks",
    "scores": ["accuracy"],
}

pipeline = AnalysisPipeline(
    AnalysisConfigurator(**config), name="results_example"
).fit(ds)

# %%
# Step 2 – Collect scores into a tidy DataFrame
# -----------------------------------------------
#
# In a real study the results would be saved with ``pipeline.save(path=...)``
# and reloaded with :func:`~sekupy.results.bids.get_results_bids`.
# Here we build the DataFrame directly from the in-memory scores dict
# to keep the example self-contained.

records = []
for roi_key, fold_results in pipeline._estimator.scores.items():
    acc_per_fold = fold_results[0]["test_accuracy"]
    for fold_i, acc in enumerate(acc_per_fold):
        records.append(
            {
                "roi": roi_key,
                "fold": fold_i + 1,
                "accuracy": acc,
            }
        )

df = pd.DataFrame(records)
print(df.head(8).to_string(index=False))
print(f"\nROIs: {df['roi'].unique().tolist()}")

# %%
# Step 3 – Group-level statistical inference
# -------------------------------------------
#
# We average across folds within each ROI and run a one-sample t-test
# against chance level (0.5 for binary classification).

from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

summary = df.groupby("roi")["accuracy"].mean().reset_index(name="mean_acc")

t_stats, p_vals = zip(*[
    ttest_1samp(df.loc[df["roi"] == roi, "accuracy"], popmean=0.5)
    for roi in summary["roi"]
])
summary["t"] = t_stats
summary["p_uncorrected"] = p_vals

_, p_corr, _, _ = multipletests(summary["p_uncorrected"], method="bonferroni")
summary["p_bonferroni"] = p_corr
summary["significant"] = summary["p_bonferroni"] < 0.05

print("\nStatistical summary:")
print(summary.sort_values("mean_acc", ascending=False).to_string(index=False))

# %%
# Step 4 – Bar chart with individual fold points
# -----------------------------------------------

order = summary.sort_values("mean_acc", ascending=False)["roi"].tolist()

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(
    data=df, x="roi", y="accuracy",
    order=order, color="steelblue", alpha=0.7, ax=ax, errorbar="se",
)
sns.stripplot(
    data=df, x="roi", y="accuracy",
    order=order, color="black", size=3, jitter=True, alpha=0.5, ax=ax,
)
ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5, label="Chance (0.5)")

# Mark significant ROIs
for x_pos, roi in enumerate(order):
    row = summary[summary["roi"] == roi]
    if row["significant"].values[0]:
        ax.text(x_pos, df["accuracy"].max() + 0.02, "*",
                ha="center", fontsize=16, color="darkgreen")

ax.set_xlabel("Brain region")
ax.set_ylabel("Decoding accuracy")
ax.set_title("Group decoding accuracy per ROI\n(* = significant after Bonferroni correction)")
ax.set_xticklabels(order, rotation=25, ha="right", fontsize=9)
ax.legend()
ax.set_ylim(0.2, 1.0)
plt.tight_layout()
plt.show()

# %%
# Step 5 – Fold accuracy distribution (violin plot)
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 4))
sns.violinplot(
    data=df, x="roi", y="accuracy",
    order=order, palette="pastel", inner="box", ax=ax,
)
ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5, label="Chance")
ax.set_xlabel("Brain region")
ax.set_ylabel("Fold accuracy")
ax.set_title("Cross-validation fold accuracy distribution per ROI")
ax.set_xticklabels(order, rotation=25, ha="right", fontsize=9)
ax.legend()
plt.tight_layout()
plt.show()
