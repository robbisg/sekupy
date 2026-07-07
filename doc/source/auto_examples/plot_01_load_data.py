"""
Loading neuroimaging data with DataLoader
=========================================

Sekupy's :class:`~sekupy.io.loader.DataLoader` abstracts the process of
locating, filtering, and reading neuroimaging files across subjects from a
BIDS-organised dataset.

This example shows:

* How to write a minimal INI configuration file for a BIDS study.
* How to call ``loader.fetch()`` to obtain a group-level
  :class:`~sekupy.dataset.base.Dataset`.
* The three-level annotation structure (``sa``, ``fa``, ``a``) of the
  returned dataset.
* A quick visualisation of the sample-attribute table.
"""

# %%
# The DataLoader pattern
# ----------------------
#
# In a real study, the user first creates an INI configuration file that
# describes the study layout::
#
#   [path]
#   data_path     = /data/my_study
#   subjects      = participants.tsv
#   experiment    = working_memory
#   types         = fmri
#
#   [fmri]
#   sub_dir       = bold
#   event_file    = attributes
#   img_pattern   = data.nii.gz
#   runs          = 1
#   mask_dir      = masks
#   brain_mask    = brain_mask.nii.gz
#
# Then loading is a two-liner:
#
# .. code-block:: python
#
#   from sekupy.io.loader import DataLoader
#   from sekupy.preprocessing.pipelines import StandardPreprocessingPipeline
#
#   loader = DataLoader(
#       configuration_file="my_study.conf",
#       task="fmri",
#       loader="base",
#   )
#   ds = loader.fetch(prepro=StandardPreprocessingPipeline())
#
# For this example we use the sekupy bundled test dataset which follows
# exactly the same convention.

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sekupy

# %%
# Load the bundled fMRI dataset
# -----------------------------
#
# The bundled dataset has 4 subjects, each with 30 trials encoding
# a memory task with two conditions (``L`` = lure, ``F`` = foil).

from sekupy.io.loader import DataLoader
from sekupy.preprocessing.pipelines import StandardPreprocessingPipeline

data_dir = os.path.join(os.path.dirname(sekupy.__file__), "io", "data", "fmri")
configuration_file = os.path.join(data_dir, "fmri.conf")

loader = DataLoader(
    configuration_file=configuration_file,
    task="fmri",
    loader="base",
)
ds = loader.fetch(prepro=StandardPreprocessingPipeline())

print(f"Dataset shape : {ds.shape}  (samples × features)")
print(f"Sample attributes : {list(ds.sa.keys())}")
print(f"Feature attributes: {list(ds.fa.keys())}")
print(f"Subjects          : {sorted(set(ds.sa.subject))}")
print(f"Conditions        : {sorted(set(ds.sa.decision))}")

# %%
# Visualise the sample-attribute table
# -------------------------------------
#
# The ``sa`` (sample attributes) dictionary behaves like a DataFrame:
# each key maps to a 1-D array aligned with the rows of ``ds.samples``.

attr_df = pd.DataFrame(
    {k: ds.sa[k].value for k in ["subject", "decision", "chunks", "evidence"]},
)
attr_df["sample_index"] = np.arange(len(attr_df))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Condition distribution per subject
counts = attr_df.groupby(["subject", "decision"]).size().unstack(fill_value=0)
counts.plot(kind="bar", ax=axes[0], color=["steelblue", "coral"])
axes[0].set_title("Trial counts per subject and condition")
axes[0].set_xlabel("Subject")
axes[0].set_ylabel("Number of trials")
axes[0].legend(title="Condition")
axes[0].tick_params(axis="x", rotation=0)

# Feature ROI label distribution
roi_counts = np.bincount(ds.fa.mask.astype(int))
axes[1].bar(np.arange(len(roi_counts)), roi_counts, color="mediumseagreen")
axes[1].set_title("Number of voxels per ROI label")
axes[1].set_xlabel("ROI label value")
axes[1].set_ylabel("Voxel count")

plt.tight_layout()
plt.show()
