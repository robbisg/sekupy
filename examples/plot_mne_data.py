"""
Loading and preprocessing MEG/EEG data with MneDataLoader
===========================================================

:class:`~sekupy.io.mne.MneDataLoader` mirrors the interface of
:class:`~sekupy.io.loader.DataLoader` but reads continuous MEG/EEG
recordings (``.fif``, ``.edf``, ``.set``, ``.vhdr``, ...) into
:class:`mne.io.Raw` objects instead of building a stacked
:class:`~sekupy.dataset.base.Dataset`.

This example shows:

* How to lay out a raw MEG/EEG dataset for ``MneDataLoader`` (one
  subfolder per subject, one file pattern per task).
* How to call ``loader.fetch()`` with a list of
  :class:`~sekupy.preprocessing.mne.base.MneTransformer` steps to load
  *and* preprocess every subject in one call.
* Inspecting the effect of filtering and resampling on the power
  spectrum.

Sekupy does not bundle a raw MEG/EEG dataset (only derived connectivity
matrices, see ``plot_01_load_data``), so here we synthesize a tiny
multi-subject EEG dataset on disk to keep the example self-contained.
"""

# %%
# Create a synthetic multi-subject EEG dataset
# ---------------------------------------------
#
# ``MneDataLoader`` expects the same layout as ``DataLoader``: one
# subfolder per subject under ``data_path``, with the recording(s) for
# a given task inside ``sub_dir``. We write two subjects, each with a
# short recording contaminated by 50 Hz line noise and slow drift, so
# that the effect of preprocessing is visible later on.

import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import mne

mne.set_log_level("WARNING")

data_dir = tempfile.mkdtemp(prefix="sekupy_mne_example_")
ch_names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
sfreq = 250.0
n_times = int(sfreq * 20)  # 20 s recording

rng = np.random.default_rng(42)
subjects = ["sub-01", "sub-02"]

for subj in subjects:
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    times = np.arange(n_times) / sfreq

    data = rng.normal(0, 2e-6, (len(ch_names), n_times))
    data += 3e-6 * np.sin(2 * np.pi * 50 * times)  # line noise
    data += 5e-6 * np.sin(2 * np.pi * 0.1 * times)  # slow drift

    raw = mne.io.RawArray(data, info)

    sub_dir = os.path.join(data_dir, subj, "eeg")
    os.makedirs(sub_dir, exist_ok=True)
    raw.save(os.path.join(sub_dir, f"{subj}_task-rest_raw.fif"), overwrite=True)

# %%
# Write the configuration file
# -----------------------------
#
# As for ``DataLoader``, a small INI file describes where the data
# lives and how to find it. Since we don't have a ``participants.csv``
# here, ``subjects`` just needs to point to a (non-existent) file: in
# that case ``MneDataLoader`` falls back to listing the subfolders of
# ``data_path``.

conf_text = f"""
[path]
data_path={data_dir}
subjects=participants.csv
experiment=eeg_demo
types=eeg

[eeg]
sub_dir=eeg
img_pattern=.fif
"""

configuration_file = os.path.join(data_dir, "eeg_demo.conf")
with open(configuration_file, "w") as f:
    f.write(conf_text)

# %%
# Load and preprocess in one call
# ---------------------------------
#
# ``fetch`` accepts a list of :class:`~sekupy.preprocessing.mne.base.MneTransformer`
# steps, applies them to every subject's ``Raw`` right after loading,
# and returns a list of ``(subject, raw)`` tuples.

from sekupy.io.mne import MneDataLoader
from sekupy.preprocessing.mne import Filter, NotchFilter, Resample

loader = MneDataLoader(
    configuration_file=configuration_file,
    task="eeg",
    preload=True,
)

prepro = [
    NotchFilter(freqs=[50]),
    Filter(l_freq=1.0, h_freq=40.0),
    Resample(sfreq=100.0),
]

data = loader.fetch(prepro=prepro)

for subj, raw in data:
    print(f"{subj}: {len(raw.ch_names)} channels, sfreq={raw.info['sfreq']} Hz, "
          f"duration={raw.times[-1]:.1f} s")

# %%
# Visualise the effect of preprocessing
# ---------------------------------------
#
# Reload the first subject without preprocessing to compare its power
# spectrum before and after filtering/resampling.

raw_orig = loader.fetch(subject=subjects[0])[0][1]
raw_clean = data[0][1]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
raw_orig.compute_psd(fmax=40).plot(axes=axes[0], show=False)
axes[0].set_title(f"{subjects[0]}: before preprocessing")

raw_clean.compute_psd(fmax=40).plot(axes=axes[1], show=False)
axes[1].set_title(f"{subjects[0]}: after notch + band-pass + resample")

plt.tight_layout()
plt.show()
