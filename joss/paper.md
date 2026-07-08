---
title: 'Sekupy: A Python package for clean and reproducible multivariate neuroimaging analysis pipelines'
tags:
  - Python
  - neuroimaging
  - machine learning
  - decoding
  - MVPA
  - pipelines
  - fMRI
  - MEG
  - EEG
authors:
  - name: Roberto Guidotti
    orcid: 0000-0002-0748-1301
    affiliation: 1
affiliations:
  - name: Department of Neuroscience, Imaging and Clinical Sciences, University "G. D'Annunzio" Chieti-Pescara, Italy
date: 1 July 2026
bibliography: paper.bib
---

# Summary

Sekupy is a Python package designed to streamline the construction of reproducible multivariate analysis pipelines for neuroimaging data. It provides a unified interface for loading BIDS-organised datasets [@gorgolewski2016brain], applying preprocessing transformations, executing a range of multivariate analyses—including decoding, Representational Similarity Analysis (RSA), fingerprint identification, and brain-state clustering—and storing results in a BIDS-inspired directory structure. By adopting the scikit-learn `fit`/`transform` paradigm [@pedregosa2011scikit] and integrating with established libraries such as MNE-Python [@gramfort2013meg], nilearn [@abraham2014machine], and imbalanced-learn [@lemaitre2017imbalanced], sekupy should reduce the technical barrier to writing readable, extensible, and reusable neuroimaging analysis scripts.

# Statement of Need

Modern cognitive neuroscience relies on a diverse ecosystem of analysis tools. While packages such as nilearn [@abraham2014machine], MNE-Python [@gramfort2013meg], and PyMVPA [@hanke2009pymvpa] offer a solid set of isolated functionalities to process neuroimaging datasets, composing them into coherent, editable and end-to-end analysis pipelines remains difficult to achieve. Researchers frequently produce analysis scripts that are tightly coupled to a specific dataset or parameter choice, making re-use, auditing, and collaboration difficult.

The reproducibility crisis in neuroscience [@open2015estimating] has pushed the field to adopt community standards such as BIDS [@gorgolewski2016brain] for data organisation. The introduction of BIDS-apps such as fmriprep or mne-bids-pipeline provided a stand-alone support for BIDS-aware analysis preprocessing pipelines with high standards. However, customization of these tools is tricky and personalization of specific pipelines is still problematic for non experts. In addition, systematically changing parameters for subsequent analyses such as decoding analyses or standard univariate models can be important for multiverse approaches that enhance reproducibility and new scientific standards.

Sekupy addresses these gaps by providing:

1. A `DataLoader` class that abstracts BIDS-aware, subject-level data loading and accepts user-defined reader functions for arbitrary file formats and modalities.
2. A `PreprocessingPipeline` that chains scikit-learn-style `Transformer` objects and automatically records transformation parameters inside the dataset object for provenance tracking.
3. An `AnalysisConfigurator` / `AnalysisPipeline` pair that expresses the full analysis in a single configuration dictionary, enabling reproducible re-runs and simple parameter sweeps via `AnalysisIterator`.
4. Multiple built-in multivariate analyses: ROI-based and searchlight decoding, RSA, fingerprint (identifiability) analysis, and brain-state clustering.
5. A results management system that saves analysis outputs in a BIDS-inspired directory structure and reconstructs them as tidy `pandas` DataFrames for downstream statistical analysis and visualisation.

Compared to general-purpose pipelines such as fMRIPrep [@esteban2019fmriprep] or MNE-BIDS-Pipeline [@appelhoff2022mne], sekupy allows to make preprocessing choices for MEEG data but it operates on already-preprocessed data and focuses on the multivariate analysis stage. Compared to lower-level packages such as PyMVPA or nilearn, sekupy offers a configuration-driven interface that enforces a clean separation between data loading, preprocessing, analysis, and parameter sweeping.

# Design and Architecture

## The Dataset Object

The central data structure in sekupy is the `Dataset` class, obtained from PyMVPA's `AttrDataset` [@hanke2009pymvpa]. It stores:

- `samples`: a 2-D (or 3-D for temporal analyses) NumPy array (observations × features).
- `sa` (sample attributes): per-observation metadata such as experimental condition, subject identifier, or run number.
- `fa` (feature attributes): per-feature metadata such as brain atlas labels or sensor names.
- `a` (dataset attributes): general metadata accumulated across loading and preprocessing steps.

This three-level annotation scheme means that subsequent pipeline stages can introspect the dataset without additional arguments, keeping function signatures minimal and the code self-documenting. After each preprocessing step, the transformation parameters are stored in `ds.a`, providing a lightweight provenance trail.

## Data Loading

The `DataLoader` class abstracts the process of locating, filtering, and reading neuroimaging files across subjects. The user specifies a BIDS data root, a participant list (TSV or CSV), optional BIDS entity filters, and a `load_fx`—a callable that reads one file and returns a `Dataset`. Built-in loaders handle standard fMRI BIDS layouts and MEG/EEG data via MNE-Python; custom loaders are registered by name and called transparently.

A minimal configuration file (INI format) declares shared parameters such as the data path, subject file, image pattern, and ROI directory:

```ini
[path]
data_path     = /data/my_study
subjects      = participants.tsv
experiment    = working_memory
types         = fmri

[fmri]
sub_dir       = bold
event_file    = attributes
img_pattern   = data.nii.gz
runs          = 1
mask_dir      = masks
brain_mask    = brain_mask.nii.gz
```

Loading the dataset then requires only a few lines:

```python
from sekupy.io.loader import DataLoader
from sekupy.preprocessing.pipelines import StandardPreprocessingPipeline

loader = DataLoader(
    configuration_file="my_study.conf",
    task="fmri",
    loader="base",
)
ds = loader.fetch(prepro=StandardPreprocessingPipeline())
```

`fetch` iterates over subjects listed in `participants.tsv`, accumulates per-subject datasets, and concatenates them into a single group-level `Dataset`. The optional `prepro` argument applies a `PreprocessingPipeline` at the subject level before concatenation, which is important for normalisation steps that should not mix subjects.

## Preprocessing

The `sekupy.preprocessing` module provides a collection of `Transformer` objects that follow the `transform(ds) → ds` signature:

- **Normalizers**: column-wise or row-wise z-score (`FeatureZNormalizer`, `SampleZNormalizer`), sigma-score, or any callable imported from NumPy or SciPy.
- **Slicers**: `SampleSlicer` and `FeatureSlicer` select subsets of the dataset by matching sample or feature attribute values.
- **Balancers**: random under- and over-sampling (wrapping imbalanced-learn) to correct class imbalance before classification.
- **Detrender**: removes linear trends within chunks (e.g., runs) to reduce slow drift artefacts.
- **Mathematical transformers**: absolute value, sign flip, or user-supplied elementwise functions.

The built-in `StandardPreprocessingPipeline` chains run-wise and global detrending followed by feature-wise and sample-wise z-scoring, which is a common starting point for fMRI decoding:

```python
from sekupy.preprocessing.pipelines import PreprocessingPipeline
from sekupy.preprocessing.normalizers import FeatureZNormalizer, SampleZNormalizer
from sekupy.preprocessing import Detrender, SampleSlicer

prepro = PreprocessingPipeline(nodes=[
    Detrender(chunks_attr="file"),   # per-run detrend
    Detrender(),                     # global detrend
    FeatureZNormalizer(),            # column-wise z-score
    SampleZNormalizer(),             # row-wise z-score
])
```

## The Analysis Pipeline

The `AnalysisConfigurator` encapsulates the full specification of an analysis in a single Python dictionary. Parameter names follow a double-underscore convention borrowed from scikit-learn pipelines (`estimator__clf__C`, `analysis__n_jobs`). The `AnalysisPipeline` consumes a configurator, applies preprocessing, and delegates to the chosen analysis class.

```python
from sekupy.analysis.configurator import AnalysisConfigurator
from sekupy.analysis.pipeline import AnalysisPipeline
from sekupy.analysis.decoding.roi_decoding import RoiDecoding
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import LeaveOneGroupOut

config = {
    # preprocessing steps (resolved via the function_mapper registry)
    "prepro": ["sample_slicer", "target_transformer"],
    "sample_slicer__condition": ["face", "object"],
    "target_transformer__attr": "condition",

    # scikit-learn estimator pipeline
    "estimator": [
        ("fsel", SelectKBest(k=100)),
        ("clf",  SVC(C=1, kernel="linear")),
    ],
    "cv": LeaveOneGroupOut,

    # analysis class and its keyword arguments
    "analysis": RoiDecoding,
    "analysis__n_jobs": -1,
    "analysis__permutation": 0,
    "kwargs__roi_values": [("roi", [1]), ("roi", [2]), ("roi", [3])],
    "kwargs__cv_attr": "subject",

    "scores": ["accuracy"],
}

configurator = AnalysisConfigurator(**config)
pipeline = AnalysisPipeline(configurator, name="face_object_decoding")
pipeline.fit(ds)
pipeline.save(path="./results")
```

`AnalysisPipeline.fit` applies the preprocessing steps in sequence, then calls the analysis estimator's `fit` method. The subsequent `save()` call serialises results to a BIDS-inspired folder hierarchy under `results/derivatives/pipeline-face_object_decoding/` (Figure 1).

## Analysis Methods

### ROI-Based Decoding

`RoiDecoding` iterates over brain regions encoded in `dataset.fa` and runs a cross-validated classification or regression pipeline for each region. Permutation testing (controlled by the `permutation` parameter) is built in and produces an empirical null distribution for each ROI, enabling non-parametric inference.

### Searchlight Decoding

`SearchLight` is based on nilearn implementation with few modification such as the possibility to include a separate metric or scores and performs the same cross-validated classification in a sliding spherical neighbourhood across the brain volume, returning an accuracy map that can be projected back to MNI space using nilearn's surface or volumetric plotting utilities.

### Representational Similarity Analysis

The `RSA` class computes a pairwise dissimilarity vector from the neural data in each ROI (using any metric from `scipy.spatial.distance`) and stores the resulting representational dissimilarity matrix (RDM). Users can then correlate the neural RDM with model RDMs to test hypotheses about representational geometry [@kriegeskorte2008representational].

```python
from sekupy.analysis.rsa.rsa import RSA
from sekupy.analysis.configurator import AnalysisConfigurator
from sekupy.analysis.pipeline import AnalysisPipeline

rsa_config = {
    "prepro": ["sample_slicer"],
    "sample_slicer__condition": ["face", "object", "scene"],
    "analysis": RSA,
    "kwargs__metric": "euclidean",
    "kwargs__roi_values": [("roi", [1]), ("roi", [2])],
}

pipeline = AnalysisPipeline(
    AnalysisConfigurator(**rsa_config), name="rsa_analysis"
).fit(ds)
pipeline.save(path="./results")
```

### Fingerprint (Identifiability) Analysis

`Identifiability` quantifies how uniquely each subject can be re-identified across conditions or sessions using a correlation-based fingerprint approach [@finn2015functional]. It produces an identifiability matrix (self-similarity minus cross-subject similarity) and a prediction accuracy matrix over all pairwise combinations of conditions.

```python
from sekupy.analysis.fingerprint.fingerprint import Identifiability
from sekupy.analysis.configurator import AnalysisConfigurator
from sekupy.analysis.pipeline import AnalysisPipeline

fp_config = {
    "prepro": ["none"],
    "analysis": Identifiability,
    "kwargs__attr": "condition",
}

pipeline = AnalysisPipeline(
    AnalysisConfigurator(**fp_config), name="fingerprint"
).fit(ds)
pipeline.save(path="./results")
```

### Brain State Analysis

`Clustering` wraps scikit-learn clustering algorithms (K-Means, Gaussian Mixture Models, etc.) and applies them to neuroimaging time series to segment brain dynamics into discrete states. A built-in `VarianceSubsampler` reduces the data to the most variable features before clustering, lowering computational cost without discarding the most informative time points.

```python
from sekupy.analysis.states.base import Clustering
from sekupy.analysis.configurator import AnalysisConfigurator
from sekupy.analysis.pipeline import AnalysisPipeline
from sklearn.cluster import KMeans

state_config = {
    "prepro": ["none"],
    "analysis": Clustering,
    "estimator": [("kmeans", KMeans(n_clusters=5, random_state=42))],
}

pipeline = AnalysisPipeline(
    AnalysisConfigurator(**state_config), name="brain_states"
).fit(ds)
pipeline.save(path="./results")
```

## Parameter Iteration

`AnalysisIterator` makes it straightforward to run an analysis over a combinatorial or list-based grid of parameter settings without nested loops. Three iteration modes are supported:

- `combination`: the Cartesian product of all value lists.
- `list`: element-wise pairing of equal-length lists.
- `configuration`: a pre-built list of complete configuration dictionaries.

```python
from sekupy.analysis.iterator import AnalysisIterator
from sekupy.analysis.configurator import AnalysisConfigurator
from sekupy.analysis.pipeline import AnalysisPipeline

options = {
    "estimator__clf__C":     [0.01, 0.1, 1, 10],
    "analysis__permutation": [0, 1000],
}

iterator = AnalysisIterator(
    options, AnalysisConfigurator,
    config_kwargs=config,        # base configuration from earlier
    kind="combination",
)

for conf in iterator:
    AnalysisPipeline(conf, name="sensitivity").fit(ds).save(path="./results")
```

The loop above runs 8 analyses (4 regularisation values × 2 permutation settings) and saves each under a uniquely identified subfolder, enabling post-hoc comparison of all parameter combinations.

## Results Management and Statistical Analysis

After `save()` is called, results are stored as MATLAB `.mat` files (readable by NumPy via `scipy.io`) together with a `configuration.json` that records all pipeline parameters. The companion `get_results_bids` function reconstructs all saved outputs into a tidy `pandas` DataFrame:

```python
from sekupy.results import get_results_bids

df = get_results_bids(
    path="./results",
    pipeline="face_object_decoding",
    field_list=["sample_slicer"],
    scores=["accuracy"],
)
```

The returned DataFrame has one row per cross-validation fold and columns for all configuration fields (subject, ROI, condition, regularisation parameter, fold index, and score). This format integrates directly with standard Python statistics and visualisation libraries.

### Group-Level Statistical Inference

Because decoding accuracy at the group level is typically tested against theoretical chance (0.5 for binary classification), a one-sample t-test is the most common inferential step. The example below aggregates fold-level scores to the subject level, tests each ROI against chance, and applies Bonferroni correction:

```python
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

# Average within subject and ROI
summary = (
    df.groupby(["subject", "roi"])["score_accuracy"]
    .mean()
    .reset_index()
)

# One-sample t-test vs chance (0.5) per ROI
results = []
for roi, group in summary.groupby("roi"):
    t, p = ttest_1samp(group["score_accuracy"], popmean=0.5)
    results.append({"roi": roi, "t": t, "p_uncorrected": p,
                    "mean": group["score_accuracy"].mean()})

stats = pd.DataFrame(results)

# Bonferroni correction
_, stats["p_corrected"], _, _ = multipletests(
    stats["p_uncorrected"], method="bonferroni"
)
stats["significant"] = stats["p_corrected"] < 0.05
print(stats.sort_values("p_corrected"))
```

### Visualisation

Tidy DataFrames produced by `get_results_bids` are ready for seaborn's categorical plotting functions. The code below produces a bar chart with individual-subject data points overlaid, which is the format recommended by recent guidelines on data visualisation in neuroscience [@weissgerber2015beyond]:

```python
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4))

# Bar chart: mean accuracy per ROI
sns.barplot(
    data=summary, x="roi", y="score_accuracy",
    order=stats.sort_values("mean", ascending=False)["roi"],
    color="steelblue", alpha=0.7, ax=ax,
)

# Overlay individual subjects
sns.stripplot(
    data=summary, x="roi", y="score_accuracy",
    order=stats.sort_values("mean", ascending=False)["roi"],
    color="black", size=4, jitter=True, ax=ax,
)

ax.axhline(0.5, color="red", linestyle="--", label="Chance level")
ax.set_xlabel("Brain Region")
ax.set_ylabel("Decoding Accuracy")
ax.set_title("ROI Decoding: Face vs. Object")
ax.legend()
plt.tight_layout()
plt.savefig("decoding_accuracy.pdf")
```

### RSA Result Visualisation

RSA results can be visualised as representational dissimilarity matrices using seaborn's heatmap:

```python
from scipy.spatial.distance import squareform
import seaborn as sns
import matplotlib.pyplot as plt

# pipeline.save() stores the RDM vector; load it back
rdm_vector = pipeline._estimator.scores["mask-roi_value-1"]
conditions  = pipeline._estimator.conditions

rdm_matrix = squareform(rdm_vector)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    rdm_matrix,
    xticklabels=conditions, yticklabels=conditions,
    cmap="viridis", square=True, ax=ax,
)
ax.set_title("Representational Dissimilarity Matrix – ROI 1")
plt.tight_layout()
plt.savefig("rdm_roi1.pdf")
```

# Research Impact

Sekupy has been used as the primary analysis framework to enhance the reproducibility and the transparency of decoding papers. Published applications include decoding of memory-related neural representations from fMRI data, functional connectivity fingerprinting in MEG resting-state recordings, and RSA-based comparison of neural and computational model representational.

# Availability

Sekupy is available on PyPI (`pip install sekupy`) and on GitHub at <https://github.com/robbisg/sekupy>. Documentation and tutorial notebooks are hosted at <https://sekupy.readthedocs.io>. An archived release is deposited on Zenodo (<https://doi.org/10.5281/zenodo.XXXXXXX>). The package is tested with pytest and continuous integration via GitHub Actions across Python 3.9–3.12, with coverage reporting through Codecov.

# Acknowledgements

The author thanks colleagues at the Mambo Lab, University "G. D'Annunzio" Chieti-Pescara, for feedback and real-world testing of early versions of the package. The author declares no conflicts of interest. Generative AI tools were used for revising this paper.

# References
