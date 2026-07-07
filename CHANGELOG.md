# Changelog

All notable changes to **sekupy** are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
sekupy uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- 10 sphinx-gallery examples covering data loading, preprocessing, ROI
  decoding, permutation testing, cross-decoding, RSA, fingerprint analysis,
  brain-state clustering, parameter sweeps, and results visualisation.
- sphinx-gallery wired into the Sphinx documentation build (`conf.py`).
- JOSS paper (`joss/paper.md`) with full bibliography (`joss/paper.bib`).
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff`, `CHANGELOG.md`.

---

## [0.0.1] – 2024-07-26

### Added
- Full migration from `pyitab` to `sekupy`; package now published on PyPI
  (`pip install sekupy`).
- `DataLoader` with BIDS-aware subject iteration and pluggable `load_fx`
  reader functions.
- `AnalysisConfigurator` / `AnalysisPipeline` pair: express full analysis
  pipelines in a single configuration dictionary.
- `AnalysisIterator`: combinatorial and list-based parameter sweeps without
  nested loops.
- `RoiDecoding`: ROI-based cross-validated classification / regression.
- `SearchLight`: spherical searchlight decoding over brain volumes.
- `CrossDecoding`: train-on-A / test-on-B generalisation analysis.
- `RSA`: Representational Similarity Analysis with arbitrary distance metrics.
- `Identifiability` (fingerprint analysis): subject re-identification from
  functional connectivity profiles.
- `Clustering`: brain-state segmentation via scikit-learn clustering
  algorithms.
- `PreprocessingPipeline` with `Detrender`, `FeatureZNormalizer`,
  `SampleZNormalizer`, `SampleSlicer`, `FeatureSlicer`, `Balancer`, and more.
- Results management: BIDS-inspired directory layout, `get_results_bids` /
  `get_results` loaders returning tidy `pandas` DataFrames.
- Sphinx documentation hosted on ReadTheDocs.
- GitHub Actions CI across Python 3.9, 3.10, 3.11.
- Codecov integration.

### Changed
- Package renamed from `pyitab` to `sekupy`; all public imports updated.
- Build system migrated to `hatchling` with VCS-based versioning.

---

## Pre-release history (pyitab → sekupy, 2018–2024)

The package was publicly developed under the name `pyitab` from 2018 and
transitioned to `sekupy` in 2024.  Key milestones during that period:

- **2020** – Added simulation utilities (autoregressive, Kuramoto,
  connectivity-state models), brain-state pipeline, temporal decoding, and
  combinatorial iterator.
- **2021** – Initial JOSS paper draft.
- **2022** – RSA searchlight integration, MNE-based MEG/EEG data loader,
  connectivity preprocessing transformers.
- **2023** – Refactored fingerprint / identifiability analysis; improved
  BIDS results storage.
- **2024** – Pypi release; ReadTheDocs deployment; migration to `hatchling`.

[Unreleased]: https://github.com/robbisg/sekupy/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/robbisg/sekupy/releases/tag/v0.0.1
