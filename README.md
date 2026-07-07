# sekupy

[![CI](https://github.com/robbisg/sekupy/actions/workflows/test.yaml/badge.svg)](https://github.com/robbisg/sekupy/actions/workflows/test.yaml)
[![codecov](https://codecov.io/gh/robbisg/sekupy/branch/master/graph/badge.svg)](https://codecov.io/gh/robbisg/sekupy)
[![Documentation Status](https://readthedocs.org/projects/sekupy/badge/?version=latest)](https://sekupy.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/sekupy.svg)](https://badge.fury.io/py/sekupy)
[![License: BSD-3](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

**sekupy** is a Python package for building clean, reproducible multivariate
analysis pipelines for neuroimaging data.  It was designed for decoding
analyses (MVPA) but also covers RSA, fingerprint identification, brain-state
clustering, and GLM-based univariate analyses.

Key features:

- **BIDS-aware data loading** — `DataLoader` iterates over subjects in a
  BIDS-organised dataset using pluggable reader functions.
- **Composable preprocessing** — chain `Transformer` objects
  (normalisation, slicing, balancing, filtering) into a
  `PreprocessingPipeline`.
- **Configuration-driven analysis** — express the full analysis in one
  Python dictionary; `AnalysisConfigurator` + `AnalysisPipeline` handle
  the rest.
- **Built-in multivariate analyses** — ROI decoding, searchlight decoding,
  cross-decoding, RSA, identifiability / fingerprint, and brain-state
  clustering.
- **Parameter sweeps** — `AnalysisIterator` runs combinatorial or list-wise
  parameter grids without nested loops.
- **BIDS-inspired results storage** — `save()` writes results to a tidy
  directory tree; `get_results_bids()` reloads them as a `pandas` DataFrame.

> **sekupy** integrates with
> [scikit-learn](https://scikit-learn.org),
> [nilearn](https://nilearn.github.io),
> [MNE-Python](https://mne.tools), and
> [imbalanced-learn](https://imbalanced-learn.org).

---

## Installation

```bash
pip install sekupy
```

Requires Python ≥ 3.9.

For the latest development version:

```bash
pip install git+https://github.com/robbisg/sekupy.git
```

---

## Quick start

```python
from sekupy.analysis.configurator import AnalysisConfigurator
from sekupy.analysis.pipeline import AnalysisPipeline
from sekupy.analysis.decoding.roi_decoding import RoiDecoding
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut

config = {
    "prepro": ["target_transformer", "sample_slicer", "balancer"],
    "target_transformer__attr": "condition",
    "sample_slicer__condition": ["face", "object"],
    "balancer__attr": "subject",

    "estimator": [("clf", SVC(C=1, kernel="linear"))],
    "cv": LeaveOneGroupOut,
    "scores": ["accuracy"],

    "analysis": RoiDecoding,
    "analysis__n_jobs": -1,
    "kwargs__roi": ["roi"],
    "kwargs__cv_attr": "subject",
}

pipeline = AnalysisPipeline(AnalysisConfigurator(**config), name="face_object").fit(ds)
pipeline.save(path="./results")
```

See the **[Examples Gallery](https://sekupy.readthedocs.io/en/latest/auto_examples/index.html)**
for 10 fully worked tutorials.

---

## Documentation

Full documentation, API reference, and examples are at
**<https://sekupy.readthedocs.io>**.

---

## Contributing

Contributions are welcome!  Please read [CONTRIBUTING.md](CONTRIBUTING.md)
before opening a pull request.  Bug reports and feature requests go to the
[issue tracker](https://github.com/robbisg/sekupy/issues).

---

## Citation

If you use sekupy in your research, please cite:

> Guidotti, R. (2026). *sekupy: A Python package for clean and reproducible
> multivariate neuroimaging analysis pipelines*. Journal of Open Source
> Software. <https://doi.org/10.21105/joss.XXXXX>

BibTeX:

```bibtex
@article{guidotti2026sekupy,
  author  = {Guidotti, Roberto},
  title   = {sekupy: A Python package for clean and reproducible
             multivariate neuroimaging analysis pipelines},
  journal = {Journal of Open Source Software},
  year    = {2026},
  doi     = {10.21105/joss.XXXXX},
}
```

A `CITATION.cff` file is included in the repository for automated citation
parsing (e.g., by GitHub's *Cite this repository* button).

---

## License

sekupy is distributed under the
[BSD 3-Clause License](LICENSE).
