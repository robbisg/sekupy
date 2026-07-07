# Contributing to sekupy

Thank you for considering a contribution to **sekupy**! This document explains
how to report bugs, propose new features, set up a development environment, run
the test suite, and submit a pull request.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Reporting Bugs](#reporting-bugs)
3. [Suggesting Enhancements](#suggesting-enhancements)
4. [Development Setup](#development-setup)
5. [Running the Tests](#running-the-tests)
6. [Code Style](#code-style)
7. [Pull Request Process](#pull-request-process)
8. [Releasing a New Version](#releasing-a-new-version)

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
By participating you agree to abide by its terms.

---

## Reporting Bugs

Please open an issue on the
[GitHub issue tracker](https://github.com/robbisg/sekupy/issues) and include:

- A **minimal, reproducible example** – the shortest script that triggers
  the bug.
- The **expected** and **actual** behaviour.
- The **sekupy version** (`python -c "import sekupy; print(sekupy.__version__)"`)
  and the output of `pip freeze | grep -E "sklearn|mne|nilearn|nibabel|numpy|scipy"`.
- The **operating system** and **Python version**.

---

## Suggesting Enhancements

Open a GitHub issue with the label `enhancement`.  Describe:

- The problem you are trying to solve.
- How you envision the enhancement working (API sketch welcome).
- Any alternative solutions you considered.

---

## Development Setup

```bash
# 1. Fork the repository on GitHub, then clone your fork
git clone https://github.com/<your-username>/sekupy.git
cd sekupy

# 2. Create a virtual environment (Python ≥ 3.9)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install the package in editable mode with all dev extras
pip install -e ".[test]"

# 4. Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

---

## Running the Tests

sekupy uses [pytest](https://docs.pytest.org/) and the tests live inside the
package under `sekupy/**/tests/`.

```bash
# Run the full test suite with coverage
pytest --pyargs sekupy --cov=sekupy --cov-report=term-missing

# Run a specific module
pytest sekupy/analysis/tests/test_pipeline.py -v

# Run tests for a particular Python version using tox
tox -e py312
```

The CI pipeline (GitHub Actions) runs the suite on Python 3.9, 3.10, 3.11,
and 3.12 on every push and pull request.

---

## Code Style

sekupy follows **PEP 8** and uses [ruff](https://docs.astral.sh/ruff/) for
linting and [numpydoc](https://numpydoc.readthedocs.io/) for docstrings.

```bash
# Check style
ruff check sekupy/

# Auto-fix safe issues
ruff check --fix sekupy/
```

Docstrings must follow the NumPy convention:

```python
def my_function(x, y):
    """Short one-line summary.

    Parameters
    ----------
    x : int
        Description of x.
    y : float
        Description of y.

    Returns
    -------
    float
        Description of the return value.
    """
```

---

## Pull Request Process

1. Create a **feature branch** from `master`:
   ```bash
   git checkout -b feature/my-new-analysis
   ```

2. Make your changes, write or update tests, and ensure the suite passes
   locally.

3. Update `CHANGELOG.md` under the `[Unreleased]` heading with a brief
   description of your change.

4. Push your branch and open a pull request against `master`.  The PR
   description should explain *what* changed and *why*.

5. A maintainer will review the PR.  Please respond to review comments
   within a reasonable time.  Once approved the maintainer will merge.

### Checklist before opening a PR

- [ ] Tests pass locally (`pytest --pyargs sekupy`)
- [ ] New functionality has tests
- [ ] Docstrings follow the NumPy convention
- [ ] `CHANGELOG.md` updated
- [ ] No secrets, credentials, or large binary files added

---

## Releasing a New Version

Releases are created by maintainers only:

1. Update `CHANGELOG.md`: rename `[Unreleased]` to the new version and date.
2. Create and push a git tag:
   ```bash
   git tag -a v0.1.0 -m "Release 0.1.0"
   git push origin v0.1.0
   ```
3. The GitHub Actions release workflow builds and publishes to PyPI
   automatically.
4. Create a Zenodo deposit for the tagged release and update the DOI badge
   in `README.md`.
