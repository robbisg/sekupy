# RSA within SearchLight Example

This document demonstrates how to use RSA (Representational Similarity Analysis) as an estimator within SearchLight analysis.

## Overview

The `RSAEstimator` class wraps RSA functionality to make it compatible with scikit-learn's estimator interface. This allows RSA to be used as the estimator parameter in SearchLight analysis, enabling local representational similarity analysis across the brain.

## Basic Usage

```python
from sekupy.analysis.rsa import RSAEstimator
from sekupy.analysis.searchlight import SearchLight
from sekupy.preprocessing import SampleSlicer, TargetTransformer
from sklearn.model_selection import StratifiedShuffleSplit

# Load and preprocess your dataset
# ds = your_dataset_loader()
ds = SampleSlicer(subject=['subj01'], evidence=[1, 2, 3]).transform(ds)
ds = TargetTransformer(attr='evidence').transform(ds)

# Create RSA estimator
rsa_estimator = RSAEstimator(metric='euclidean')

# Define a custom scorer for RSA
# This scorer calls the estimator's score() method which computes
# the negative mean distance (higher is better)
class RSAScorer:
    """Custom scorer for RSA that uses estimator.score()."""
    def __call__(self, estimator, X, y=None):
        return estimator.score(X, y)
    
    def __repr__(self):
        return "RSAScorer()"

# Create SearchLight analysis with RSA
analysis = SearchLight(
    estimator=rsa_estimator,
    radius=9.0,  # Radius in mm for searchlight sphere
    scoring={'rsa': RSAScorer()},  # Use custom RSA scorer
    cv=StratifiedShuffleSplit(n_splits=2, test_size=0.2),
    verbose=1,
    permutation=0
)

# Fit the searchlight with RSA
analysis.fit(ds)

# Access results
scores = analysis.scores[0]
print(f"RSA scores shape: {scores['test_rsa'].shape}")

# Save results if needed
# analysis.save(path='/path/to/save/results')
```

## Supported Distance Metrics

The `RSAEstimator` supports any distance metric from `scipy.spatial.distance.pdist`:

- `'euclidean'`: Euclidean distance (default)
- `'correlation'`: Pearson correlation distance
- `'cosine'`: Cosine distance
- `'cityblock'`: Manhattan distance
- `'hamming'`: Hamming distance
- And many more...

Example with correlation distance:

```python
rsa_estimator = RSAEstimator(metric='correlation')
```

## How It Works

1. **RSA Computation**: For each searchlight sphere, the RSA estimator computes pairwise distances between samples using the specified metric.

2. **Scoring**: The estimator returns the negative mean distance as a score. Higher scores indicate more similar representations (lower distances).

3. **Cross-Validation**: The SearchLight framework applies cross-validation, computing RSA scores for each fold and each voxel/feature location.

4. **Output**: The result is a map of RSA scores across the brain, showing which regions have structured representational similarity.

## Key Classes

- **`RSAEstimator`**: sklearn-compatible estimator for RSA
  - `fit(X, y)`: Computes distance matrix for training data
  - `score(X, y)`: Returns negative mean distance for test data
  - `predict(X)`: Returns condensed distance matrix
  - `transform(X)`: Returns condensed distance matrix

- **`RSAScorer`**: Custom scorer that calls `estimator.score()`
  - Required because standard sklearn scorers expect predict() output

## Testing

The implementation includes comprehensive tests:

```python
# Test basic RSAEstimator functionality
def test_rsa_estimator():
    X = np.random.randn(10, 5)
    estimator = RSAEstimator(metric='euclidean')
    estimator.fit(X)
    score = estimator.score(X)
    assert isinstance(score, (float, np.floating))

# Test RSA within SearchLight
def test_rsa_with_searchlight(fetch_ds):
    ds = fetch_ds
    ds = SampleSlicer(subject=['subj01'], evidence=[1, 2, 3]).transform(ds)
    ds = TargetTransformer(attr='evidence').transform(ds)
    
    rsa_estimator = RSAEstimator(metric='euclidean')
    
    analysis = SearchLight(
        estimator=rsa_estimator,
        radius=9.0,
        scoring={'rsa': RSAScorer()},
        cv=StratifiedShuffleSplit(n_splits=2, test_size=0.2),
        verbose=0
    )
    
    analysis.fit(ds)
    assert hasattr(analysis, 'scores')
```

## References

- Kriegeskorte, N., Goebel, R., & Bandettini, P. (2006). Information-based functional brain mapping. PNAS, 103(10), 3863-3868.
- Kriegeskorte, N., Mur, M., & Bandettini, P. (2008). Representational similarity analysis - connecting the branches of systems neuroscience. Frontiers in Systems Neuroscience, 2, 4.
