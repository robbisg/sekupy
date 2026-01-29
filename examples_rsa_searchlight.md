# RSA within SearchLight Example

This document demonstrates how to use RSA (Representational Similarity Analysis) as an estimator within SearchLight analysis.

## Overview

The `RSAEstimator` class wraps RSA functionality to make it compatible with scikit-learn's estimator interface. This allows RSA to be used as the estimator parameter in SearchLight analysis, enabling local representational similarity analysis across the brain.

**Important:** RSA analyzes representations of different experimental conditions, not individual trials. The estimator requires condition labels (y) and computes distances between condition-averaged representations.

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
# the negative mean distance between condition averages (higher is better)
class RSAScorer:
    """Custom scorer for RSA that uses estimator.score()."""
    def __call__(self, estimator, X, y):
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

## How RSA Works

RSA (Representational Similarity Analysis) analyzes how different experimental conditions are represented in the brain:

1. **Grouping by Condition**: For each experimental condition (e.g., viewing faces vs. houses), all samples/trials are grouped together.

2. **Averaging**: Within each condition, samples are averaged to get a single representative pattern for that condition.

3. **Distance Computation**: Pairwise distances are computed between these condition-averaged patterns, creating a representational dissimilarity matrix (RDM).

4. **Scoring**: The negative mean distance is returned as a score (higher values indicate more structured representations).

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

## How It Works with SearchLight

1. **Local Neighborhood**: For each searchlight sphere, a subset of voxels/features is selected.

2. **Condition Averaging**: Within that sphere, samples are grouped by condition and averaged.

3. **RSA Computation**: The RSA estimator computes pairwise distances between these condition averages.

4. **Cross-Validation**: The SearchLight framework applies cross-validation, computing RSA scores for each fold.

5. **Scoring**: The negative mean distance is used as the score (higher = more structured representations).

6. **Output**: The result is a brain map showing RSA scores across all searchlight locations.

## Key Classes

- **`RSAEstimator`**: sklearn-compatible estimator for RSA
  - `fit(X, y)`: Groups by y, averages within conditions, computes distance matrix
  - `score(X, y)`: Returns negative mean distance between condition averages
  - `predict(X, y)`: Returns condensed distance matrix between condition averages
  - `transform(X, y)`: Returns condensed distance matrix between condition averages

- **`RSAScorer`**: Custom scorer that calls `estimator.score(X, y)`
  - Required because RSA computes distances, not predictions
  - Must pass both X and y to compute condition-averaged distances

## Example: Understanding the Process

```python
import numpy as np
from sekupy.analysis.rsa import RSAEstimator

# Example data: 6 samples, 2 features, 3 conditions
X = np.array([[1, 2], [1.1, 2.1],      # Condition 0: 2 samples
              [5, 6], [5.1, 6.1],       # Condition 1: 2 samples  
              [9, 10], [9.1, 10.1]])    # Condition 2: 2 samples
y = np.array([0, 0, 1, 1, 2, 2])

# Fit RSA estimator
rsa = RSAEstimator(metric='euclidean')
rsa.fit(X, y)

# What happens internally:
# 1. Group by condition:
#    Condition 0: [[1, 2], [1.1, 2.1]]
#    Condition 1: [[5, 6], [5.1, 6.1]]
#    Condition 2: [[9, 10], [9.1, 10.1]]
#
# 2. Average within condition:
#    Condition 0 avg: [1.05, 2.05]
#    Condition 1 avg: [5.05, 6.05]
#    Condition 2 avg: [9.05, 10.05]
#
# 3. Compute pairwise distances:
#    dist(0, 1) ≈ 5.66
#    dist(0, 2) ≈ 11.31
#    dist(1, 2) ≈ 5.66

print(f"Condition averages shape: {rsa.condition_averages_.shape}")  # (3, 2)
print(f"Distance matrix shape: {rsa.distance_matrix_.shape}")  # (3,) - 3 pairwise distances
```

## Testing

The implementation includes comprehensive tests:

```python
# Test basic RSAEstimator functionality
def test_rsa_estimator():
    X = np.random.randn(10, 5)
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])  # Condition labels required
    estimator = RSAEstimator(metric='euclidean')
    estimator.fit(X, y)
    score = estimator.score(X, y)
    assert isinstance(score, (float, np.floating))

# Test RSA within SearchLight
def test_rsa_with_searchlight(fetch_ds):
    ds = fetch_ds
    ds = SampleSlicer(subject=['subj01'], evidence=[1, 2, 3]).transform(ds)
    ds = TargetTransformer(attr='evidence').transform(ds)
    
    rsa_estimator = RSAEstimator(metric='euclidean')
    
    class RSAScorer:
        def __call__(self, estimator, X, y):  # y is required
            return estimator.score(X, y)
    
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
