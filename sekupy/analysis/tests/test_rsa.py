from sekupy.tests import fetch_ds
from sekupy.preprocessing import SampleSlicer
from sekupy.preprocessing import TargetTransformer
from sekupy.analysis.rsa import RSA, RSAEstimator, rsa_scorer
from sekupy.analysis.searchlight import SearchLight
from sklearn.model_selection import StratifiedShuffleSplit

import numpy as np
import os
import tempfile
import shutil
import pytest

def test_rsa(fetch_ds):

    ds = fetch_ds

    ds = SampleSlicer(subject=['subj01'], 
                      decision=['L', 'F']).transform(ds)

    ds = TargetTransformer(attr='decision').transform(ds)

    n_samples = ds.shape[0]

    np.testing.assert_array_equal(ds.targets, ds.sa.decision)

    analysis = RSA()
    analysis.fit(ds)

    scores = analysis.scores
    assert len(scores.keys()) == 26 # No. of ROI
    
    roi_result = scores['mask-brain_value-2.0']
    assert roi_result.size == n_samples * (n_samples-1) * .5



@pytest.mark.skip()
def test_rsa_save(fetch_ds):
    ds = fetch_ds

    # Preprocessing steps as before
    ds = SampleSlicer(subject=['subj01'], decision=['L', 'F']).transform(ds)
    ds = TargetTransformer(attr='decision').transform(ds)

    analysis = RSA()
    analysis.fit(ds)

    # Use a temporary directory to save the results
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Assuming RSA class has a method save() which uses this path
        analysis.save(path=tmpdirname)

        # Generate the expected file name(s)
        for roi, scores in analysis.scores.items():
            mask_name = roi.split('_value-')[0].replace('mask-', '')
            value_name = roi.split('_value-')[1]
            filename = analysis._get_filename(prefix='test', mask=mask_name, roi_value=value_name)
            expected_path = os.path.join(tmpdirname, filename)

            # Check if file exists
            assert os.path.exists(expected_path), f"File {expected_path} does not exist"

            # Optionally: Check contents of the file
            from scipy.io import loadmat
            data = loadmat(expected_path)
            np.testing.assert_array_almost_equal(data['test_score'], scores, decimal=5)
            assert (data['conditions'] == ds.targets).all(), "Conditions do not match the targets"

    # Cleanup if not using with statement
    shutil.rmtree(tmpdirname, ignore_errors=True)


def test_rsa_estimator():
    """Test RSAEstimator basic functionality."""
    # Create simple test data
    X = np.random.randn(10, 5)
    
    # Create and fit RSA estimator
    estimator = RSAEstimator(metric='euclidean')
    estimator.fit(X)
    
    # Check that distance matrix was computed
    assert hasattr(estimator, 'distance_matrix_')
    expected_size = 10 * 9 / 2  # n * (n-1) / 2 for condensed distance matrix
    assert estimator.distance_matrix_.shape[0] == expected_size
    
    # Test score method
    score = estimator.score(X)
    assert isinstance(score, (float, np.floating))
    assert score < 0  # negative mean distance
    
    # Test transform method
    distances = estimator.transform(X)
    assert distances.shape[0] == expected_size


def test_rsa_with_searchlight(fetch_ds):
    """Test RSA as an estimator within SearchLight."""
    ds = fetch_ds
    
    # Preprocess data
    ds = SampleSlicer(subject=['subj01'], evidence=[1, 2, 3]).transform(ds)
    ds = TargetTransformer(attr='evidence').transform(ds)
    
    np.testing.assert_array_equal(ds.targets, ds.sa.evidence)
    
    # Create RSA estimator
    rsa_estimator = RSAEstimator(metric='euclidean')
    
    # For RSA within SearchLight, we can use a custom callable scorer
    # that directly calls the estimator's score method
    class RSAScorer:
        """Custom scorer for RSA that uses estimator.score()."""
        def __call__(self, estimator, X, y=None):
            return estimator.score(X, y)
        
        def __repr__(self):
            return "RSAScorer()"
    
    n_splits = 2
    analysis = SearchLight(
        estimator=rsa_estimator,
        radius=9.0,
        scoring={'rsa': RSAScorer()},  # Use custom scorer object
        cv=StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2),
        verbose=0,
        permutation=0
    )
    
    # Fit the searchlight with RSA
    analysis.fit(ds)
    
    # Check that analysis completed successfully
    assert hasattr(analysis, 'scores')
    assert len(analysis.scores) == 1  # No permutations, so just 1 result
    
    # Check the structure of results
    scores = analysis.scores[0]
    assert isinstance(scores, dict)
    
    # The scores should contain the test scores
    for key in scores.keys():
        assert 'test_' in key
        # Check shape: (n_voxels, n_splits)
        assert scores[key].shape[0] == ds.shape[1]  # n_voxels/features
        assert scores[key].shape[1] == n_splits
