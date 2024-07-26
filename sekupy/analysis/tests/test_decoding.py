from sekupy.preprocessing import TargetTransformer
from sekupy.preprocessing import SampleSlicer
from sekupy.analysis.decoding.temporal_decoding import TemporalDecoding
from sekupy.analysis.decoding.roi_decoding import RoiDecoding

from sklearn.model_selection import StratifiedShuffleSplit

import numpy as np
import os
import pytest
from sekupy.tests import fetch_ds


def test_temporal_decoding(fetch_ds):
    ds = fetch_ds
    ds = SampleSlicer(subject=['subj01']).transform(ds)
    ds = TargetTransformer(attr='trial_decoding').transform(ds)

    np.testing.assert_array_equal(ds.targets, ds.sa.trial_decoding)

    n_splits = 2
    n_permutation = 2

    analysis = TemporalDecoding(cv=StratifiedShuffleSplit(n_splits=n_splits, 
                                                          test_size=0.2), 
                                verbose=0,
                                scoring='accuracy',
                                permutation=n_permutation)

    analysis.fit(ds, time_attr='trial')

    scores = analysis.scores
    assert len(scores.keys()) == 26 # No. of ROI
    
    roi_result = scores['mask-brain_value-2.0']
    assert len(roi_result) == n_permutation + 1
    
    test_results = np.array(roi_result[0]['test_score'])
    assert test_results.shape == (n_splits, 3, 3)
    assert np.max(roi_result[0]['test_score']) <= 1.
   

def test_decoding(fetch_ds):

    ds = fetch_ds

    ds = SampleSlicer(subject=['subj01'], 
                      decision=['L', 'F']).transform(ds)

    ds = TargetTransformer(attr='decision').transform(ds)

    np.testing.assert_array_equal(ds.targets, ds.sa.decision)

    n_splits = 2
    n_permutation = 2

    analysis = RoiDecoding(cv=StratifiedShuffleSplit(n_splits=n_splits, 
                                                     test_size=0.2), 
                           verbose=0,
                           permutation=n_permutation)

    analysis.fit(ds, cv_attr='chunks')

    scores = analysis.scores
    assert len(scores.keys()) == 26 # No. of ROI
    
    roi_result = scores['mask-brain_value-2.0']
    assert len(roi_result) == n_permutation + 1
    assert roi_result[0]['test_accuracy'].shape == (n_splits,)

