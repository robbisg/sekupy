from pyitab.preprocessing.functions import SampleSlicer, TargetTransformer
from pyitab.analysis.decoding.temporal_decoding import TemporalDecoding
from pyitab.analysis.decoding.roi_decoding import RoiDecoding

from sklearn.model_selection import StratifiedShuffleSplit

import numpy as np
import os
import pytest
from pyitab.tests import fetch_ds


def test_clustering(fetch_ds):
    ds = fetch_ds
    ds = SampleSlicer(subject=['subj01']).transform(ds)
    ds = TargetTransformer(attr='trial_decoding').transform(ds)

    np.testing.assert_array_equal(ds.targets, ds.sa.trial_decoding)

    n_splits = 2
    n_permutation = 2

    analysis = TemporalDecoding(cv=StratifiedShuffleSplit(n_splits=n_splits, 
                                                          test_size=0.2), 
                                verbose=0,
                                permutation=n_permutation)

    analysis.fit(ds, time_attr='trial')

    scores = analysis.scores
    assert len(scores.keys()) == 26 # No. of ROI
    
    roi_result = scores['mask-brain_value-2.0']
    assert len(roi_result) == n_permutation + 1
    assert roi_result[0]['test_score'].shape == (n_splits, 3, 3)