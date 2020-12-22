from pyitab.preprocessing.functions import SampleSlicer, TargetTransformer
from pyitab.analysis.decoding.temporal_decoding import TemporalDecoding
from pyitab.analysis.decoding.cross_decoding import CrossDecoding

from sklearn.model_selection import StratifiedShuffleSplit

import numpy as np
import os
import pytest
from pyitab.tests import fetch_ds


def test_crossdecoding(fetch_ds):
    ds = fetch_ds
    ds = TargetTransformer(attr='trial_decoding').transform(ds)

    np.testing.assert_array_equal(ds.targets, ds.sa.trial_decoding)

    n_splits = 2
    n_permutation = 2

    analysis = CrossDecoding(cv=StratifiedShuffleSplit(n_splits=n_splits, 
                                                       test_size=0.2), 
                             verbose=0,
                             decoder=TemporalDecoding,
                             permutation=n_permutation)

    analysis.fit(ds, 
                 training_conditions={'subject':['subj01']}, 
                 testing_conditions={'subject':['subj02']}, 
                 time_attr='trial')

    scores = analysis.cross_scores
    assert len(scores.keys()) == 26 # No. of ROI
    
    roi_result = scores['mask-brain_value-2.0']
    assert len(roi_result) == n_permutation + 1
    assert roi_result[0]['test_score'].shape == (n_splits, 3, 3)

    


