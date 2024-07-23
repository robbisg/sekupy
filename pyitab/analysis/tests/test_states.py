from pyitab.preprocessing import SampleSlicer, TargetTransformer
from pyitab.analysis.decoding.temporal_decoding import TemporalDecoding

from sklearn.model_selection import StratifiedShuffleSplit
from pyitab.analysis.states.pipeline import StateAnalyzer
from pyitab.analysis.states.subsamplers import VarianceSubsampler
from pyitab.tests import fetch_ds

import numpy as np
import pytest



@pytest.mark.skip()
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


def test_state_analyzer(fetch_ds):

    ds = fetch_ds

    state_analyzer = StateAnalyzer()
    state_analyzer.fit(ds, n_clusters=range(2, 10), prepro=VarianceSubsampler())

    state_analyzer.score()
