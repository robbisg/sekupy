from sekupy.preprocessing import SampleSlicer, TargetTransformer
from sekupy.analysis.searchlight import SearchLight
from sklearn.model_selection import StratifiedShuffleSplit

import numpy as np
import os
import pytest
from sekupy.tests import fetch_ds


def test_searchlight(fetch_ds):
    ds = fetch_ds
    ds = SampleSlicer(subject=['subj01'], evidence=[1,2,3]).transform(ds)
    ds = TargetTransformer(attr='evidence').transform(ds)

    np.testing.assert_array_equal(ds.targets, ds.sa.evidence)

    n_splits = 2
    n_permutation = 2
    scoring = ['r2', 'explained_variance']
    analysis = SearchLight(scoring=scoring, 
                           cv=StratifiedShuffleSplit(n_splits=n_splits, 
                                                     test_size=0.2), 
                           verbose=0,
                           permutation=n_permutation)

    analysis.fit(ds)

    scores = analysis.scores
    assert len(scores) == n_permutation + 1
    
    score = scores[0]
    for k in scoring:
        assert "test_%s" % (k) in score.keys()

    assert score['test_r2'].shape == (843, n_splits)

    scoring = 'r2'
    analysis = SearchLight(scoring=scoring, 
                           cv=StratifiedShuffleSplit(n_splits=n_splits, 
                                                     test_size=0.2), 
                           verbose=0,
                           permutation=1)

    analysis.fit(ds)
    score = scores[0]
    assert "test_r2" in score.keys()