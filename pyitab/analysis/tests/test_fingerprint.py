from pyitab.preprocessing import TargetTransformer
from pyitab.preprocessing import SampleSlicer
from pyitab.analysis.fingerprint import BehaviouralFingerprint, \
    Identifiability, TaskPredictionTavor

from sklearn.model_selection import KFold

import numpy as np
import os
import pytest
from pyitab.tests import fetch_ds


def test_fingerprint(fetch_ds):

    ds = fetch_ds

    ds = SampleSlicer(subject=['subj01'], 
                      decision=['L', 'F']).transform(ds)

    ds = TargetTransformer(attr='evidence').transform(ds)

    np.testing.assert_array_equal(ds.targets, ds.sa.evidence)

    analysis = BehaviouralFingerprint(cv=KFold())

    analysis.fit(ds)

    scores = analysis._scores
    assert len(scores) == 2 # Positive vs negative
    
    roi_result = scores['positive']['mask-brain_value-2.0']
    assert len(roi_result) == 1
    assert roi_result[0]['test_r2'].shape == (5,)


def test_tavor(fetch_ds):

    ds = fetch_ds

    analysis = TaskPredictionTavor()

    y_attr = dict(
        time_indices=[4]
    )

    x_attr = dict(
        time_indices=[7]
    )

    analysis.fit(ds, y_attr=y_attr, x_attr=x_attr)

    scores = analysis.scores
    assert 1 == 1


def test_identifiability(fetch_ds):

    ds = fetch_ds

    ds = SampleSlicer(memory_status=['L', 'F']).transform(ds)

    analysis = Identifiability()

    analysis.fit(ds, attr='memory_status')

    scores = analysis.scores
    assert scores['matrix'].shape == (len(scores['vars']), len(scores['vars']))
    assert scores['accuracy'].shape == (len(scores['vars']), len(scores['vars']))