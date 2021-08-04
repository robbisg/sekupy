from pyitab.tests import fetch_ds
from pyitab.preprocessing.regression import SampleResidualTransformer, \
    FeatureResidualTransformer
import numpy as np
import pytest


def test_sampleresidual(fetch_ds):

    ds = fetch_ds
    assert ds.shape == (120, 843)

    dsr = SampleResidualTransformer(attr=['age', 'evidence']).transform(ds)

    last_item = list(dsr.a.prepro[-1].keys())[0]
    assert last_item == 'sample_residual'

    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_equal(ds.samples, dsr.samples)


def test_featureresidual(fetch_ds):

    ds = fetch_ds
    assert ds.shape == (120, 843)

    dsr = FeatureResidualTransformer(attr=['brain']).transform(ds)

    last_item = list(dsr.a.prepro[-1].keys())[0]
    assert last_item == 'feature_residual'

    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_equal(ds.samples, dsr.samples)