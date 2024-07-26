from sekupy.tests import fetch_ds
from sekupy.preprocessing.slicers import *

from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pytest

def test_featureslicer(fetch_ds):
    ds = fetch_ds

    unique = np.unique(ds.fa.brain)
    assert len(unique) != 2

    ds = FeatureSlicer(brain=[2, 3]).transform(ds)
    unique = np.unique(ds.fa.brain)
    assert len(unique) == 2

    last_item = list(ds.a.prepro[-1].keys())[0]
    assert last_item == 'feature_slicer'


def test_sampleslicer(fetch_ds):
    ds = fetch_ds

    unique = np.unique(ds.sa.subject)
    assert len(unique) != 1

    ds = SampleSlicer(subject=['subj01']).transform(ds)
    unique = np.unique(ds.sa.subject)
    assert len(unique) == 1

    last_item = list(ds.a.prepro[-1].keys())[0]
    assert last_item == 'sample_slicer'
    

def test_datasetmasker(fetch_ds):

    ds = fetch_ds
    assert ds.shape == (120, 843)

    mask = np.zeros_like(ds.samples[:, 0], dtype=bool)
    mask[np.arange(10)] = True

    ds = DatasetMasker(mask=mask).transform(ds)

    assert ds.shape[0] == 10

    last_item = list(ds.a.prepro[-1].keys())[0]
    assert last_item == 'dataset_masker'


def test_fx_slicer(fetch_ds):

    ds = fetch_ds
    assert ds.shape == (120, 843)

    ds_1 = SampleExpressionSlicer(attr='age').transform(ds)
    assert ds_1.shape == (30, 843)

    value = np.mean(ds.sa.age) + 1.5*np.std(ds.sa.age)
    ds_2 = SampleExpressionSlicer(attr='age').transform(ds, value=value)

    np.testing.assert_array_equal(ds_1.samples, ds_2.samples)

    ds_3 = SampleSlicer(age=[32]).transform(ds)
    np.testing.assert_array_equal(ds_1.samples, ds_3.samples)