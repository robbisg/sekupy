from pyitab.tests import fetch_ds
from pyitab.preprocessing.functions import *

from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pytest


def test_detrender():
    pass

def test_sampleaverager(fetch_ds):
    ds = fetch_ds

    ds_ = SampleAverager(attributes=['subject', 'memory_status']).transform(ds)

    sample_ds = SampleSlicer(subject=['subj01'], memory_status=['F']).transform(ds)

    np.testing.assert_array_equal(sample_ds.samples.mean(0), ds_.samples[0])

    last_item = list(ds_.a.prepro[-1].keys())[0]
    assert last_item == 'sample_averager' 



def test_targettransformer(fetch_ds):
    ds = fetch_ds

    prev_target = ds.targets.copy()
    np.testing.assert_array_equal(ds.targets, prev_target)
    ds = TargetTransformer(attr='decision').transform(ds)
    
    with pytest.raises(Exception):
        np.testing.assert_array_equal(ds.targets, prev_target)

    last_item = list(ds.a.prepro[-1].keys())[0]
    assert last_item == 'target_transformer'


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

    mask = np.zeros_like(ds.samples[:, 0], dtype=np.bool)
    mask[np.arange(10)] = True

    ds = DatasetMasker(mask=mask).transform(ds)

    assert ds.shape[0] == 10

    last_item = list(ds.a.prepro[-1].keys())[0]
    assert last_item == 'dataset_masker' 


def test_featurestacker(fetch_ds):

    ds = fetch_ds

    assert ds.shape == (120, 843)

    ds = FeatureStacker(stack_attr=['evidence'], 
                         keep_attr=['memory_status'],
                         selection_dictionary={'subject':['subj03'], 'evidence':[1,2,3]}
                         ).transform(ds)

    assert ds.shape == (2, 12645)

    last_item = list(ds.a.prepro[-1].keys())[0]
    assert last_item == 'feature_stacker'    
