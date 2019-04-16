from pyitab.tests import fetch_ds
from pyitab.preprocessing.functions import SampleSlicer, TargetTransformer
from pyitab.preprocessing.balancing.base import Balancer
from pyitab.preprocessing.balancing.imbalancer import Imbalancer

from imblearn.over_sampling import RandomOverSampler
import numpy as np


def test_underbalancer(fetch_ds):
    ds = fetch_ds

    ds = SampleSlicer(subject=['subj01'], decision=['L', 'F']).transform(ds)
    ds = TargetTransformer(attr='decision').transform(ds)

    balancer = Balancer(attr='all')

    unique, counts = np.unique(ds.targets, return_counts=True)
    assert counts[0] != counts[1]

    ds = balancer.transform(ds)
    unique, counts = np.unique(ds.targets, return_counts=True)
    assert counts[0] == counts[1]

    last_item = list(ds.a.prepro[-1].keys())[0]
    assert last_item == 'under_balancer'


def test_imbalancer(fetch_ds):
    ds = fetch_ds

    ds = SampleSlicer(subject=['subj01'], memory_status=['L', 'F']).transform(ds)
    ds = TargetTransformer(attr='memory_status').transform(ds)

    ratio = 0.2
    imbalancer = Imbalancer(sampling_strategy=ratio)

    unique, counts = np.unique(ds.targets, return_counts=True)
    assert counts[0] - counts[1] == 0

    ds = imbalancer.transform(ds)
    unique, counts = np.unique(ds.targets, return_counts=True)
    assert counts[0] + counts[1] == np.ceil((1+ratio) * counts[1])

    last_item = list(ds.a.prepro[-1].keys())[0]
    assert last_item == 'imbalancer'


def test_overbalancer(fetch_ds):    
    
    ds = fetch_ds

    ds = SampleSlicer(subject=['subj01'], decision=['L', 'F']).transform(ds)
    ds = TargetTransformer(attr='decision').transform(ds)

    balancer = Balancer(attr='all', balancer=RandomOverSampler())

    unique, counts = np.unique(ds.targets, return_counts=True)
    assert counts[0] != counts[1]

    ds = balancer.transform(ds)
    unique, counts = np.unique(ds.targets, return_counts=True)
    assert counts[0] == counts[1]

    last_item = list(ds.a.prepro[-1].keys())[0]
    assert last_item == 'over_balancer'