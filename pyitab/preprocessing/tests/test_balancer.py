from pyitab.tests import fetch_ds
from pyitab.preprocessing.functions import SampleSlicer, TargetTransformer
from pyitab.preprocessing.balancing.base import Balancer
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