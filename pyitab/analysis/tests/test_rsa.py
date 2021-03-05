from pyitab.tests import fetch_ds
from pyitab.preprocessing import SampleSlicer
from pyitab.preprocessing import TargetTransformer
from pyitab.analysis.rsa import RSA

import numpy as np

def test_rsa(fetch_ds):

    ds = fetch_ds

    ds = SampleSlicer(subject=['subj01'], 
                      decision=['L', 'F']).transform(ds)

    ds = TargetTransformer(attr='decision').transform(ds)

    n_samples = ds.shape[0]

    np.testing.assert_array_equal(ds.targets, ds.sa.decision)

    analysis = RSA()
    analysis.fit(ds)

    scores = analysis.scores
    assert len(scores.keys()) == 26 # No. of ROI
    
    roi_result = scores['mask-brain_value-2.0']
    assert roi_result.size == n_samples * (n_samples-1) * .5
