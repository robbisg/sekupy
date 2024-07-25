from sekupy.tests import fetch_ds
from sekupy.preprocessing import SampleSlicer
from sekupy.preprocessing import TargetTransformer
from sekupy.analysis.rsa import RSA

import numpy as np
import os
import tempfile
import shutil
import pytest

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



@pytest.mark.skip()
def test_rsa_save(fetch_ds):
    ds = fetch_ds

    # Preprocessing steps as before
    ds = SampleSlicer(subject=['subj01'], decision=['L', 'F']).transform(ds)
    ds = TargetTransformer(attr='decision').transform(ds)

    analysis = RSA()
    analysis.fit(ds)

    # Use a temporary directory to save the results
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Assuming RSA class has a method save() which uses this path
        analysis.save(path=tmpdirname)

        # Generate the expected file name(s)
        for roi, scores in analysis.scores.items():
            mask_name = roi.split('_value-')[0].replace('mask-', '')
            value_name = roi.split('_value-')[1]
            filename = analysis._get_filename(prefix='test', mask=mask_name, roi_value=value_name)
            expected_path = os.path.join(tmpdirname, filename)

            # Check if file exists
            assert os.path.exists(expected_path), f"File {expected_path} does not exist"

            # Optionally: Check contents of the file
            from scipy.io import loadmat
            data = loadmat(expected_path)
            np.testing.assert_array_almost_equal(data['test_score'], scores, decimal=5)
            assert (data['conditions'] == ds.targets).all(), "Conditions do not match the targets"

    # Cleanup if not using with statement
    shutil.rmtree(tmpdirname, ignore_errors=True)

    
    

