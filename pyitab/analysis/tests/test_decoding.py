from pyitab.tests import BaseTest

from pyitab.preprocessing.functions import SampleSlicer, TargetTransformer
from pyitab.analysis.decoding.temporal_decoding import TemporalDecoding
from pyitab.analysis.decoding.roi_decoding import RoiDecoding

from sklearn.model_selection import StratifiedShuffleSplit

import numpy as np
import os
import unittest


class TestDecoding(BaseTest):

    def setUp(self):
        BaseTest.setUp(self)


    def test_temporal_decoding(self):

        self.ds = SampleSlicer(subject=['subj01']).transform(self.ds)
        self.ds = TargetTransformer(attr='trial_decoding').transform(self.ds)
        ds = self.ds

        np.testing.assert_array_equal(ds.targets, ds.sa.trial_decoding)

        ds = self.ds
        n_splits = 2
        n_permutation = 2

        analysis = TemporalDecoding(cv=StratifiedShuffleSplit(n_splits=n_splits, 
                                                              test_size=0.2), 
                                    verbose=0,
                                    permutation=n_permutation)
        analysis.fit(ds, time_attr='trial')

        scores = analysis.scores
        assert len(scores.keys()) == 26 # No. of ROI
        
        roi_result = scores['brain_2.0']
        assert len(roi_result) == n_permutation + 1
        assert roi_result[0]['test_score'].shape == (n_splits, 3, 3)

    
    def test_decoding(self):

        self.ds = SampleSlicer(subject=['subj01'], 
                               decision=['L', 'F']).transform(self.ds)

        self.ds = TargetTransformer(attr='decision').transform(self.ds)
        ds = self.ds

        np.testing.assert_array_equal(ds.targets, ds.sa.decision)

        ds = self.ds
        n_splits = 2
        n_permutation = 2

        analysis = RoiDecoding(cv=StratifiedShuffleSplit(n_splits=n_splits, 
                                                         test_size=0.2), 
                               verbose=0,
                               permutation=n_permutation)

        analysis.fit(ds, cv_attr='chunks')

        scores = analysis.scores
        assert len(scores.keys()) == 26 # No. of ROI
        
        roi_result = scores['brain_2.0']
        assert len(roi_result) == n_permutation + 1
        assert roi_result[0]['test_score'].shape == (n_splits,)



if __name__ == '__main__':
    unittest.main()