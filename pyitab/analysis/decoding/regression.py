from pyitab.analysis.decoding.roi_decoding import RoiDecoding
from pyitab.analysis.decoding.temporal_decoding import TemporalDecoding
from sklearn.svm import SVR


class RoiRegression(RoiDecoding):

    def __init__(self, 
                 estimator=SVR(C=1, kernel='linear'),
                 n_jobs=1,
                 scoring=['r2'],
                 permutation=0, 
                 verbose=1):

        return RoiDecoding.__init__(estimator=estimator,
                                    n_jobs=n_jobs, 
                                    scoring=scoring, 
                                    permutation=permutation, 
                                    verbose=verbose)


class TemporalRegression(TemporalDecoding):

    def __init__(self, 
                 estimator=SVR(C=1, kernel='linear'), 
                 n_jobs=1, 
                 scoring='r2',
                 permutation=0, 
                 verbose=1):

        return TemporalDecoding.__init__(estimator=estimator, 
                                         n_jobs=n_jobs, 
                                         scoring=scoring,
                                         permutation=permutation, 
                                         verbose=verbose)