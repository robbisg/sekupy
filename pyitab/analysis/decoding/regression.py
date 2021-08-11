from pyitab.analysis.decoding.roi_decoding import RoiDecoding
from pyitab.analysis.decoding.temporal_decoding import TemporalDecoding
from sklearn.svm import SVR


class RoiRegression(RoiDecoding):

    def __init__(self, 
                 estimator=SVR(C=1, kernel='linear'),
                 n_jobs=1,
                 scoring=['r2'],
                 permutation=0, 
                 verbose=1,
                 name='roi_regression',
                 **kwargs
                 ):

        return RoiDecoding.__init__(self,
                                    estimator=estimator,
                                    n_jobs=n_jobs, 
                                    scoring=scoring, 
                                    permutation=permutation, 
                                    verbose=verbose,
                                    name=name,
                                    **kwargs
                                    )


class TemporalRegression(TemporalDecoding):

    def __init__(self, 
                 estimator=SVR(C=1, kernel='linear'), 
                 n_jobs=1, 
                 scoring='r2',
                 permutation=0, 
                 verbose=1,
                 **kwargs
                 ):

        return TemporalDecoding.__init__(self,
                                         estimator=estimator, 
                                         n_jobs=n_jobs, 
                                         scoring=scoring,
                                         permutation=permutation, 
                                         verbose=verbose,
                                         name='temporal_regression',
                                         **kwargs
                                         )