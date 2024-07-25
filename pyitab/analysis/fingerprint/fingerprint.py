
from pyitab.utils.math import dot_correlation
from pyitab.analysis.decoding.regression import RoiRegression
from pyitab.analysis.base import Analyzer
from pyitab.preprocessing import SampleSlicer
from pyitab.preprocessing.base import Transformer
from pyitab.ext.sklearn.feature_selection import positive_correlated, \
    negative_correlated

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFpr, f_regression
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVR

from scipy.io import savemat

import numpy as np
import os
import itertools

import logging
logger = logging.getLogger(__name__)

class Identifiability(Analyzer):

    def fit(self, ds, attr='targets'):

        unique = np.unique(ds.sa[attr].value)
        row, col = np.triu_indices(len(unique), k=0)
        
        identifiability_matrix = np.zeros((len(unique), len(unique)))
        accuracy_matrix = np.zeros((len(unique), len(unique)))

        task_combinations = itertools.combinations_with_replacement(unique, 2)
        correlation_matrix = dict()
        for j, (t1, t2) in enumerate(task_combinations):
            ds1 = SampleSlicer(**{attr: [t1]}).transform(ds)
            ds2 = SampleSlicer(**{attr: [t2]}).transform(ds)

            r = dot_correlation(ds1.samples, ds2.samples)
             
            i_self = np.mean(np.diag(r))

            id1 = np.triu_indices(r.shape[0], k=1)
            id2 = np.tril_indices(r.shape[0], k=-1)

            i_diff1 = np.mean(r[id1])
            i_diff2 = np.mean(r[id2])

            identifiability_matrix[row[j], col[j]] = i_self - i_diff1
            identifiability_matrix[col[j], row[j]] = i_self - i_diff2

            prediction1 = np.argmax(r, axis=0)
            prediction2 = np.argmax(r, axis=1)

            accuracy1 = np.count_nonzero(prediction1 == np.arange(r.shape[0])) / r.shape[0]
            accuracy2 = np.count_nonzero(prediction2 == np.arange(r.shape[0])) / r.shape[0]
            
            accuracy_matrix[row[j], col[j]] = accuracy1
            accuracy_matrix[col[j], row[j]] = accuracy2

            correlation_matrix[t1+'+'+t2] = r

        self.scores = dict()
        self.scores['matrix'] = identifiability_matrix  # idrate
        self.scores['vars'] = unique
        self.scores['accuracy'] = accuracy_matrix  # pscore
        self.scores['r'] = correlation_matrix

        self._info = self._store_info(ds, attr=attr)

        return

    def save(self, path=None, scores=None, **kwargs):

        self.name = 'identifiability'
        path, prefix = super().save(path, **kwargs)
        kwargs.update({'prefix': prefix})

        filename = self._get_filename(**kwargs)
        logger.info("Saving %s" % (filename))
        
        if scores is None:
            scores = self.scores

        savemat(os.path.join(path, filename), scores)


class BehaviouralFingerprint(RoiRegression):
    """This analysis is based on the paper  `Shen et al. 2017, Nature Protocol 
    <http://dx.doi.org/10.1038/nprot.2016.178>`_

    The pipeline is used to predict individual behaviour from brain connectivity.

    """

    def __init__(self, 
                 estimator=None,
                 n_jobs=1,
                 scoring=['r2'],
                 permutation=0, 
                 verbose=1,
                 **kwargs
                 ):

        def fx_transformation(X):
            X_ = np.sum(X, axis=1, keepdims=True)
            return X_

        if estimator is None:
            
            estimator = Pipeline([
                ('fsel', SelectFpr(f_regression)),
                ('trans', FunctionTransformer(fx_transformation)),
                ('clf', SVR())
            ])
        elif not isinstance(estimator, Pipeline):

            estimator = Pipeline([
                ('fsel', SelectFpr(f_regression)),
                ('trans', FunctionTransformer(fx_transformation)),
                ('clf', estimator)
            ])


        return RoiRegression.__init__(self,
                                      estimator=estimator,
                                      n_jobs=n_jobs, 
                                      scoring=scoring, 
                                      permutation=permutation, 
                                      verbose=verbose,
                                      name='fingerprint_shen',
                                      **kwargs
                                      )

    def fit(self, ds, 
            cv_attr='chunks', 
            roi='all', 
            roi_values=None, 
            prepro=Transformer(),
            return_predictions=False,
            return_splits=True,
            return_decisions=False,
            separate_posneg=True,
            **kwargs):

        if not separate_posneg:
            return super().fit(ds, cv_attr, roi, roi_values, prepro, 
                               return_predictions, return_splits, 
                               return_decisions, **kwargs)
        
        strategies = {
            'positive': positive_correlated,
            'negative': negative_correlated
        }

        self._scores = dict()
        for i, (s, selection) in enumerate(strategies.items()):
            self.estimator.steps[0] = ('fsel', SelectFpr(selection, alpha=1))
            super().fit(ds, cv_attr, roi, roi_values, prepro, 
                        return_predictions, return_splits, 
                        return_decisions, **kwargs)
            self._scores[s] = self.scores.copy()


    def save(self, path=None, **kwargs):

        for feature, score in self._scores.items():
            self.scores = score
            super().save(path, feature=feature, **kwargs)