from pyitab.preprocessing.base import PreprocessingPipeline, Transformer
from pyitab.analysis.base import Analyzer
from pyitab.preprocessing import SampleSlicer

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.metrics import mean_squared_error, r2_score

from scipy.io import savemat

import numpy as np
import os
import logging
logger = logging.getLogger(__name__)

class TaskPredictionTavor(Analyzer):

    def __init__(self, 
                 estimator=None,
                 n_jobs=1, 
                 scoring=['neg_mean_squared_error', 'r2'], 
                 permutation=0,
                 verbose=1,
                 name='tavor',
                 **kwargs):
        
        if estimator is None:
            estimator = Pipeline(steps=[('clf', LinearRegression())])

        if not isinstance(estimator, Pipeline):
            estimator = Pipeline(steps=[('clf', estimator)])

        self.estimator = estimator
        self.n_jobs = n_jobs
        self.permutation = permutation
        
        self.verbose = verbose

        if isinstance(scoring, str):
            scoring = [scoring]

        self.scoring = _check_multimetric_scoring(self.estimator, 
                                                  scoring=scoring)

        logger.debug(self.scoring)

        Analyzer.__init__(self, name=name, **kwargs)


    def _get_data(self, ds, subj, y_attr, x_attr, prepro):

        ds_ = SampleSlicer(subject=[subj]).transform(ds)

        ds_x = SampleSlicer(**x_attr).transform(ds_)
        ds_y = SampleSlicer(**y_attr).transform(ds_)

        if prepro is not None:
            ds_x = prepro.transform(ds_x)
            ds_y = prepro.transform(ds_y)

        X = ds_x.samples.T
        y = ds_y.samples.T

        return X, y


    def _fit(self, ds, y_attr, x_attr, prepro):
        betas = list()
        intercepts = list()

        self._subjects = np.unique(ds.sa.subject)


        for subj in self._subjects:

            X, y = self._get_data(ds, subj, y_attr, x_attr, prepro)
            
            _ = self.estimator.fit(X, y)
            linear = self.estimator.steps[0][1]

            betas.append(linear.coef_.squeeze())
            intercepts.append(linear.intercept_.squeeze())
        
        self._betas = np.array(betas)
        self._intercepts = np.array(intercepts)


    def _predict(self, ds, y_attr, x_attr, prepro):
        
        betas = self._betas
        intercepts = self._intercepts

        self._y = list()
        self._y_hat = list()

        for s, subj in enumerate(self._subjects):

            X, y = self._get_data(ds, subj, y_attr, x_attr, prepro)

            average_beta = np.delete(betas, s).mean()
            average_intercept = np.delete(intercepts, s).mean()

            logger.debug(average_beta, betas[s])

            y_hat = np.dot(X, average_beta) + average_intercept

            self._y.append(y.squeeze())
            self._y_hat.append(y_hat.squeeze())

        self._y = np.array(self._y)
        self._y_hat = np.array(self._y_hat)

    def _score(self):

        self.scores = dict()

        errors = {
            'mse': mean_squared_error,
            'r2': r2_score 
        }


        for k, l in errors.items():
            self.scores[k] = [l(self._y[i], self._y_hat[i]) for i in range(self._y.shape[0])]

        for k, l in self.scoring.items():
            self.scores[k] = [l._score_func(self._y[i], self._y_hat[i]) for i in range(self._y.shape[0])]

        self.scores['betas'] = self._betas


    def fit(self, ds, y_attr=dict(), x_attr=dict(), prepro=None):


        self._fit(ds, y_attr, x_attr, prepro)
        self._predict(ds, y_attr, x_attr, prepro)
        self._score()

        return

    
    def save(self, path=None, **kwargs):

        path, prefix = super().save(path, **kwargs)
        kwargs.update({'prefix': prefix})

        filename = self._get_filename(**kwargs)
        logger.info("Saving %s" % (filename))
        savemat(os.path.join(path, filename), self.scores)
