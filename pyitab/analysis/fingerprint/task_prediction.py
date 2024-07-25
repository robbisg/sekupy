from pyitab.preprocessing.base import PreprocessingPipeline, Transformer
from pyitab.analysis.base import Analyzer
from pyitab.preprocessing import SampleSlicer
from pyitab.analysis.utils import get_params
from pyitab.utils.scores import correlation

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle

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
            estimator = Pipeline(steps=[('clf', LinearRegression())])

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


    def _get_data(self, ds, subj, y_attr, x_attr, prepro, ):

        ds_ = SampleSlicer(subject=[subj]).transform(ds)

        ds_x = SampleSlicer(**x_attr).transform(ds_)
        ds_y = SampleSlicer(**y_attr).transform(ds_)

        if prepro is not None:
            ds_x = prepro.transform(ds_x)
            ds_y = prepro.transform(ds_y)

        X = ds_x.samples.T
        y = ds_y.samples.T

        return X, y


    def _fit(self, ds, y_attr, x_attr, prepro, perm_id):
        betas = list()
        intercepts = list()

        for subj in self._subjects:

            X, y = self._get_data(ds, subj, y_attr, x_attr, prepro)

            if perm_id != 0:
                y = shuffle(y)

            _ = self.estimator.fit(X, y)
            linear = self.estimator.steps[0][1]

            betas.append(linear.coef_.squeeze())
            intercepts.append(linear.intercept_.squeeze())

        betas = np.array(betas)
        intercepts = np.array(intercepts)

        return betas, intercepts

    def _predict(self, ds, y_attr, x_attr, prepro, perm_id):

        betas = self._betas[perm_id]
        intercepts = self._intercepts[perm_id]

        y_true = list()
        y_pred = list()

        for s, subj in enumerate(self._subjects):

            X, y = self._get_data(ds, subj, y_attr, x_attr, prepro)

            average_beta = np.delete(betas, s).mean()
            average_intercept = np.delete(intercepts, s).mean()

            logger.debug(average_beta, betas[s])

            y_hat = np.dot(X, average_beta) + average_intercept

            y_true.append(y.squeeze())
            y_pred.append(y_hat.squeeze())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        return y_true, y_pred


    def _score(self, y_true, y_pred):

        scores = dict()

        errors = {
            'mse': mean_squared_error,
            'r2': r2_score,
            'corr': correlation
        }

        for k, l in errors.items():
            scores[k] = [l(y_true[i], y_pred[i]) for i in range(y_true.shape[0])]

        for k, l in self.scoring.items():
            scores[k] = [l._score_func(y_true[i], y_pred[i]) for i in range(y_true.shape[0])]

        return scores



    def fit(self, ds, y_attr=dict(), x_attr=dict(), prepro=None):

        self.scores = list()
        self._subjects = np.unique(ds.sa.subject)

        self._betas = np.zeros((self.permutation + 1 , len(self._subjects)))
        self._intercepts = np.zeros((self.permutation + 1 , len(self._subjects)))

        for p in range(self.permutation + 1):

            b, i = self._fit(ds, y_attr, x_attr, prepro, p)
            self._betas[p, :] = b
            self._intercepts[p, :] = i

            yt, yp = self._predict(ds, y_attr, x_attr, prepro, p)

            scores = self._score(yt, yp)

            scores['betas'] = b

            scores['y_true'] = yt
            scores['y_pred'] = yp

            self.scores.append(scores)


        self._info = self._store_info(ds, x_attr=x_attr, y_attr=y_attr)
        logger.debug(self._info)

        return

    
    def save(self, path=None, **kwargs):

        path, prefix = super().save(path, **kwargs)

        for p, score in enumerate(self.scores):
            kwargs.update({'prefix': prefix, 'perm': "%04d" % p})

            filename = self._get_filename(**kwargs)
            logger.info("Saving %s" % (filename))
            savemat(os.path.join(path, filename), score)


    def _get_filename(self, **kwargs):
        "target-<values>_id-<datetime>_mask-<mask>_value-<roi_value>_data.mat"
        logger.debug(kwargs)
        params = {}

        params_ = self._get_prepro_info(**kwargs)
        params.update(params_)
       
        logger.debug(params)

        # TODO: Solve empty prefix
        prefix = kwargs.pop('prefix')
        midpart = "_".join(["%s-%s" % (k, str(v).replace("_", "+")) \
             for k, v in params.items()])
        trailing = "perm-%s" % (kwargs.pop('perm'))
        filename = "%s_data.mat" % ("_".join([prefix, midpart, trailing]))

        return filename