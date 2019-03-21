import numpy as np
import os

from nilearn.image.resampling import coord_transform
from nilearn import masking
from nilearn.decoding.searchlight import search_light

from sklearn.metrics.scorer import _check_multimetric_scoring
from sklearn.svm import SVC
from sklearn.preprocessing.label import LabelEncoder
from sklearn.model_selection._split import LeaveOneGroupOut
from sklearn.pipeline import Pipeline

from pyitab.analysis.searchlight.utils import _get_affinity, check_proximity
from pyitab.analysis.searchlight.utils import load_proximity, save_proximity
from pyitab.analysis.searchlight import SearchLight, get_seeds
from pyitab.analysis.base import Analyzer

from mne.decoding import GeneralizingEstimator
from imblearn.under_sampling import RandomUnderSampler

from pyitab.utils.dataset import get_ds_data
from pyitab.utils.image import save_map
from pyitab.utils.files import make_dir
from pyitab.utils.dataset import temporal_attribute_reshaping, temporal_transformation

import logging
logger = logging.getLogger(__name__)



class TemporalSearchLight(SearchLight):
    """Implement search_light analysis using an arbitrary type of classifier.
    This is a wrapper of the nilearn algorithm to work with pymvpa dataset.

    Parameters
    -----------
    mask_img : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        boolean image giving location of voxels containing usable signals.

    process_mask_img : Niimg-like object, optional
        See http://nilearn.github.io/manipulating_images/input_output.html
        boolean image giving voxels on which searchlight should be
        computed.

    radius : float, optional
        radius of the searchlight ball, in millimeters. Defaults to 2.

    estimator : 'svr', 'svc', or an estimator object implementing 'fit'
        The object to use to fit the data

    n_jobs : int, optional. Default is -1.
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    scoring : string or callable, optional
        The scoring strategy to use. See the scikit-learn documentation
        If callable, takes as arguments the fitted estimator, the
        test data (X_test) and the test target (y_test) if y is
        not None.

    cv : cross-validation generator, optional
        A cross-validation generator. If None, a 3-fold cross
        validation is used or 3-fold stratified cross-validation
        when y is supplied.

    verbose : int, optional
        Verbosity level. Defaut is False
    """

    def __init__(self, 
                 radius=9.,
                 estimator=None,
                 n_jobs=1, 
                 scoring='accuracy', 
                 cv=LeaveOneGroupOut(), 
                 permutation=0,
                 verbose=1):


        super().__init__(radius=radius,
                         estimator=estimator,
                         n_jobs=n_jobs, 
                         scoring=scoring, 
                         cv=cv, 
                         permutation=permutation,
                         verbose=verbose)

        if estimator is None:
            estimator = Pipeline(steps=[('clf', SVC(C=1, kernel='linear'))])

        if not isinstance(estimator, Pipeline):
            estimator = Pipeline(steps=[('clf', estimator)])

        self.estimator = GeneralizingEstimator(estimator)



    def fit(self, ds, 
            time_attr='frame', cv_attr='chunks',
            balancer=RandomUnderSampler(return_indices=True)):
        """
        Fit the searchlight
        """
        
        A = get_seeds(ds, self.radius)
        
        estimator = self.estimator
        
        self._balancer = balancer

        X, y = get_ds_data(ds)
        t_values = ds.sa[time_attr].value

        X, y = temporal_transformation(X, y, t_values)

        _, _, indices = self._balancer.fit_sample(X[:,:,0], y)
        indices = np.sort(indices)

        groups = None
        if cv_attr is not None:
            _reshape = temporal_attribute_reshaping
            if isinstance(cv_attr, list):
                groups = np.vstack([_reshape(ds.sa[att].value, t_values) for att in cv_attr]).T
            else:
                groups = _reshape(ds.sa[cv_attr].value, t_values)
            groups = groups[indices]

        X, y = X[indices], y[indices]
        logger.info(np.unique(y, return_counts=True))

        indices = self._get_permutation_indices(len(y))
        values = []
        
        for idx in indices:
            y_ = y[idx] 

            scores = search_light(X, y_, estimator, A, groups=groups,
                                  cv=self.cv, n_jobs=self.n_jobs,
                                  verbose=self.verbose)
            
            values.append(scores)
        
        self.scores = values

        splits = self._split_name(X, y, self.cv, groups)

        self._info = self._store_ds_info(ds, cv_attr=cv_attr, test_order=splits)

        return self


    def _reshape_image(self, image):
        new_shape = (image.shape[0], -1)
        return np.reshape(image, new_shape)


    def save(self, path=None, save_cv=True, fx_image=lambda x: np.reshape(x, (x.shape[0], -1))):

        super().save(path=path,
                     save_cv=save_cv,
                     fx_image=fx_image)
