import numpy as np

from sklearn.metrics.scorer import _check_multimetric_scoring
from sklearn.svm import SVC
from sklearn.preprocessing.label import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection._split import LeaveOneGroupOut
from sklearn.model_selection._validation import cross_validate

from tqdm import tqdm

from pyitab.io.utils import get_ds_data

from pyitab.preprocessing.functions import FeatureSlicer
from pyitab.analysis.decoding import Decoding
from pyitab.preprocessing.functions import Transformer

from scipy.io.matlab.mio import savemat

import logging
logger = logging.getLogger(__name__)


# TODO: Inherit from MetaDecoding
class RoiDecoding(Decoding):
    """Implement decoding analysis using an arbitrary type of classifier.

    Parameters
    -----------

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
        
    permutation : int. Default is 0.
        The number of permutation to be performed.
        If the number is 0, no permutation is performed.

    cv : cross-validation generator, optional
        A cross-validation generator. If None, a 3-fold cross
        validation is used or 3-fold stratified cross-validation
        when y is supplied.

    verbose : int, optional
        Verbosity level. Defaut is False
        
    
    Attributes
    -----------

    scores : dict.
            The dictionary of results for each roi selected.
            The key is the union of the name of the roi and the value(s).
            The value is a list of values, the number is equal to the permutations.
            
    """

    def __init__(self, 
                 estimator=None,
                 n_jobs=1, 
                 scoring='accuracy', 
                 cv=LeaveOneGroupOut(),
                 permutation=0,
                 verbose=1):


        Decoding.__init__(self,
                          estimator=estimator,
                          n_jobs=n_jobs,
                          scoring=scoring,
                          cv=cv,
                          permutation=permutation,
                          verbose=verbose,
                          name='roi_decoding')


    def _get_rois(self, ds, roi):
        """Gets the roi list if the attribute is all"""
        
        rois = [r for r in ds.fa.keys() if r != 'voxel_indices']
        
        if roi != 'all':
            rois = roi
        
        rois_values = []
        
        for r in rois:
            value = [(r, [v]) for v in np.unique(ds.fa[r].value) if v != 0]
            rois_values.append(value)
            
        return list(*rois_values)    
    
    
    def _get_permutation_indices(self, n_samples, groups):
        """Permutes the indices of the dataset"""
        
        # TODO: Permute labels based on cv_attr
        from numpy.random.mtrand import permutation
        
        if self.permutation == 0:
            return [range(n_samples)]
        
        
        # reset random state
        indices = [range(n_samples)]
        for _ in range(self.permutation):
            idx = permutation(indices[0])
            indices.append(idx)
        
        return indices

    def fit(self, ds, cv_attr='chunks', roi='all', roi_values=None, prepro=Transformer(), **kwargs):
        
        return Decoding.fit(self, ds, 
                           cv_attr=cv_attr, 
                           roi=roi, 
                           roi_values=roi_values, 
                           prepro=prepro, 
                           **kwargs)
        
    
    def _fit(self, ds, cv_attr=None, return_predictions=False, return_splits=True, **kwargs):
        """General method to fit data"""


        self.scoring, _ = _check_multimetric_scoring(self.estimator, scoring=self.scoring)
        
        X, y = get_ds_data(ds)
        y = LabelEncoder().fit_transform(y)

        groups = None
        if cv_attr is not None:
            if isinstance(cv_attr, list):
                groups = np.vstack([ds.sa[att].value for att in cv_attr]).T
            else:
                groups = ds.sa[cv_attr].value

        indices = self._get_permutation_indices(len(y), groups)
                
        values = []

        for idx in tqdm(indices):
            
            y_ = y[idx]

            scores = cross_validate(self.estimator, X, y_, groups,
                                    self.scoring, self.cv, self.n_jobs,
                                    self.verbose, return_estimator=True, 
                                    return_splits=return_splits, 
                                    return_predictions=return_predictions)
            
            values.append(scores)
            if cv_attr is not None:
                scores['split_name'] = self._split_name(scores['splits'], 
                                                        cv_attr,
                                                        groups)
       
        return values
