import numpy as np

from sklearn.metrics.scorer import check_scoring
from sklearn.svm import SVC
from sklearn.preprocessing.label import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection._split import LeaveOneGroupOut

from tqdm import tqdm

from pyitab.utils.dataset import get_ds_data
from pyitab.utils.dataset import temporal_attribute_reshaping, \
    temporal_transformation

from pyitab.preprocessing.functions import FeatureSlicer
from pyitab.analysis.decoding.roi_decoding import RoiDecoding
from pyitab.preprocessing.base import Transformer

from scipy.io.matlab.mio import savemat

from mne.decoding import GeneralizingEstimator
from imblearn.under_sampling import RandomUnderSampler
import logging
logger = logging.getLogger(__name__)


class TemporalDecoding(RoiDecoding):
    """Implement temporal generalization decoding analysis 
        using an arbitrary type of classifier.

        see King 2014 TICS

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
                 verbose=1,
                 **kwargs
                 ):
        
        RoiDecoding.__init__(self,
                             estimator=estimator,
                             n_jobs=n_jobs,
                             scoring=scoring,
                             cv=cv,
                             permutation=permutation,
                             verbose=verbose,
                             name='temporal_decoding',
                             **kwargs
                             )
        
        if estimator is None:
            estimator = Pipeline(steps=[('clf', SVC(C=1, kernel='linear'))])

        if not isinstance(estimator, Pipeline):
            estimator = Pipeline(steps=[('clf', estimator)])

        self.estimator = GeneralizingEstimator(estimator)
        self.scoring = None 
    

    def _get_data(self, ds, cv_attr, 
                  time_attr='frame',
                  balancer=RandomUnderSampler(),
                  **kwargs):
        
        import warnings
        warnings.warn("This function must be replaced by super function _get_data", 
                        DeprecationWarning)

        X, y = get_ds_data(ds)

        if len(X.shape) == 3:
            return RoiDecoding._get_data(self, ds, cv_attr, **kwargs)

        t_values = ds.sa[time_attr].value
        X, y = temporal_transformation(X, y, t_values)

        _ = balancer.fit_sample(X[:,:,0], y)
        indices = balancer.sample_indices_
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

        return X, y, groups
    

  
    def fit(self, ds, 
             time_attr='frame',
             roi='all',
             roi_values=None,
             cv_attr=None,
             prepro=Transformer(),
             balancer=RandomUnderSampler(),
             return_splits=True,
             return_predictions=False,
             **kwargs):

        """General method to fit data"""
        
        super().fit(ds, 
                    cv_attr=cv_attr,
                    roi=roi, 
                    roi_values=roi_values, 
                    prepro=prepro,
                    return_predictions=return_predictions,
                    return_splits=return_splits,
                    time_attr=time_attr,
                    balancer=balancer,
                    **kwargs)

  
    def _save_score(self, score, save_estimator=False):
         
        mat_file = dict()
        
        for key, value in score.items():
            
            if key.find("test_") != -1:
                mat_file[key] = value
            
            elif key == 'estimator':
                mat_estimator = self._save_estimator(value, save_estimator)
                mat_file.update(mat_estimator)
        
            elif key == "splits":
                mat_splits = self._save_splits(value)
                mat_file.update(mat_splits)

            elif key == "split_name":
                mat_file['split_name'] = [s['test'] for s in value]
            
        
        return mat_file



    def _save_estimator(self, estimators, save_estimator):

        from joblib import dump
        
        mat_ = dict()
        mat_['weights'] = []
        mat_['features'] = []
        
        # For each fold
        for estimator in estimators:
            est_weights = []
            est_features = []

            estimators_ = estimator.estimators_
            # For each timepoint
            for est in estimators_:
                if hasattr(est.named_steps['clf'], 'coef_'): 
                    w = est.named_steps['clf'].coef_
                    est_weights.append(w)
                
                if 'fsel' in est.named_steps.keys():
                    f = est.named_steps['fsel'].get_support()
                    
                    est_features.append(f)

            mat_['features'].append(est_features)
            mat_['weights'].append(est_weights)

        mat_['features'] = np.array(mat_['features'])
        mat_['weights'] = np.array(mat_['weights'])

        return mat_