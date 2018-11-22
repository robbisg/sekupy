import numpy as np

from sklearn.metrics.scorer import check_scoring
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

from mne.decoding import GeneralizingEstimator
from imblearn.under_sampling import RandomUnderSampler
import logging
logger = logging.getLogger(__name__)

# TODO: Inherit from metadecoding
class TemporalDecoding(Decoding):
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
                 verbose=1):
        
        Decoding.__init__(self,
                          estimator=estimator,
                          n_jobs=n_jobs,
                          scoring=scoring,
                          cv=cv,
                          permutation=permutation,
                          verbose=verbose,
                          name='temporal_decoding')
        
        if estimator is None:
            estimator = Pipeline(steps=[('clf', SVC(C=1, kernel='linear'))])

        if not isinstance(estimator, Pipeline):
            estimator = Pipeline(steps=[('clf', estimator)])

        self.estimator = GeneralizingEstimator(estimator)   
    
    
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


    def _transform_data(self, X, y, time_attr):
        times = np.unique(time_attr)

        X_ = X.reshape(-1, len(times), X.shape[1])
        X_ = np.rollaxis(X_, 1, 3)

        y_ = self._reshape_attributes(y, time_attr)

        return X_, y_


    def _reshape_attributes(self, attribute_list, time_attribute):
        times = np.unique(time_attribute)

        y = attribute_list.reshape(-1, len(times))
        labels = []
        for yy in y:
            l, c = np.unique(yy, return_counts=True)
            labels.append(l[np.argmax(c)])

        return np.array(labels)

  
    def _fit(self, ds, 
             time_attr='frame', 
             cv_attr=None, 
             balancer=RandomUnderSampler(return_indices=True),
             return_splits=True,
             return_predictions=False,
             **kwargs):
        """General method to fit data"""
        
        self._balancer = balancer

        X, y = get_ds_data(ds)
        t_values = ds.sa[time_attr].value

        X, y = self._transform_data(X, y, t_values)

        _, _, indices = self._balancer.fit_sample(X[:,:,0], y)
        indices = np.sort(indices)

        groups = None
        if cv_attr is not None:
            _reshape = self._reshape_attributes
            if isinstance(cv_attr, list):
                groups = np.vstack([_reshape(ds.sa[att].value, t_values) for att in cv_attr]).T
            else:
                groups = _reshape(ds.sa[cv_attr].value, t_values)
            groups = groups[indices]

        X, y = X[indices], y[indices]
        logger.info(np.unique(y, return_counts=True))

        indices = self._get_permutation_indices(len(y), groups)
        values = []
                
        for idx in tqdm(indices):
            
            y_ = y[idx]

            scores = cross_validate(self.estimator, X, y_, 
                                    groups=groups,
                                    #scoring={'score' : self.scoring}, 
                                    cv=self.cv, 
                                    n_jobs=self.n_jobs,
                                    verbose=self.verbose, 
                                    return_estimator=True, 
                                    return_splits=return_splits,
                                    return_predictions=return_predictions)
            
            values.append(scores)

        return values

  
    def _save_score(self, score):
         
        mat_file = dict()
        
        for key, value in score.items():
            
            if key.find("test_") != -1:
                mat_file[key] = value
            
            #elif key == 'estimator':
            #    mat_estimator = self._save_estimator(value)
            #    mat_file.update(mat_estimator)
        
            elif key == "splits":
                mat_splits = self._save_splits(value)
                mat_file.update(mat_splits)

            elif key == "split_name":
                mat_file['split_name'] = [s['test'] for s in value]
            
        
        return mat_file

