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
from pyitab.analysis.base import Analyzer
from pyitab.preprocessing.functions import Transformer

from scipy.io.matlab.mio import savemat

import logging
logger = logging.getLogger(__name__)


class Decoding(Analyzer):
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
                 verbose=1,
                 name="abstact_decoding"):
        
        if estimator is None:
            estimator = Pipeline(steps=[('clf', SVC(C=1, kernel='linear'))])

        if not isinstance(estimator, Pipeline):
            estimator = Pipeline(steps=[('clf', estimator)])

        self.estimator = estimator
        self.n_jobs = n_jobs
        self.permutation = permutation
        self.scoring = scoring
        self.cv = cv
        self.verbose = verbose
        
        Analyzer.__init__(self, name=name)
        

    # TODO: Must include kwargs
    def fit(self, ds, 
            cv_attr='chunks', roi='all', 
            roi_values=None, 
            prepro=Transformer(),
            return_predictions=False,
            return_splits=True,
            **kwargs):


        if roi_values is None:
            roi_values = self._get_rois(ds, roi)
                
        self.scores = dict()
        
        # TODO: How to use multiple ROIs
        for r, value in roi_values:
            
            ds_ = FeatureSlicer(**{r:value}).transform(ds)
            ds_ = prepro.transform(ds_)
            
            logger.info("Dataset shape %s" % (str(ds_.shape)))
            summary_cv = cv_attr
            if isinstance(cv_attr, list):
                summary_cv = cv_attr[0]
            
            scores = self._fit(ds_, 
                               cv_attr=cv_attr,
                               return_predictions=return_predictions,
                               return_splits=return_splits,
                               **kwargs)


            
            string_value = "_".join([str(v) for v in value])
            self.scores["%s_%s" % (r, string_value)] = scores
        
        
        self._info = self._store_ds_info(ds, 
                                         cv_attr=cv_attr,
                                         roi=roi,
                                         prepro=prepro)
        
        return self
    

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
        
    
    def _fit(self, ds, cv_attr=None, **kwargs):
        """This method must be implemented in subclasses"""
        pass
    


    def _split_name(self, splits, attr, groups):

        if isinstance(attr, str):
            groups = np.vstack((groups, groups)).T
            cv_attr = [attr, attr]

        split_ = []

        groups = groups.T

        for split in splits:
            test_name = np.unique(groups[1][split['test']])
            train_name = np.unique(groups[0][split['train']])

            test_name = [str(s) for s in test_name]
            train_name = [str(s) for s in train_name]

            split_.append({'train': "_".join(train_name), 
                           'test' : "_".join(test_name)})

        return split_

    def save(self, path=None):
        
        import os
        
        path = Analyzer.save(self, path=path)
        
        for roi, scores in self.scores.items():
                                   
            for p, score in enumerate(scores):
                    
                mat_score = self._save_score(score)
                    
                # TODO: Better use of cv and attributes for leave-one-subject-out
                filename = "%s_perm_%04d_data.mat" %(roi, p)
                logger.info("Saving %s" %(filename))
                
                savemat(os.path.join(path, filename), mat_score)
                #logger.debug(hpy().heap())
                del mat_score
                
        return

        
        
    def _save_score(self, score):
         
        mat_file = dict()
        
        for key, value in score.items():
            
            if key.find("test_") != -1:
                mat_file[key] = value
                
            elif key == 'estimator':
                mat_estimator = self._save_estimator(value)
                mat_file.update(mat_estimator)
                
            elif key == "splits":
                mat_splits = self._save_splits(value)
                mat_file.update(mat_splits)

            elif key == "split_name":
                mat_file['split_name'] = [s['test'] for s in value]
            
        
        return mat_file
        
    
    
    def _save_estimator(self, estimator):
        
        mat_ = dict()
        mat_['weights'] = []
        mat_['features'] = []
        
        for est in estimator:
            
            w = est.named_steps['clf'].coef_
            mat_['weights'].append(w)
            
            if 'fsel' in est.named_steps.keys():
                f = est.named_steps['fsel'].get_support()
                mat_['features'].append(f)
                
        return mat_
        
        
        
    def _save_splits(self, splits):
        
        
        mat_ = dict()
        mat_['train'] = []
        mat_['test'] = []
        
        for spl in splits:
            
            for set_ in mat_.keys():
                mat_[set_].append(spl[set_])

                
        return mat_        