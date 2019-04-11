import numpy as np

from sklearn.metrics.scorer import _check_multimetric_scoring
from sklearn.svm import SVC
from sklearn.preprocessing.label import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection._split import LeaveOneGroupOut

from tqdm import tqdm


from pyitab.ext.sklearn._validation import cross_validate
from pyitab.analysis.base import Analyzer
from pyitab.analysis.utils import get_params
from pyitab.utils.dataset import get_ds_data
from pyitab.utils.time import get_time
from pyitab.utils import get_id

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
                 **kwargs):
        
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
        self.scoring, _ = _check_multimetric_scoring(self.estimator, 
                                                     scoring=self.scoring)

        Analyzer.__init__(self, **kwargs)

    
    def _get_data(self, ds, cv_attr, **kwargs):
        
        X, y = get_ds_data(ds)
        y = LabelEncoder().fit_transform(y)

        groups = None
        if cv_attr is not None:
            if isinstance(cv_attr, list):
                groups = np.vstack([ds.sa[att].value for att in cv_attr]).T
            else:
                groups = ds.sa[cv_attr].value

        return X, y, groups


    

    def fit(self, ds, 
            cv_attr=None,
            return_predictions=False,
            return_splits=True,
            return_decisions=False,
            **kwargs):
        """General method to fit data"""
        

        X, y, groups = self._get_data(ds, cv_attr, **kwargs)

        indices = self._get_permutation_indices(len(y))
                
        self.scores = []
        for idx in tqdm(indices):
            
            y_ = y[idx]

            scores = cross_validate(self.estimator, X, y_, groups,
                                    self.scoring, self.cv, self.n_jobs,
                                    self.verbose, return_estimator=True, 
                                    return_splits=return_splits, 
                                    return_decisions=return_decisions,
                                    return_predictions=return_predictions)
            
            self.scores.append(scores)
            if cv_attr is not None:
                scores['split_name'] = self._split_name(scores['splits'], 
                                                        cv_attr,
                                                        groups)
       

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


    def save(self, path=None, **kwargs):
        """[summary]
        
        Parameters
        ----------
        path : [type], optional
            [description] (the default is None, which [default_description])
        
        Returns
        -------
        [type]
            [description]

        <source_keywords>_target-<values>_task-<task>_mask-<mask>_value-<roi_value>_date-<datetime>_num-<num>_<key>-<value>_data.mat
        """
        
        import os
        
        path, prefix = Analyzer.save(self, path=path, **kwargs)
        kwargs.update({'prefix': prefix})

        
        for roi, scores in self.scores.items():
            for p, score in enumerate(scores):
                    
                mat_score = self._save_score(score)
                    
                # TODO: Better use of cv and attributes for leave-one-subject-out
                kwargs.update({'mask': roi, 'perm': "%04d" % p})

                filename = self._get_filename(**kwargs)
                logger.info("Saving %s" %(filename))
                
                savemat(os.path.join(path, filename), mat_score)
                #logger.debug(hpy().heap())
                del mat_score
                
        return


    # TODO: Is it better to use a function in utils?
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

            elif key == "predictions":
                mat_file[key] = value

            elif key == 'decisions':
                mat_file[key] = list(value)
            
        return mat_file
        

    # TODO: Is it better to use a function in utils?
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
        
        
    # TODO: Is it better to use a function in utils?
    def _save_splits(self, splits):
        
        mat_ = dict()
        mat_['train'] = []
        mat_['test'] = []
        
        for spl in splits:
            
            for set_ in mat_.keys():
                mat_[set_].append(spl[set_])

                
        return mat_

    
    def _get_filename(self, **kwargs):
        "target-<values>_id-<datetime>_mask-<mask>_value-<roi_value>_data.mat"
        logger.debug(kwargs)
       
        params = dict()
        if len(kwargs.keys()) != 0:

            for keyword in ["sample_slicer", "target_trans"]:
                if keyword in kwargs['prepro']:
                    params_ = get_params(kwargs, keyword)

                    if keyword == "sample_slicer":
                        params_ = {k: "+".join([str(v) for v in value]) for k, value in params_.items()}

                    params.update(params_)
        else:
            params['targets'] = "+".join(list(self._info['sa']['targets']))


        logger.debug(params)


        trailing = kwargs.pop('mask')
        trailing += "_perm-%s" % (kwargs.pop('perm'))
        # TODO: Solve empty prefix
        prefix = kwargs.pop('prefix')
        midpart = "_".join(["%s-%s" % (k, v) for k, v in params.items()])
        
        filename = "%s_data.mat" % ("_".join([prefix, midpart, trailing]))

        return filename
