import numpy as np

from sklearn.metrics.scorer import _check_multimetric_scoring
from sklearn.svm import SVC
from sklearn.preprocessing.label import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection._split import LeaveOneGroupOut
from sklearn.model_selection._validation import cross_validate

from tqdm import tqdm

from pyitab.utils.dataset import get_ds_data

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
                 name='roi_decoding',
                 verbose=1):


        Decoding.__init__(self,
                          estimator=estimator,
                          n_jobs=n_jobs,
                          scoring=scoring,
                          cv=cv,
                          permutation=permutation,
                          verbose=verbose,
                          name=name)


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
    

    def fit(self, ds, 
            cv_attr='chunks', 
            roi='all', 
            roi_values=None, 
            prepro=Transformer(),
            return_predictions=False,
            return_splits=True,
            return_decisions=False,
            **kwargs):

        """[summary]
        
        Parameters
        ----------
        ds : [type]
            [description]
        cv_attr : str, optional
            [description] (the default is 'chunks', which [default_description])
        roi : str, optional
            [description] (the default is 'all', which [default_description])
        roi_values : [type], optional
            [description] (the default is None, which [default_description])
        prepro : [type], optional
            [description] (the default is Transformer(), which [default_description])
        return_predictions : bool, optional
            [description] (the default is False, which [default_description])
        return_splits : bool, optional
            [description] (the default is True, which [default_description])
        
        Returns
        -------
        [type]
            [description]
        """

        if roi_values is None:
            roi_values = self._get_rois(ds, roi)
                
        scores = dict()
        # TODO: How to use multiple ROIs
        for r, value in roi_values:
            
            ds_ = FeatureSlicer(**{r:value}).transform(ds)
            ds_ = prepro.transform(ds_)
            
            logger.info("Dataset shape %s" % (str(ds_.shape)))
            summary_cv = cv_attr
            if isinstance(cv_attr, list):
                summary_cv = cv_attr[0]
            
            super().fit(ds_, 
                        cv_attr=cv_attr,
                        return_predictions=return_predictions,
                        return_splits=return_splits,
                        return_decisions=return_decisions,
                        **kwargs)


            
            string_value = "_".join([str(v) for v in value])
            scores["%s_%s" % (r, string_value)] = self.scores
        
        
        self._info = self._store_ds_info(ds, 
                                         cv_attr=cv_attr,
                                         roi=roi,
                                         prepro=prepro)

        self.scores = scores
        
        return self    

    
    

