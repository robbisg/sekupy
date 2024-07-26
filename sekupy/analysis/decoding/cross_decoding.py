import numpy as np

from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection._split import LeaveOneGroupOut


from sekupy.utils.dataset import get_ds_data

from sekupy.ext.sklearn._validation import cross_validate

from sekupy.preprocessing import FeatureSlicer, SampleSlicer
from sekupy.analysis.decoding import Decoding
from sekupy.analysis.decoding.roi_decoding import RoiDecoding
from sekupy.preprocessing.base import Transformer

from sekupy.utils.bids import get_dictionary

from scipy.io.matlab.mio import savemat

import logging
logger = logging.getLogger(__name__)


# TODO: Inherit from MetaDecoding
class CrossDecoding(Decoding):
    """Implement cross-decoding analyses using an arbitrary type of classifier.

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
                 decoder=RoiDecoding,
                 cv=LeaveOneGroupOut(),
                 permutation=0,
                 verbose=1,
                 name='roi_decoding',
                 **kwargs):


        self.analysis = decoder(
                          #self,
                          estimator=estimator,
                          n_jobs=n_jobs,
                          scoring=scoring,
                          cv=cv,
                          permutation=permutation,
                          verbose=verbose,
                          #name=name,
                          **kwargs,
                          )

  

    def fit(self, ds, 
            training_conditions,
            testing_conditions,
            targets_map=None,
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
        training_condition : dictionary
            sub-dataset to be used in training phase, for example {'subject':['sub-1', 'sub-2'], 'session':['2']}
            is used to train dataset on subjects 1 and 2 and session 2.
        testing_conditions : dictionary
            sub-dataset to be used in testing phase, for example {'subject':['sub-3', 'sub-4'], 'session':['2']}
            is used to test the decoder in subjects 1 and 2 and session 2.
        targets_map : dictionary
            This is used for across-conditions decoding in which we want to translate conditions from, for example,
            Left-Cue/Right-Cue to Left-Hand/Right-Hand.
        cv_attr : str, optional
            [description] (the default is 'chunks', which [default_description])
        roi : list, optional
            list of strings that must be present in ds.fa keys
            (the default is 'all', which [default_description])
        roi_values : list, optional
            A list of key, value tuple where the key is the
            roi name, specified in ds.fa.roi and value is the value of the
            subroi. (the default is None, which [default_description])
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

        ds_train = SampleSlicer(**training_conditions).transform(ds)
        ds_test  = SampleSlicer(**testing_conditions).transform(ds)

        # Map conditions

        self.analysis.fit(ds_train, 
                          cv_attr=cv_attr,
                          roi=roi,
                          roi_values=roi_values,
                          prepro=prepro,
                          return_predictions=return_predictions,
                          return_splits=return_splits,
                          return_decisions=return_decisions,
                          **kwargs)


        scores = self.analysis.scores

        cross_scores = dict()
        for maskname, results in scores.items():
            estimator = results[0]['estimator'] # First permutation

            info = get_dictionary(maskname)
            mask = info['mask']
            value = info['value']

            ds_ = FeatureSlicer(**{mask:[float(value)]}).transform(ds_test)
            ds_ = prepro.transform(ds_)
            
            X, y, groups = self.analysis._get_data(ds_, cv_attr, **kwargs)
            cross_score = np.array([est.score(X, y) for est in estimator])

            cross_scores["mask-%s_value-%s" % (mask, value)] = cross_score

        self.cross_scores = cross_scores
        
        return self    


