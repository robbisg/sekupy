import numpy as np

from pyitab.utils.dataset import get_ds_data
from pyitab.analysis.utils import get_rois, get_params

from pyitab.preprocessing import FeatureSlicer
from pyitab.analysis.base import Analyzer
from pyitab.preprocessing.base import Transformer

from scipy.io.matlab.mio import savemat
from scipy.spatial.distance import pdist

import logging
logger = logging.getLogger(__name__)

class RSA(Analyzer):
    """Implement representational similarity analysis (RSA) using 
    an arbitrary type of similarity measure.

    Parameters
    -----------

    n_jobs : int, optional. Default is -1.
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    scoring : string or callable, optional
        The scoring strategy to use. See the scikit-learn documentation
        
        
    permutation : int. Default is 0.
        The number of permutation to be performed.
        If the number is 0, no permutation is performed.

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
                 n_jobs=1, 
                 permutation=0,
                 verbose=1,
                 name='rsa',
                 **kwargs):

        
        self.n_jobs = n_jobs
        self.permutation = permutation
        self.verbose = verbose

        Analyzer.__init__(self,
                          name=name,
                          **kwargs,
                          )

  

    def fit(self, ds,  
            roi='all', 
            roi_values=None,
            metric='euclidean',
            prepro=Transformer(),
            **kwargs):

        """Fits the RSA on the dataset
        
        Parameters
        ----------
        ds : [type]
            [description]
        cv_attr : str, optional
            [description] (the default is 'chunks', which [default_description])
        roi : list, optional
            list of strings that must be present in ds.fa keys
            (the default is 'all', which [default_description])
        roi_values : list, optional
            A list of key, value tuple where the key is the
            roi name, specified in ds.fa.roi and value is the value of the
            subroi. (the default is None, which [default_description])
        distance : str, optional
            The metric to be used to calculate the dissimilarity. (default: euclidean)
        prepro : [type], optional
            [description] (the default is Transformer(), which [default_description])
        return_predictions : bool, optional
            [description] (the default is False, which [default_description])
        return_splits : bool, optional
            [description] (the default is True, which [default_description])
        
        """

        if roi_values is None:
            roi_values = get_rois(ds, roi)
                
        scores = dict()
        # TODO: How to use multiple ROIs
        for r, value in roi_values:
            
            ds_ = FeatureSlicer(**{r: value}).transform(ds)
            ds_ = prepro.transform(ds_)
            
            logger.info("Dataset shape %s" % (str(ds_.shape)))
            
            X = ds_.samples

            distance = pdist(X, metric=metric)
           
            string_value = "+".join([str(v) for v in value])
            scores["mask-%s_value-%s" % (r, string_value)] = distance
        

        self._info = self._store_info(ds, 
                                      distance=metric,
                                      roi=roi,
                                      prepro=prepro)

        self.scores = scores
        self.distance = metric
        self.conditions = ds.targets.copy()
        
        return self

    def transform(self, ds, roi='all', 
                  roi_values=None, prepro=Transformer(),
                  **kwargs):

        if not hasattr(self, 'scores'):
            self.fit(ds, roi=roi, roi_values=roi_values,
                    prepro=prepro, **kwargs)

        
        samples = np.vstack([v for k,v in self.scores.items()]).T
        #ds_ = Dataset(samples)
        
        return samples


    
    # Only in subclasses
    def _get_analysis_info(self):

        info = Analyzer._get_analysis_info(self)
        info['roi'] = self._info['roi']

        return info


    def save(self, path=None, **kwargs):
        """Save the results
        
        Parameters
        ----------
        path : str, optional
            path where to store files (the default is 
            set up by :class:`pyitab.analysis.Analyzer`)
        """
        
        import os

        path, prefix = Analyzer.save(self, path=path, **kwargs)
        kwargs.update({'prefix': prefix})

        mat_score = dict()
        mat_score['conditions'] = self.conditions
        for roi, scores in self.scores.items():
            kwargs.update({'mask': roi})
            filename = self._get_filename(**kwargs)
            logger.info("Saving %s" % (filename))
            
            mat_score['test_score'] = scores
            savemat(os.path.join(path, filename), mat_score)
                
        return


    def _get_filename(self, **kwargs):
        "target-<values>_id-<datetime>_mask-<mask>_value-<roi_value>_data.mat"
        logger.debug(kwargs)

        params = {'distance': self.distance}
       
        if 'prepro' in kwargs.keys():

            for keyword in ["sample_slicer", "target_transformer", "sample_transformer"]:
                if keyword in kwargs['prepro']:
                    params_ = get_params(kwargs, keyword)

                    if 'fx' in params_.keys() and keyword == 'target_transformer':
                        params_['target_transformer-fx'] = params_['fx'][0]

                    if keyword == "sample_slicer":
                        params_ = {k: "+".join([str(v) for v in value]) for k, value in params_.items()}
                    
                    if keyword == "sample_transformer":
                        params_ = {k: "+".join([str(v) for v in value]) for k, value in params_['attr'].items()}

                    params.update(params_)

        logger.debug(params)

        # TODO: Solve empty prefix
        prefix = kwargs.pop('prefix')
        midpart = "_".join(["%s-%s" % (k, str(v).replace("_", "+")) \
             for k, v in params.items()])
        trailing = kwargs.pop('mask')
        filename = "%s_data.mat" % ("_".join([prefix, midpart,trailing]))

        return filename