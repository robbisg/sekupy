import numpy as np

from pyitab.utils.dataset import get_ds_data
from pyitab.analysis.utils import get_rois, get_params

from pyitab.preprocessing import FeatureSlicer
from pyitab.analysis.base import Analyzer
from pyitab.preprocessing.base import Transformer
from pyitab.preprocessing.permutation import Permutator

from scipy.io.matlab.mio import savemat

from joblib import Parallel, delayed

import logging
logger = logging.getLogger(__name__)


# TODO: TEST
class RoiWrapper(Analyzer):
    """RoiWrapper is a wrapper class that can be used to iterate the analysis on 
    a subset of features, usually is performed on ROI.


    Parameters
    ----------
    analysis : [type], optional
        [description], by default Analyzer()
    n_jobs : int, optional
        [description], by default -1
    """

    def __init__(self, analysis=Analyzer(), n_jobs=-1, **kwargs):


        self.analysis = analysis
        self.n_jobs = n_jobs
        Analyzer.__init__(self,
                          **kwargs,
                          )
  

    def fit(self, ds, roi='all', 
            roi_values=None, 
            prepro=Transformer(),
            **kwargs):

        """Fit the analysis on a set of features, specified in the `roi`
        attribute.
        
        Parameters
        ----------
        ds : [type]
            [description]
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

        if roi_values is None:
            roi_values = get_rois(ds, roi)
                
        scores = dict()
        # TODO: How to use multiple ROIs

        scores = [self._parallel(ds, r, v, prepro, **kwargs) for r, v in roi_values]

        """
        scores = Parallel(n_jobs=self.n_jobs, verbose=1) \
                    (delayed(self._parallel)(ds, r, v, prepro, **kwargs)
                        for r, v in roi_values)
        """
        self.scores = scores

        self._info = self._store_info(ds)
        
        return self

    def _parallel(self, ds, roi, value, prepro, **kwargs):
        
        ds_ = FeatureSlicer(**{roi: value}).transform(ds)
        ds_ = prepro.transform(ds_)
            
        logger.info("Dataset shape %s" % (str(ds_.shape)))
                    
        self.analysis.fit(ds_, **kwargs)
        
        string_value = "+".join([str(v) for v in value])
        key = "mask-%s_value-%s" % (roi, string_value)
        value = self.analysis.scores.copy()

        return (key, value)

    
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

        <source_keywords>_task-<task>_mask-<mask>_
        value-<roi_value>_distance-<datetime>_<key>-<value>_data.mat
        """
        
        import os

        path, prefix = Analyzer.save(self, path=path, **kwargs)
        kwargs.update({'prefix': prefix})

        mat_score = dict()
        for roi, scores in self.scores.items():
            kwargs.update({'mask': roi})
            filename = self._get_filename(**kwargs)
            logger.info("Saving %s" % (filename))

            self.analysis.save(path)
            
            #mat_score['test_score'] = scores
            #savemat(os.path.join(path, filename), mat_score)
                
        return

    
    def _get_filename(self, **kwargs):
        "target-<values>_id-<datetime>_mask-<mask>_value-<roi_value>_data.mat"
        logger.debug(kwargs)

        params = {'analysis': self.analysis.name}
        params_ = self._get_prepro_info(**kwargs)
        params.update(params_)
       
        logger.debug(params)

        # TODO: Solve empty prefix
        prefix = kwargs.pop('prefix')
        midpart = "_".join(["%s-%s" % (k, str(v).replace("_", "+")) \
             for k, v in params.items()])
        trailing = kwargs.pop('mask')
        filename = "%s_data.mat" % ("_".join([prefix, midpart, trailing]))

        return filename



class PermutationWrapper(Analyzer):
    def __init__(self, analysis=Analyzer(), n_jobs=-1, n=100, **kwargs):

        self.analysis = analysis
        self.n_jobs = n_jobs
        self.n_permutations = n + 1
        self._permutator = Permutator(n=n)
        Analyzer.__init__(self,
                          **kwargs,
                          )

        return

    def fit(self, ds, **kwargs):
                
        scores = dict()
        # TODO: How to use multiple ROIs

        scores = Parallel(n_jobs=self.n_jobs, verbose=1) \
                    (delayed(self._parallel_roi)(ds, **kwargs)
                        for _ in np.arange(self.n_permutations))

        self.scores = scores

        self._info = self._store_info(ds)
        
        return self


    def _parallel(self, ds, **kwargs):
        
        ds_ = self._permutator.transform(ds)
        logger.info("Dataset shape %s" % (str(ds_.shape)))
                    
        self.analysis.fit(ds_, **kwargs)

        return self.analysis.scores.copy()