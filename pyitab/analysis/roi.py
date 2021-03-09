import numpy as np

from pyitab.utils.dataset import get_ds_data
from pyitab.analysis.utils import get_rois, get_params

from pyitab.preprocessing import FeatureSlicer
from pyitab.analysis.base import Analyzer
from pyitab.preprocessing.base import Transformer

from scipy.io.matlab.mio import savemat

from joblib import Parallel, delayed

import logging
logger = logging.getLogger(__name__)


# TODO: TEST
class RoiAnalyzer(Analyzer):
    """
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

        """[summary]
        
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

        scores = Parallel(n_jobs=self.n_jobs) \
                    (delayed(self._parallel_roi)(ds, r, v, prepro, **kwargs)
                        for r, v in roi_values)

        """
        for r, value in roi_values:
            
            ds_ = FeatureSlicer(**{r: value}).transform(ds)
            ds_ = prepro.transform(ds_)
            
            logger.info("Dataset shape %s" % (str(ds_.shape)))
                        
            self.analysis.fit(ds_, **kwargs)
            
            string_value = "+".join([str(v) for v in value])
            scores["mask-%s_value-%s" % (r, string_value)] = self.analysis.scores.copy()
        """

        self.scores = scores

        self._info = self._store_info(ds)
        
        return self

    def _parallel_roi(self, ds, roi, value, prepro, **kwargs):
        
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
            
            mat_score['test_score'] = scores
            savemat(os.path.join(path, filename), mat_score)
                
        return

    
    def _get_filename(self, **kwargs):
        "target-<values>_id-<datetime>_mask-<mask>_value-<roi_value>_data.mat"
        logger.debug(kwargs)

        params = {'analysis': self.analysis.name}
       
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
        filename = "%s_data.mat" % ("_".join([prefix, midpart, trailing]))

        return filename

