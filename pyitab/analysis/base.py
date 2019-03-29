import logging
import os
from pyitab.utils.files import make_dir
from pyitab.utils.time import get_time
from pyitab.io.configuration import save_configuration
from pyitab.base import Node
logger = logging.getLogger(__name__)


class Analyzer(Node):
    
    def __init__(self, name='analyzer', **kwargs):
        
        self.__common__ = ['estimator', 
                           'scoring', 
                           'cv', 
                           'permutation']
        
        
        
        Node.__init__(self, name=name)
        
        
    def fit(self, ds, **kwargs):
        return self


    def save(self, path=None, **kwargs):

        
        if not hasattr(self, "scores"):
            logger.error("Please run fit() before saving results.")
            
            return None
        
        if path is None:
            info = self._get_fname_info()
            path = self._get_path(**info)
            
            info.update(self._get_analysis_info(**kwargs))

            make_dir(path)
            save_configuration(path, info)
            return path
            
        return path
    
        
    def _get_fname_info(self):

        info = dict()
        logger.debug(self._info)
        info['path'] = self._info['a'].data_path
        info['experiment'] = self._info['a'].experiment
        info['task'] = self._info['a'].task
        info['analysis'] = self.name
               
        return info
            
            
    def _store_ds_info(self, ds, **kwargs):

        import numpy as np
        info = dict()
        info['a'] = ds.a.copy()
        info['sa'] = ds.sa.copy()
        info['targets'] = np.unique(ds.targets)
        info['summary'] = ds.summary()
        for k, v in kwargs.items():
            info[k] = str(v)
        logger.debug(info)
        return info
    
    
    def _get_analysis_info(self, attributes=['estimator', 
                                             'scoring', 
                                             'cv', 
                                             'permutation']):
        import numpy as np
        info = dict()
        for k in attributes:
            info[k] = getattr(self, k)
        info['targets'] = self._info['targets']
        for k in self._info['sa'].keys():
            info[k] = np.unique(self._info['sa'][k].value)
        info['summary'] = self._info['summary']
        return info


    # TODO: Look if can be applied to connectivity
    def _get_permutation_indices(self, n_samples):
        """Permutes the indices of the dataset"""
        
        # TODO: Permute labels based on cv_attr
        from sklearn.utils import shuffle
        
        if self.permutation == 0:
            return [range(n_samples)]
        
        
        indices = [range(n_samples)]
        for r in range(self.permutation):
            idx = shuffle(indices[0], random_state=r)
            indices.append(idx)
        
        return indices