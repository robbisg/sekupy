import logging
import os
from pyitab.io.utils import make_dir
from pyitab.utils import get_time
from pyitab.io.configuration import save_configuration
from pyitab.analysis import Node
logger = logging.getLogger(__name__)


class Analyzer(Node):
    
    def __init__(self, name='analyzer', **kwargs):
        
        self.__common__ = ['estimator', 
                           'scoring', 
                           'cv', 
                           'permutation']
        
        
        
        Node.__init__(self, name=name, **kwargs)
        
        
    def fit(self, ds, **kwargs):
        return self


    def save(self, path=None):
        
        if not hasattr(self, "scores"):
            logger.error("Please run fit() before saving results.")
            
            return None
        
        
        if path is None:
            info = self._get_fname_info()
            path = self._get_path(**info)
            
            info.update(self._get_analysis_info())

            make_dir(path)
            save_configuration(path, info)
            return path
            
        return path
    
        
    def _get_fname_info(self):

        info = dict()
        
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