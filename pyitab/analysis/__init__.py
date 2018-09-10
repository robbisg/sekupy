import logging
import os
from mvpa_itab.results import get_time, make_dir
from pyitab.io.configuration import save_configuration
logger = logging.getLogger(__name__)    


class Node(object):


    def __init__(self, name='none'):
        self.name = name
        
    
    def save(self, path=None):
        return
    
    
    def _get_path(self, **kwargs):
        
        # Get information to make the results dir
        datetime = get_time()
        path = kwargs.pop('path')
        
        items = [datetime]
        items += [v for _, v in kwargs.items()]
        
        dir_ = "_".join(items)

        path = os.path.join(path, '0_results', dir_)
            
        return path
    



class Transformer(Node):
    
    def __init__(self, name='transformer', **kwargs):
        self.foo = "foo"
        Node.__init__(self, name=name, **kwargs)
        
        
    def transform(self, ds):
        return ds
    
    
    def save(self, path=None):
        return Node.save(self, path=path)




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
        # TODO: Superclass?
        info = dict()
        
        info['path'] = self._info['a'].data_path
        info['experiment'] = self._info['a'].experiment
        info['task'] = self._info['a'].task
        info['analysis'] = self.name
               
        return info
            
            
    def _store_ds_info(self, ds, **kwargs):
        # TODO: Superclass function ?
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

