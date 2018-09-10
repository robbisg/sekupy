import os
from pyitab.utils.io import get_time, make_dir

class Node(object):


    def __init__(self, name='none'):
        self.name = name
        self._info = dict()
        
    
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
        