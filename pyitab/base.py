import os
from pyitab.utils.files import make_dir
from pyitab.utils.time import get_time


class Node(object):

    def __init__(self, name='none', **kwargs):
        self.name = name
        self._info = dict()
        
    
    def save(self, path=None):
        return
    
    # DEPRECATED
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
        """Base class for the transformer. 
        
        Parameters
        ----------
        name : str, optional
            Name of the transformer (the default is 'transformer')
        
        """
        
        Node.__init__(self, name=name, **kwargs)

        self._mapper = self._set_mapper(**kwargs)
    

    def _set_mapper(self, **kwargs):
        return {self.name: kwargs}


        
    def transform(self, ds):
        self.map_transformer(ds)
        return ds
    

    def map_transformer(self, ds):
        if 'prepro' in ds.a.keys():
            ds.a['prepro'] = [self._mapper]
        else:
            ds.a['prepro'].append(self._mapper)
        

    
    def save(self, path=None):
        return Node.save(self, path=path)