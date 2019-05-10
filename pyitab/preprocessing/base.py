from pyitab.base import Node

import logging
logger = logging.getLogger(__name__)

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

        if 'prepro' not in ds.a.keys():
            ds.a['prepro'] = [self._mapper]
        else:
            ds.a.prepro.append(self._mapper)

        logger.debug(ds.a.prepro)
        

    
    def save(self, path=None):
        return Node.save(self, path=path)