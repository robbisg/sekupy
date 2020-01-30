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


class PreprocessingPipeline(Transformer):
    
    
    def __init__(self, name='pipeline', nodes=None):
        from pyitab.preprocessing.mapper import function_mapper
        
        self.nodes = []
        
        if nodes is not None:
            self.nodes = nodes
        
            if isinstance(nodes[0], str):
                self.nodes = [function_mapper(node)() for node in nodes]

        self.sliced_nodes = self.nodes
                    
        Transformer.__init__(self, name)
    
    
    def add(self, node):
        
        self.nodes.append(node)
        return self
    
    
    def transform(self, ds):
        logger.info("%s is performing..." % (self.name))
        for node in self.nodes:
            ds = node.transform(ds)

        return ds


    def __getitem__(self, ind):

        if isinstance(ind, slice):
            return self.__class__(nodes=self.nodes[ind])

        elif isinstance(ind, int):
            return self.nodes[ind]