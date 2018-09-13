from pyitab.analysis import Node


class Transformer(Node):
    
    def __init__(self, name='transformer', **kwargs):
        """Base class for transformer. 
        
        Parameters
        ----------
        name : str, optional
            Name of the transformer (the default is 'transformer')
        
        """

        Node.__init__(self, name=name, **kwargs)
        
        
    def transform(self, ds):
        return ds
    
    
    def save(self, path=None):
        return Node.save(self, path=path)