from pyitab.analysis import Node


class Transformer(Node):
    
    def __init__(self, name='transformer'):
        """Base class for the transformer. 
        
        Parameters
        ----------
        name : str, optional
            Name of the transformer (the default is 'transformer')
        
        """

        Node.__init__(self, name=name)
        
        
    def transform(self, ds):
        return ds
    
    
    def save(self, path=None):
        return Node.save(self, path=path)