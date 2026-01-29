class Node(object):
    """Base class for all sekupy nodes.
    
    This is the fundamental building block for analysis components
    in the sekupy framework. All analysis classes inherit from Node
    to provide consistent naming and information storage capabilities.
    
    Parameters
    ----------
    name : str, optional
        Name identifier for the node, by default 'none'
    **kwargs : dict
        Additional keyword arguments passed to the node
        
    Attributes
    ----------
    name : str
        The name identifier of the node
    _info : dict
        Internal dictionary for storing node information
    """

    def __init__(self, name='none', **kwargs):
        """Initialize a Node instance.
        
        Parameters
        ----------
        name : str, optional
            Name identifier for the node, by default 'none'
        **kwargs : dict
            Additional keyword arguments
        """
        self.name = name
        self._info = dict()
        
    
    def save(self, path=None):
        """Save the node to a specified path.
        
        Base implementation that should be overridden by subclasses
        to provide actual saving functionality.
        
        Parameters
        ----------
        path : str, optional
            Path where to save the node, by default None
            
        Returns
        -------
        None
        """
        return