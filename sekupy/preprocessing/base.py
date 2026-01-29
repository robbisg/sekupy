from sekupy.base import Node

import logging
logger = logging.getLogger(__name__)

class Transformer(Node):
    """Base class for data transformation components.
    
    Transformers are used to preprocess datasets in the sekupy framework.
    They inherit from Node and provide functionality to transform datasets
    while tracking the applied transformations.
    
    Parameters
    ----------
    name : str, optional
        Name of the transformer, by default 'transformer'
    **kwargs : dict
        Additional parameters for the transformer
        
    Attributes
    ----------
    _mapper : dict
        Dictionary storing the transformer's configuration
    """
    
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
        """Set the mapper configuration for the transformer.
        
        Parameters
        ----------
        **kwargs : dict
            Configuration parameters for the transformer
            
        Returns
        -------
        dict
            Dictionary with transformer name as key and kwargs as value
        """
        return {self.name: kwargs}


    def transform(self, ds):
        """Transform the provided dataset.
        
        This method applies the transformation to the dataset and
        records the transformation in the dataset's preprocessing history.
        
        Parameters
        ----------
        ds : Dataset
            The dataset to transform
            
        Returns
        -------
        Dataset
            The transformed dataset
        """
        self.map_transformer(ds)
        return ds
    

    def map_transformer(self, ds):
        """Map the transformer to the dataset's preprocessing history.
        
        This method records the transformer configuration in the dataset's
        preprocessing attribute for reproducibility.
        
        Parameters
        ----------
        ds : Dataset
            The dataset to which the transformer mapping is applied
        """

        if 'prepro' not in ds.a.keys():
            ds.a['prepro'] = [self._mapper]
        else:
            ds.a.prepro.append(self._mapper)
        logger.debug(ds.a.prepro)
        

    def save(self, path=None):
        return Node.save(self, path=path)



class PreprocessingPipeline(Transformer):
    """Pipeline for chaining multiple preprocessing transformers.
    
    This class allows combining multiple preprocessing steps into a single
    pipeline that can be applied to datasets sequentially.
    
    Parameters
    ----------
    name : str, optional
        Name of the pipeline, by default 'pipeline'
    nodes : list, optional
        List of transformer nodes or node names to include in the pipeline
    nodes_kwargs : dict, optional
        Keyword arguments for nodes if nodes are specified as strings
        
    Attributes
    ----------
    nodes : list
        List of transformer nodes in the pipeline
    sliced_nodes : list
        Copy of nodes list for internal use
    """
    
    
    def __init__(self, name='pipeline', nodes=None, nodes_kwargs=None):
                
        self.nodes = []
        
        if nodes is not None:
            self.nodes = nodes
        
            if isinstance(nodes[0], str):
                self.nodes = self._get_nodes(nodes, nodes_kwargs)

        self.sliced_nodes = self.nodes
                    
        Transformer.__init__(self, name)
    
    
    def add(self, node):
        """Add a transformer node to the pipeline.
        
        Parameters
        ----------
        node : Transformer
            The transformer node to add to the pipeline
            
        Returns
        -------
        PreprocessingPipeline
            Self, for method chaining
        """
        
        self.nodes.append(node)
        return self
    
    
    def transform(self, ds):
        """Transform the dataset through all nodes in the pipeline.
        
        This method applies each transformer in the pipeline sequentially
        to the dataset.
        
        Parameters
        ----------
        ds : Dataset
            The dataset to transform
            
        Returns
        -------
        Dataset
            The transformed dataset after applying all pipeline nodes
        """
        logger.info("%s is performing..." % (self.name))
        for node in self.nodes:
            ds = node.transform(ds)

        return ds

    def _get_nodes(self, nodes, nodes_kwargs):

        from sekupy.preprocessing.mapper import function_mapper
                
        node_list = []
        for key in nodes:
            class_ = function_mapper(key)
            
            arg_dict = nodes_kwargs[key]

            if key == 'sample_slicer' and 'attr' in arg_dict.keys():
                arg_dict = arg_dict['attr']

            object_ = class_(**arg_dict)
            node_list.append(object_)

        return node_list





    def __getitem__(self, ind):

        if isinstance(ind, slice):
            return self.__class__(nodes=self.nodes[ind])

        elif isinstance(ind, int):
            return self.nodes[ind]