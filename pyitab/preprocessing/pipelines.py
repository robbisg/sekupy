from pyitab.preprocessing.functions import Detrender, SampleSlicer
from pyitab.preprocessing.normalizers import SampleZNormalizer, FeatureZNormalizer
from pyitab.base import Node
from pyitab.preprocessing.mapper import function_mapper

import logging
logger = logging.getLogger(__name__)


class PreprocessingPipeline(Node):
    
    
    def __init__(self, name='pipeline', nodes=None):

        
        self.nodes = []
        
        if nodes is not None:
            self.nodes = nodes
        
            if isinstance(nodes[0], str):
                self.nodes = [function_mapper(node)() for node in nodes]
                    
        Node.__init__(self, name)
    
    
    def add(self, node):
        
        self.nodes.append(node)
        return self
    
    
    def transform(self, ds):
        logger.info("%s is performing..." %(self.name))
        for node in self.nodes:
            ds = node.transform(ds)

        return ds
            
    
    
class StandardPreprocessingPipeline(PreprocessingPipeline):
    
    def __init__(self, **kwargs):
        
        self.nodes = [
                      Detrender(chunks_attr='file'),
                      Detrender(),
                      FeatureZNormalizer(),            
                      SampleZNormalizer(),
                      ]
        
        PreprocessingPipeline.__init__(self, nodes=self.nodes)
        


class MonksPreprocessingPipeline(PreprocessingPipeline):
    
    def __init__(self, **kwargs):
        
        self.nodes = [
                      Detrender(chunks_attr='file'),
                      Detrender(),
                      FeatureZNormalizer(),
                      SampleSlicer(selection_dictionary={'events_number':range(1, 13)})                 
                      
                      ]
        
        PreprocessingPipeline.__init__(self, nodes=self.nodes)
        
        

class MonksConnectivityPipeline(PreprocessingPipeline):
    
    def __init__(self, **kwargs):
        
        self.nodes = [
                      Detrender(chunks_attr='file'),
                      Detrender(),
                      FeatureZNormalizer(),
                      SampleSlicer(selection_dictionary={'events_number':range(1,13)})                 
                      
                      ]
        
        PreprocessingPipeline.__init__(self, nodes=self.nodes)              