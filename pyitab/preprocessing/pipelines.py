from pyitab.preprocessing import  Detrender, SampleSlicer
from pyitab.preprocessing.normalizers import SampleZNormalizer, FeatureZNormalizer
from pyitab.preprocessing.base import Transformer, PreprocessingPipeline
from pyitab.preprocessing.mapper import function_mapper

import logging
logger = logging.getLogger(__name__)
    
    
class StandardPreprocessingPipeline(PreprocessingPipeline):
    
    def __init__(self, **kwargs):
        
        self.nodes = [
                      Detrender(chunks_attr='file'),
                      Detrender(),
                      FeatureZNormalizer(),            
                      SampleZNormalizer(),
                      ]
        
        PreprocessingPipeline.__init__(self, nodes=self.nodes)