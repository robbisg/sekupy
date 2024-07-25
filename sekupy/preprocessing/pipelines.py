from sekupy.preprocessing import  Detrender, SampleSlicer
from sekupy.preprocessing.normalizers import SampleZNormalizer, FeatureZNormalizer
from sekupy.preprocessing.base import Transformer, PreprocessingPipeline
from sekupy.preprocessing.mapper import function_mapper

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