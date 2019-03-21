import numpy as np
from pyitab.preprocessing import Transformer

import logging
logger = logging.getLogger(__name__)

class MemoryReducer(Transformer):
    
    def __init__(self, dtype, **kwargs):
        self._dtype = dtype
        Transformer.__init__(self, name='memory_reducer', **kwargs)
        
        
    def transform(self, ds):
        
        logger.info("Converted ds to %s", str(self._dtype))
        
        ds.samples = self._dtype(ds.samples)
                        
        return ds