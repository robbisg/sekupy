import numpy as np
from pyitab.base import Transformer

import logging
logger = logging.getLogger(__name__)

class MemoryReducer(Transformer):
    
    def __init__(self, dtype=np.float16, **kwargs):
        self._dtype = dtype
        Transformer.__init__(self, name='memory_reducer', dtype=dtype)
        
        
    def transform(self, ds):
        
        logger.info("Converted ds to %s", str(self._dtype))
        
        ds.samples = self._dtype(ds.samples)
                        
        return Transformer.transform(self, ds)