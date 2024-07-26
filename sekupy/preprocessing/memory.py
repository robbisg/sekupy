import numpy as np
from sekupy.preprocessing.base import Transformer

import logging
logger = logging.getLogger(__name__)

class MemoryReducer(Transformer):
    
    def __init__(self, dtype=np.float16, **kwargs):
        self._dtype = dtype
        Transformer.__init__(self, name='memory_reducer', dtype=dtype)
        
        
    def transform(self, ds):
        
        logger.info("Converted ds to %s", str(self._dtype))
        
        ds.samples = self._dtype(ds.samples)

        for attribute in ds.fa.keys():
            array = ds.fa[attribute].value
            ds.fa[attribute] = self._minify(array)

        for attribute in ds.sa.keys():
            array = ds.sa[attribute].value
            ds.sa[attribute] = self._minify(array)

                        
        return Transformer.transform(self, ds)

    
    def _minify(self, array):

        if np.issubdtype(array.dtype, str):
            return array
        
        elif np.issubdtype(array.dtype, float):
            return np.float16(array)

        elif np.issubdtype(array.dtype, int):
            return self._check_int(array)
        
        return np.float16(array)

    
    def _check_int(self, array):
        if np.abs(array).max() > 127:
            dtype = np.int16
        elif np.abs(array).max() > 32767:
            dtype = np.int32
        else:
            dtype = np.int8

        return dtype(array)
        

