import numpy as np

import logging
from mvpa_itab.conn.operations import array_to_matrix, copy_matrix
from mvpa2.base.collections import SampleAttributesCollection
from mvpa2.datasets.base import Dataset
from mvpa_itab.io.base import add_attributes
from mvpa_itab.pipeline import Transformer

logger = logging.getLogger(__name__)


class SingleRowMatrixTransformer(Transformer):
    
    def __init__(self, name='upper_matrix', **kwargs):
        Transformer.__init__(self, name=name, **kwargs)       

    def transform(self, ds):
                
        data = np.dstack([copy_matrix(array_to_matrix(a)) for a in ds.samples])
        data = np.hstack([d for d in data[:,:]]).T
        
        attr = self._edit_attr(ds, data.shape)
        
        ds_ = Dataset.from_wizard(data)
        ds_ = add_attributes(ds_, attr)
        
        
        return ds_
        
        
        
    def _edit_attr(self, ds, shape):
        
        attr = dict()
        for key in ds.sa.keys():
            attr[key] = []
            for v in ds.sa[key].value:
                attr[key] += [v for _ in range(shape[1])]
        
        attr['roi_labels'] = []
        for _ in range(shape[0]/shape[1]):
            for i in range(shape[1]):
                attr['roi_labels'] += ["roi_%02d" % (i+1)]
                
                
        logger.debug(shape)
        
        return SampleAttributesCollection(attr)
    
    
    