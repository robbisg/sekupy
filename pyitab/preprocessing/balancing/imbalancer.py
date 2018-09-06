from mvpa_itab.pipeline import Transformer
from mvpa_itab.io.utils import get_ds_data
from mvpa2.datasets.base import Dataset
from collections import Counter
import numpy as np

import logging
from imblearn.under_sampling._prototype_selection._random_under_sampler import RandomUnderSampler
logger = logging.getLogger(__name__)

class Imbalancer(Transformer):
    
    def __init__(self, ratio=0.75, name='imbalancer', **kwargs):
        self.ratio = ratio
        Transformer.__init__(self, name=name, **kwargs)
        
        
    def transform(self, ds):
            
        X, y = get_ds_data(ds)
        
        #X_res, y_res = make_imbalance(X, y, self.get_ratio(y))
        
        logger.info('The original target distribution in the dataset is: %s', Counter(y))
        _, _, idx = RandomUnderSampler(ratio=self.get_ratio(y), 
                                       return_indices=True).fit_sample(X, y)
        
        ds_ = ds[idx]
        logger.info('Make the dataset imbalanced: %s', Counter(ds_.targets))
        
        return ds_
    
    
    def get_ratio(self, y):
        
        ratio = {}
        for i, (label, num) in enumerate(Counter(y).items()):
            if i == 0:
                ratio[label] = np.int_(self.ratio * num)
            else:
                ratio[label] = num
                
        return ratio
    
    