from pyitab.base import Transformer
from pyitab.preprocessing.functions import SampleSlicer
from pyitab.utils.dataset import get_ds_data
from collections import Counter
import numpy as np

import logging
from imblearn.under_sampling import RandomUnderSampler
logger = logging.getLogger(__name__)


class Imbalancer(Transformer):
    
    def __init__(self, sampling_strategy=0.75, attr=None, **kwargs):
        
        self.ratio = sampling_strategy
        self._attr = attr
        Transformer.__init__(self, name='imbalancer', attr=attr, ratio=sampling_strategy)
        
        
    def _balance(self, ds):
            
        X, y = get_ds_data(ds)

        mask = np.zeros_like(y, dtype=np.bool)
        logger.debug('Attribute balanced dataset: %s', Counter(ds.targets))
        
        ratio = self.get_ratio(y)
        _, _, idx = RandomUnderSampler(sampling_strategy=ratio, 
                                       return_indices=True).fit_sample(X, y)
        
        mask[idx] = True
        
        logger.debug('Attribute imbalanced dataset: %s', Counter(ds[mask].targets))
        
        return mask
    
    
    def get_ratio(self, y):
        
        ratio = {}
        labels, counts = np.unique(y, return_counts=True)
        for i, (label, num) in enumerate(zip(labels, counts)):
            if i == 0:
                ratio[label] = np.int_(np.ceil(self.ratio * num))
            else:
                ratio[label] = num
                
        return ratio


    def transform(self, ds):

        logger.info('The original target distribution in the dataset is: %s', 
            Counter(ds.targets))

        if self._attr is None:
            attributes = [list(np.unique(ds.targets))]
            self._attr = 'targets'
        else:
            attributes = [[a] for a in np.unique(ds.sa[self._attr].value)]

        masks = []
        for attribute in attributes:
            selection_dict = {self._attr : [*attribute]}
            ds_ = SampleSlicer(**selection_dict).transform(ds)

            mask = self._balance(ds_)
            masks.append(mask)

        
        logger.info('Imbalanced dataset is: %s', 
            Counter(ds[np.hstack(masks)].targets))

        ds = ds[np.hstack(masks)]
        return Transformer.transform(self, ds)

    
    