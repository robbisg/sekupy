from pyitab.base import Transformer
from pyitab.preprocessing.functions import SampleSlicer
from pyitab.utils.dataset import get_ds_data
from pyitab.preprocessing.balancing.utils import sample_generator
from mvpa2.datasets.base import Dataset
from imblearn.under_sampling import RandomUnderSampler

from collections import Counter
import numpy as np
from mvpa2.base.dataset import vstack

import logging



logger = logging.getLogger(__name__)


class Balancer(Transformer):
    """[summary]
    
    Parameters
    ----------
    balancer : [type], optional
        [description] (the default is RandomUnderSampler(return_indices=True), which [default_description])
    attr : str, optional
        [description] (the default is 'chunks', which [default_description])
    force_balance : boolean
        [description]
    
    """
    
    
    def __init__(self, 
                 balancer=RandomUnderSampler(return_indices=True), 
                 attr='chunks', **kwargs):

        # TODO: attribute list
        
        self._attr = attr
        self._balancer_algorithm = balancer
        self._balancer = self._check_balancer(balancer)     
                   
        Transformer.__init__(self, name='balancer', **kwargs)
        
    
    
    def _check_balancer(self, balancer):
        
        balancer_type = str(balancer.__class__).split('.')[1]
                
        balancer_ = OverSamplingBalancer(balancer, self._attr)
        if balancer_type == 'under_sampling':
            balancer_ = UnderSamplingBalancer(balancer, self._attr)
        
        logger.debug(balancer_type)

        return balancer_
        
        
    
    def transform(self, ds):
        logger.info("Using %s" %(str(self._balancer._balancer)))
        return self._balancer.transform(ds)
     
    
    

class SamplingBalancer(Transformer):
    
    
    def __init__(self, balancer, attr='chunks', name='balancer', **kwargs):
        
        self._attr = attr
        self._balancer = balancer
        self._force_balancing = False

        if isinstance(balancer.sampling_strategy, dict):
            self._force_balancing = True
        
        Transformer.__init__(self, name=name, **kwargs)
        
        
        
    def transform(self, ds):
        
        logger.info("Init: %s" % (str(Counter(ds.targets))))
        
        if self._attr != 'all':
            balanced_ds = self._balance_attr(ds)
        else:
            balanced_ds = self._balance(ds)

        logger.info("Final: %s" % (str(Counter(balanced_ds.targets))))
        
        return balanced_ds

    def _balance(self, ds):
        return
    
    
    
    def _balance_attr(self, ds):
        
        balanced_ds = []
        logger.debug(np.unique(ds.sa[self._attr].value))
        for attribute in np.unique(ds.sa[self._attr].value):

            selection_dict = {self._attr : [attribute]}
            ds_ = SampleSlicer(**selection_dict).transform(ds)
            
            ds_b = self._balance(ds_)  
            logger.debug(Counter(ds_b.targets))
            balanced_ds.append(ds_b)
                        
        balanced_ds = vstack(balanced_ds)
        balanced_ds.a.update(ds.a)
        
        return balanced_ds
    

class UnderSamplingBalancer(SamplingBalancer):
    
    
    def __init__(self, balancer, attr='chunks', **kwargs):
        
        if not balancer.return_indices:
            logger.info("Balancer must return indices, set return_indices to True")
            logger.info("Balancer set to default RandomUnderSampler.")
            balancer = RandomUnderSampler(return_indices=True)
        
        SamplingBalancer.__init__(self, balancer, attr, name='under_balancer', **kwargs)
        

    # TODO: Multiple attributes
    def _balance(self, ds):
        
        X, y = get_ds_data(ds)
        self._mask = np.arange(len(y))
        _, count = np.unique(y, return_counts=True)
        
        if len(np.unique(count)) == 1 and not self._force_balancing:
            logger.debug(count)
            return ds
        
        _, _, indices = self._balancer.fit_sample(X, y)
        self._mask = indices
        
        return ds[indices]
    

        
class OverSamplingBalancer(SamplingBalancer):
    
    
    def __init__(self, balancer, attr='chunks', **kwargs):     
        SamplingBalancer.__init__(self, balancer, attr, name='over_balancer', **kwargs)
           
      
    def _balance(self, ds):
        
        X, y = get_ds_data(ds)
        
        X_, y_ = self._balancer.fit_sample(X, y)
        
        ds_ = self._update_ds(ds, X_, y_)

        return ds_
        
        
    def _update_ds(self, ds, X, y):
        
        ds_ = Dataset.from_wizard(X)
        
        samples_difference = len(y) - len(ds.targets) 
        
        for key in ds.sa.keys():
            
            values = ds.sa[key].value      
            values_ = sample_generator(key, values, samples_difference, y)
            u, c = np.unique(values_, return_counts=True)
            logger.debug("%s - sample per key: %s" %(key, str([u,c])))
            logger.debug(values_)
            
            ds_.sa[key] = values_
            
            
        return ds_   
        
