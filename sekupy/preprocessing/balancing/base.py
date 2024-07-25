from sekupy.preprocessing.base import Transformer
from sekupy.preprocessing.slicers import SampleSlicer
from sekupy.utils.dataset import get_ds_data
from sekupy.preprocessing.balancing.utils import sample_generator
from sekupy.dataset.base import Dataset
from imblearn.under_sampling import RandomUnderSampler

from collections import Counter
import numpy as np
from sekupy.dataset.dataset import vstack

import logging
logger = logging.getLogger(__name__)


class Balancer(Transformer):
    """This class is used to transform an unblanced dataset in a balanced
    dataset.

    Parameters
    ----------
    balancer : :class:`~imblearn.base.BaseSampler`, optional
        [description] (the default is :class:`~imblearn.under_sampling.RandomUnderSampler` 
        which [default_description])
    attr : str, optional
        [description] (the default is 'chunks', which [default_description])
    force_balance : boolean
        [description]
    """

    def __init__(self, 
                 balancer=RandomUnderSampler(),
                 attr='chunks', **kwargs):

        # TODO: attribute list
        
        self._attr = attr
        self._balancer_algorithm = balancer
        self._balancer = self._check_balancer(balancer)     
                   
        Transformer.__init__(self, 
                             name=self._balancer.name, 
                             attr=attr, 
                             balancer=self._balancer._balancer)
        
    
    
    def _check_balancer(self, balancer):
        
        balancer_type = str(balancer.__class__).split('.')[1]
                
        balancer_ = OverSamplingBalancer(balancer, self._attr)
        if balancer_type == 'under_sampling':
            balancer_ = UnderSamplingBalancer(balancer, self._attr)
        
        logger.debug(balancer_type)

        return balancer_


    def transform(self, ds):
        logger.info("Using %s" % (str(self._balancer._balancer)))
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

        return Transformer.transform(self, balanced_ds)


    def _balance(self, ds):
        return ds


    def _balance_attr(self, ds):

        from itertools import product

        if not isinstance(self._attr, list):
            self._attr = [self._attr]

        n_attributes = len(self._attr)
        unique_attributes = product(*[np.unique(ds.sa[v].value)
                                        for v in self._attr])

        logger.debug(unique_attributes)

        balanced_ds = []
        for attributes in unique_attributes:
            selection_dict = {self._attr[i]: [attributes[i]]
                                    for i in range(n_attributes)}

            logger.debug(selection_dict)
            ds_ = SampleSlicer(**selection_dict).transform(ds)
            
            ds_b = self._balance(ds_)  
            logger.debug(Counter(ds_b.targets))
            balanced_ds.append(ds_b)
                        
        balanced_ds = vstack(balanced_ds)
        balanced_ds.a.update(ds.a)
        
        return balanced_ds
    

class UnderSamplingBalancer(SamplingBalancer):
    
    
    def __init__(self, balancer, attr='chunks', **kwargs):      
        SamplingBalancer.__init__(self, balancer, attr, name='under_balancer', **kwargs)
        

    def _balance(self, ds):
        
        X, y = get_ds_data(ds)
        if len(X.shape) > 2:
            X = X[..., 0]


        self._mask = np.arange(len(y))
        _, count = np.unique(y, return_counts=True)
        
        if len(np.unique(count)) == 1 and not self._force_balancing:
            logger.debug(count)
            return ds
        
        _ = self._balancer.fit_resample(X, y)
        self._mask = self._balancer.sample_indices_
        
        return ds[self._mask]
    

        
class OverSamplingBalancer(SamplingBalancer):
    
    
    def __init__(self, balancer, attr='chunks', **kwargs):     
        SamplingBalancer.__init__(self, balancer, attr, name='over_balancer', **kwargs)
           
      
    def _balance(self, ds):
        
        X, y = get_ds_data(ds)
        if len(X.shape) > 2:
            raise NotImplementedError('Over-sampling not implemented for this dataset.')
        
        X_, y_ = self._balancer.fit_resample(X, y)
        
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
        
