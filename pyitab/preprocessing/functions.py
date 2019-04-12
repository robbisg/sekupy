from __future__ import print_function

import numpy as np
from mvpa2.mappers.detrend import PolyDetrendMapper
from mvpa2.mappers.fx import mean_group_sample
from mvpa2.mappers.zscore import ZScoreMapper
from mvpa2.base.dataset import vstack

import logging
from itertools import product
from mvpa2.base.dataset import hstack, vstack
from pyitab.base import Transformer
logger = logging.getLogger(__name__)


class Detrender(Transformer):
    """This transformer is used to detrend data.
    
    Parameters
    ----------
    degree : int, optional
        The polynomial degree of the detrending function
        (the default is 1)
    chunks_attr : str, optional
        The attribute used to get data for the detrend
        (the default is 'chunks', which can be only a sample attribute
        of the dataset)
    
    """

    
    def __init__(self, degree=1, chunks_attr='chunks', **kwargs):

        self._degree = degree
        self.node = PolyDetrendMapper(chunks_attr=chunks_attr, polyord=degree)
        Transformer.__init__(self, name='detrender', 
                            degree=degree, chunks_attr=chunks_attr)
            
    
    def transform(self, ds):
        """This function performs the detrending
        
        Parameters
        ----------
        ds : pymvpa Dataset
            The dataset to be transformed.
        
        Returns
        -------
        ds : pymvpa Dataset
            The transformed dataset
        """
        
        self.node.train(ds)

        logger.info('Dataset preprocessing: Detrending with polynomial of order %s...', (str(self._degree)))
        return self.node.forward(ds)
                   


class SampleAverager(Transformer):
    """This transformer is used to average data.
    
    Parameters          
    ----------
    attributes : list
      List of sample attributes whose unique values will be used to identify the
      samples groups.   
    """
    
    
    def __init__(self, attributes):
        self.node = mean_group_sample(attributes)

        attr_string = '.'.join(attributes)

        Transformer.__init__(self, name='sample_averager',
                            attributes=attr_string)
        
        
    def transform(self, ds):
        """This function performs the averaging
        
        Parameters
        ----------
        ds : pymvpa Dataset
            The dataset to be transformed.
        
        Returns
        -------
        ds : pymvpa Dataset
            The transformed dataset
        """
        logger.info('Dataset preprocessing: Averaging samples...')
        return ds.get_mapped(self.node)  



class TargetTransformer(Transformer):
    
    def __init__(self, attr=None, **kwargs):
        self._attribute = attr
        Transformer.__init__(self, name='target_transformer', attr=attr)
    
    def transform(self, ds):
        logger.info("Dataset preprocessing: Target set to %s" , (self._attribute))
        ds.targets = ds.sa[self._attribute]
        
        return ds



class FeatureSlicer(Transformer):
    """ This transformer filters the dataset using features as specified on a dictionary
    The dictionary indicates the feature attributes to be used as key and a list
    with conditions to be selected:
    
    selection_dict = {
                        'accuracy': ['I'],
                        'frame':[1,2,3]
                        }
                        
    This dictionary means that we will select all features with frame attribute
    equal to 1 OR 2 OR 3 AND all samples with accuracy equal to 'I'.

    
    """
    
    def __init__(self, **kwargs):
        print(kwargs)
        
        self._selection = dict()
        for arg in kwargs:
            self._selection[arg] = kwargs[arg]
        Transformer.__init__(self, name='feature_slicer', **self._selection)  

    
    def _set_mapper(self, **kwargs):

        for k, v in kwargs.items():
            kwargs[k] = "+".join(str(v)) 

        return Transformer._set_mapper(self, **kwargs)


    def transform(self, ds):
        
        selection_dict = self._selection
    
        selection_mask = np.ones(ds.shape[1], dtype=np.bool)
        for key, values in selection_dict.items():
            
            logger.info("Selected %s from %s attribute.", str(values), key)
            
            ds_values = ds.fa[key].value
            condition_mask = np.zeros_like(ds_values, dtype=np.bool)
            
            for value in values:

                if str(value)[0] == '!':
                    array_val = np.array(value[1:]).astype(ds_values.dtype)
                    condition_mask = np.logical_or(condition_mask, ds_values != array_val)
                else:
                    condition_mask = np.logical_or(condition_mask, ds_values == value)
                    
            selection_mask = np.logical_and(selection_mask, condition_mask)
            
        
        return ds[:, selection_mask]



class SampleSlicer(Transformer):
    """
    Selects only portions of the dataset based on a dictionary
    The dictionary indicates the sample attributes to be used as key and a list
    with conditions to be selected:
    
    selection_dict = {
                        'frame':[1,2,3]
                        }
                        
    This dictionary means that we will select all samples with frame attribute
    equal to 1 OR 2 OR 3 AND all samples with accuracy equal to 'I'.
    
    """

    def __init__(self, **kwargs):
        
        self._selection = dict()
        for arg in kwargs:
            self._selection[arg] = kwargs[arg]
         
        Transformer.__init__(self, name='sample_slicer', **kwargs)

    
    def _set_mapper(self, **kwargs):

        for k, v in kwargs.items():
            kwargs[k] = "+".join(str(v)) 

        return Transformer._set_mapper(self, **kwargs)



    def transform(self, ds):
        
        selection_dict = self._selection
    
        selection_mask = np.ones_like(ds.targets, dtype=np.bool)
        for key, values in selection_dict.items():
            
            logger.info("Selected %s from %s attribute.", str(values), key)
            
            ds_values = ds.sa[key].value
            condition_mask = np.zeros_like(ds_values, dtype=np.bool)
            
            for value in values:        
                condition_mask = np.logical_or(condition_mask, ds_values == value)
                
            selection_mask = np.logical_and(selection_mask, condition_mask)
            
        
        return ds[selection_mask]
    


# TODO: Document
class FeatureStacker(Transformer):
    """
    Use features in the dictionary to build a rich dataset
    The dictionary indicates the attributes to be used as key and a list
    with conditions to be selected:
    
    selection_dict = {
                        'frame':[1,2,3]
                        }
                        
    This dictionary means that we will select all samples with frame attribute
    equal to 1 OR 2 OR 3 AND all samples with accuracy equal to 'I'.   
    """

    def __init__(self, 
                 selection_dictionary=None, 
                 stack_attr=['targets', 'chunks'], 
                 **kwargs):
        
        self._selection = selection_dictionary
        self._attr = stack_attr
        Transformer.__init__(self, name='sample_stacker', 
                                   selection=selection_dictionary,
                                   attr=stack_attr
                                   )   

    
    def _set_mapper(self, **kwargs):

        for k, v in kwargs.items():
            kwargs[k] = "+".join(str(v)) 

        return Transformer._set_mapper(self, **kwargs)


    def transform(self, ds):
        
        ds_ = SampleSlicer(**self._selection).transform(ds)
        
        iterable = [np.unique(ds_.sa[a].value) for a in self._attr]
        
        ds_stack = []
        for attr in product(*iterable):
            
            mask = np.ones_like(ds_.targets, dtype=np.bool)
            
            for i, a in enumerate(attr):
                mask = np.logical_and(mask, ds_.sa[self._attr[i]].value == a)
            
            ds_stacked = hstack([d for d in ds_[mask]])
            ds_stacked = self.update_attribute(ds_stacked)
            ds_stack.append(ds_stacked)
        
        return vstack(ds_stack)
    
    
    def update_attribute(self, ds):
        
        key = self._selection.keys()[0]
        value = "-".join(self._selection[key])
        
        logger.debug(key)
        logger.debug(value)
        logger.debug(ds.shape)
        
        ds.sa[key] = [value]
        
        return ds


class DatasetMasker(Transformer):
    """
    """

    def __init__(self, 
                 mask=None, 
                 **kwargs):
        
        self._mask = mask
        Transformer.__init__(self, name='dataset_masker')    


    def transform(self, ds):

        if self._mask is None:
            self._mask = np.ones_like(ds.samples[:,0])

        return ds[self._mask]
        