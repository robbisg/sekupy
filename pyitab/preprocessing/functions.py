from __future__ import print_function

import numpy as np
from mvpa2.mappers.detrend import PolyDetrendMapper
from mvpa2.mappers.fx import mean_group_sample
from mvpa2.mappers.zscore import ZScoreMapper
from mvpa2.base.dataset import vstack

import logging
from itertools import product
from mvpa2.base.dataset import hstack, vstack
from pyitab.preprocessing.base import Transformer
from pyitab.preprocessing.base import PreprocessingPipeline
from pyitab.preprocessing.slicers import SampleSlicer

from pyitab.utils.dataset import temporal_attribute_reshaping, \
    temporal_transformation
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
        ds : :class:`~mvpa2.dataset.Dataset`
            The dataset to be detrended.
        
        Returns
        -------
        ds : :class:`~mvpa2.dataset.Dataset`
            The detrended dataset
        """
        
        self.node.train(ds)

        logger.info('Dataset preprocessing: Detrending with polynomial of order %s...', (str(self._degree)))
        ds = self.node.forward(ds)
        return Transformer.transform(self, ds)
                   

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
        ds : :class:`~mvpa2.dataset.Dataset`
            The dataset to be transformed.
        
        Returns
        -------
        ds : :class:`~mvpa2.dataset.Dataset`
            The transformed dataset
        """
        logger.info('Dataset preprocessing: Averaging samples...')

        ds = ds.get_mapped(self.node)  

        return Transformer.transform(self, ds)  


class SampleAttributeTransformer(Transformer):

    
    def __init__(self, attr=None, fx=None, **kwargs):
        """[summary]

        Parameters
        ----------
        attr : [type], optional
            [description], by default None
        fx : [type], optional
            [description], by default None
        """
        self._attribute = attr
        self._fx = fx
        Transformer.__init__(self, name='target_transformer', attr=attr)
    
    def transform(self, ds):
        if self._fx is not None:
            fx = self._fx[1]
            logger.info("Dataset preprocessing: targets modified using %s" , (self._fx[0]))
            ds.sa[self._attribute] = fx(ds.sa[self._attribute])
        
        return Transformer.transform(self, ds)




class TargetTransformer(Transformer):

    
    def __init__(self, attr='targets', fx=None, **kwargs):
        """[summary]

        Parameters
        ----------
        attr : [type], optional
            [description], by default None
        fx : list or tuple, optional
            First element is the label of the fx while second is a callable that takes the 
            target vector and modifies it, by default None
        """
        self._attribute = attr
        self._fx = fx
        Transformer.__init__(self, name='target_transformer', attr=attr)
    
    def transform(self, ds):
        logger.info("Dataset preprocessing: Target set to %s" , (self._attribute))
        ds.targets = ds.sa[self._attribute]

        if self._fx is not None:
            fx = self._fx[1]
            logger.info("Dataset preprocessing: targets modified using %s" , (self._fx[0]))
            ds.targets = fx(ds.targets)
        
        return Transformer.transform(self, ds)


class FeatureStacker(Transformer):
    """This function is used to stack features with different sample attribute keys, to
    use these features, jointly.
    
    Parameters
    ----------
    stack_attr : list, optional
        This is the attribute to be used for stacking, the resulting dataset will have
        a sample attribute given by the union of unique attributes
         (the default is 'chunks')
    keep_attr : list, optional
        The attributes to keep, unique values of these attributes will be used 
        to mask the dataset. (the default is ['targets'])
    selection_dictionary : dict, optional
        This will be used to filter the dataset see ```SampleSlicer```.
    
    """

    def __init__(self, 
                 stack_attr=['chunks'],
                 keep_attr=['targets'],
                 selection_dictionary={}, 
                 **kwargs):             


        self._selection = selection_dictionary
        self._stack_attr = stack_attr
        self._attr = keep_attr
        Transformer.__init__(self, name='feature_stacker', 
                                   selection=selection_dictionary,
                                   attr=stack_attr
                                   )   

    
    def _set_mapper(self, **kwargs):

        if self._selection != {}:
            for k, v in kwargs.items():
                kwargs[k] = "+".join([str(vv) for vv in v]) 

        return Transformer._set_mapper(self, **kwargs)


    def transform(self, ds):
        
        ds_ = SampleSlicer(**self._selection).transform(ds)
  
        iterable = [np.unique(ds_.sa[a].value) for a in self._attr]
              
        ds_stack = []
        
        key = self._stack_attr[0]
        unique_stack_attr = np.unique(ds_.sa[key].value)
        
        for attr in product(*iterable):
            logger.debug(attr)
            
            mask = np.ones_like(ds_.targets, dtype=np.bool)
            
            for i, a in enumerate(attr):
                mask = np.logical_and(mask, ds_.sa[self._attr[i]].value == a)

            logger.debug(ds_[mask].shape)
            
            ds_stacked = []
            for _, k in enumerate(unique_stack_attr):
                values = ds_.sa[key].value
                mask_attr = np.logical_and(mask, values == k)
                ds_stacked.append(ds_[mask_attr])
            
            ds_stacked = hstack(ds_stacked, a='unique')
            #print(ds_stacked.shape)
            
            ds_stacked = hstack([d for d in ds_[mask]], a='unique')
            ds_stacked = self.update_attribute(ds_stacked, ds_[mask])
            ds_stack.append(ds_stacked)
        
        ds = vstack(ds_stack, a='unique')
        return Transformer.transform(self, ds)
    
    
    def update_attribute(self, ds, ds_orig):

        key = list(self._stack_attr)[0]
        uniques = np.unique(ds_orig.sa[key].value)
        value = "+".join([str(v) for v in uniques])
        
        logger.debug(key)
        logger.debug(value)
        logger.debug(ds.shape)
        
        ds.sa[key] = [value]
        
        return ds

class SampleTransformer(Transformer):
    """This function is used when we need to lock SampleSlicer with
    TargetTransformer in order to be used with AnalysisIterator.
    
    Parameters
    ----------
    attr : dictionary
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    
    def __init__(self, attr={}):

        self.key, self.value = list(attr.items())[0]
        self._pipeline = PreprocessingPipeline(nodes=[TargetTransformer(self.key),
                                                     SampleSlicer(**{self.key: self.value})])
        Transformer.__init__(self, name='sample_transformer', attr=attr)
    
    def transform(self, ds):
        ds = self._pipeline.transform(ds)
        return Transformer.transform(self, ds)


class TemporalTransformer(Transformer):
    """
    
    Parameters
    ----------
    attr : dictionary
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    
    def __init__(self, attr='frame'):

        self.attr = attr
        Transformer.__init__(self, name='temporal_transformer', attr=attr)
    
    def transform(self, ds):


        temporal_attributes = ds.sa[self.attr].value

        X, y = temporal_transformation(ds.samples, 
                                       ds.targets, 
                                       temporal_attributes)

        logger.info(X.shape)
        ds.sa.set_length_check(len(y))

        for k in ds.sa.keys():
            ds.sa[k] = temporal_attribute_reshaping(ds.sa[k].value, temporal_attributes)

        logger.info(X.shape)
        ds.samples = X

        logger.info(ds.samples.shape)

        return Transformer.transform(self, ds)

    

class Resampler(Transformer):

    def __init__(self, up=1, down=1):
        self.up = up
        self.down = down
        Transformer.__init__(self, name='resampler')

    
    def transform(self, ds):

        from mne.filter import resample
        logger.info("Resampling...")
        ds.samples = resample(ds.samples, down=self.down, up=self.up)

        logger.info("Dataset resampled "+str(ds.shape))

        return ds