from pyitab.preprocessing.base import Transformer
from pyitab.preprocessing.slicers import SampleSlicer
from mvpa2.mappers.zscore import ZScoreMapper
from mvpa2.datasets import vstack

import numpy as np

import logging
logger = logging.getLogger(__name__)


class FeatureZNormalizer(Transformer):
    
    def __init__(self, chunks_attr='chunks', param_est=None, **kwargs):
        
        self.node = ZScoreMapper(chunks_attr=chunks_attr, param_est=param_est)
        Transformer.__init__(self, name='feature_znormalizer', 
                                    chunks_attr=chunks_attr)
        
    
    def transform(self, ds):
        logger.info('Dataset preprocessing: Zscoring feature-wise...')
        self.node.train(ds)
        ds = self.node.forward(ds)
        return Transformer.transform(self, ds)
    

class SampleZNormalizer(Transformer):
    
    def __init__(self, name='sample_znormalizer', **kwargs):
        Transformer.__init__(self, name=name)       

    def transform(self, ds):
        logger.info('Dataset preprocessing: Zscoring sample-wise...')
        ds.samples -= np.mean(ds, axis=1)[:, None]
        ds.samples /= np.std(ds, axis=1)[:, None]
        
        ds.samples[np.isnan(ds.samples)] = 0
        
        return Transformer.transform(self, ds)


class SampleSigmaNormalizer(Transformer):
    
    def __init__(self, name='sample_sigma_normalizer', **kwargs):
        Transformer.__init__(self, name=name)       

    def transform(self, ds):
        logger.info('Dataset preprocessing: st. dev. normalization sample-wise...')
        ds.samples /= np.std(ds, axis=1)[:, None]
        
        ds.samples[np.isnan(ds.samples)] = 0
        
        return Transformer.transform(self, ds)


class FeatureSigmaNormalizer(Transformer):
    
    # TODO: This is for a particular variable, not the join and so on
    def __init__(self, name='sample_sigma_normalizer', attr='targets'):
        self.attr = attr
        Transformer.__init__(self, name=name, attr=attr)       

    def transform(self, ds):
        
        ds_merged = []
        for target in np.unique(ds.sa[self.attr].value):
            
            selection_dict = {self.attr: [target]}
            ds_target = SampleSlicer(**selection_dict).transform(ds)
            ds_target.samples /= np.std(ds_target, axis=0)
            logger.info('Dataset preprocessing: st. dev. normalization feature-wise...')
            
            ds_target.samples[np.isnan(ds_target.samples)] = 0
            ds_merged.append(ds_target)
        
        ds_merged = vstack(ds_merged)
        ds_merged.a.update(ds.a)
        
        return Transformer.transform(self, ds_merged)


class FeatureAttrNormalizer(Transformer):
    
    # TODO: This is for a particular variable, not the join and so on
    def __init__(self, name='sample_target_normalizer', attr_dict={'targets':'rest'}):
        self.attr, self.value = list(attr_dict.items())[0]
        Transformer.__init__(self, name=name)       

    def transform(self, ds):
        
        ds_merged = []
        selection_dict = {self.attr: [self.value]}
        baseline_ds = SampleSlicer(**selection_dict).transform(ds)
        

        for target in np.unique(ds.sa[self.attr].value):
            
            selection_dict = {self.attr: [target]}
            ds_target = SampleSlicer(**selection_dict).transform(ds)
            ds_target.samples /= np.std(ds_target, axis=0)
            logger.info('Dataset preprocessing: st. dev. normalization feature-wise...')
            
            ds_target.samples[np.isnan(ds_target.samples)] = 0
            ds_merged.append(ds_target)
        
        ds_merged = vstack(ds_merged)
        ds_merged.a.update(ds.a)
        
        return Transformer.transform(self, ds_merged)



class DatasetFxNormalizer(Transformer):
    # TODO: This can be more generic by using a lambda
    def __init__(self, name='ds_fx_normalizer', norm_fx=np.divide, ds_fx=np.std):
        
        """This class normalize the entire dataset using a function norm_fx that is used
        to normalize the dataset with respect to a number calculated on the same dataset
        using a ds_fx.
        
        Parameters
        ----------
        name : str, optional
            [description] (the default is 'ds_sigma_normalizer', which [default_description])
        norm_fx : [type], optional
            [description] (the default is np.divide, which [default_description])
        ds_fx : [type], optional
            [description] (the default is np.std, which [default_description])
        
        """

        self._ds_fx = ds_fx
        self._norm_fx = norm_fx
        Transformer.__init__(self, name=name)       

    def transform(self, ds):

        logger.info("Normalizing dataset with %s and %s" % (str(self._norm_fx), 
                                                            str(self._ds_fx)))
        
        ds.samples = self._norm_fx(ds.samples, self._ds_fx(ds.samples))
        
        return Transformer.transform(self, ds)


class SampleFxNormalizer(Transformer):
    def __init__(self, name='sample_fx_normalizer', fx=np.log):
        """This class normalize the entire dataset using a function ```fx``` that is applied
        to the whole dataset.

        Parameters
        ----------
        name : str, optional
            [description] (the default is 'ds_sigma_normalizer', which [default_description])
        fx : function, optional
            [description] (the default is np.divide, which [default_description])
        
        """

        self._fx = fx
        Transformer.__init__(self, name=name)


    def transform(self, ds):

        logger.info("Normalizing dataset with %s" % (str(self._fx)))
        
        ds.samples = self._fx(ds.samples)
        
        return Transformer.transform(self, ds)