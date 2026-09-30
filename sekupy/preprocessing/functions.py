from __future__ import print_function

import numpy as np

from sekupy.dataset.detrend import poly_detrend
from sekupy.dataset.dataset import Dataset, vstack, hstack

from itertools import product
from sekupy.preprocessing.base import Transformer
from sekupy.preprocessing.base import PreprocessingPipeline
from sekupy.preprocessing.slicers import SampleSlicer

from sekupy.utils.dataset import temporal_attribute_reshaping, \
    temporal_transformation

import logging
logger = logging.getLogger(__name__)


class Detrender(Transformer):
    """Detrend data.

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
        self._chunks_attr = chunks_attr
        Transformer.__init__(self, name='detrender',
                             degree=degree, chunks_attr=chunks_attr)

    def transform(self, ds):
        """Detrend the dataset.

        Parameters
        ----------
        ds : :class:`~sekupy.dataset.base.Dataset`
            The dataset to be detrended.

        Returns
        -------
        ds : :class:`~sekupy.dataset.base.Dataset`
            The detrended dataset
        """
        logger.info('Dataset preprocessing: Detrending with polynomial of order %s...', (str(self._degree)))
        ds = poly_detrend(ds, chunks_attr=self._chunks_attr, polyord=self._degree)
        return Transformer.transform(self, ds)


class SampleAverager(Transformer):
    """Init the transformer.

    Parameters
    ----------
    attributes : list
      List of sample attributes whose unique values 
      will be used to identify the samples groups.
    """

    def __init__(self, attributes):
        self._attributes = attributes

        attr_string = '.'.join(attributes)

        Transformer.__init__(self,
                             name='sample_averager',
                             attributes=attr_string)


    def transform(self, ds):
        """Average samples.

        Groups samples by the unique combination of `self._attributes`
        (sorted lexicographically) and replaces each group with its mean.

        Parameters
        ----------
        ds : :class:`~sekupy.dataset.base.Dataset`
            The dataset to be transformed.

        Returns
        -------
        ds : :class:`~sekupy.dataset.base.Dataset`
            The transformed dataset
        """
        logger.info('Dataset preprocessing: Averaging samples...')

        attrs = self._attributes
        combos = sorted(set(zip(*(ds.sa[a] for a in attrs))))

        samples = []
        sa = {a: [] for a in attrs}
        for combo in combos:
            mask = np.ones(len(ds), dtype=bool)
            for a, v in zip(attrs, combo):
                mask &= (ds.sa[a] == v)
            samples.append(ds.samples[mask].mean(axis=0))
            for a, v in zip(attrs, combo):
                sa[a].append(v)

        ds = Dataset(np.array(samples), sa=sa, fa=dict(ds.fa), a=dict(ds.a))

        return Transformer.transform(self, ds)


class SampleAttributeTransformer(Transformer):


    def __init__(self, attr=None, fx=None, **kwargs):
        """Initialize the transformer.

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
        """Initialize the transformer.

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
    """Stack features.

    This function is used to stack features with different 
    sample attribute keys, to use these features, jointly.

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

        iterable = [np.unique(ds_.sa[a]) for a in self._attr]

        ds_stack = []

        for attr in product(*iterable):
            logger.debug(attr)

            mask = np.ones_like(ds_.targets, dtype=bool)

            for i, a in enumerate(attr):
                mask = np.logical_and(mask, ds_.sa[self._attr[i]] == a)

            logger.debug(ds_[mask].shape)

            ds_stacked = hstack([d for d in ds_[mask]])
            ds_stacked = self.update_attribute(ds_stacked, ds_[mask])
            ds_stack.append(ds_stacked)

        ds = vstack(ds_stack)
        return Transformer.transform(self, ds)


    def update_attribute(self, ds, ds_orig):

        key = list(self._stack_attr)[0]
        uniques = np.unique(ds_orig.sa[key])
        value = "+".join([str(v) for v in uniques])

        logger.debug(key)
        logger.debug(value)
        logger.debug(ds.shape)

        ds.sa[key] = [value]

        return ds

class SampleTransformer(Transformer):
    """Transforms samples.

    This function is used when we need to lock SampleSlicer with
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

        temporal_attributes = ds.sa[self.attr]

        X, y = temporal_transformation(ds.samples,
                                       ds.targets,
                                       temporal_attributes)

        logger.info(X.shape)

        new_sa = {
            k: temporal_attribute_reshaping(v, temporal_attributes)
            for k, v in ds.sa.items()
        }
        ds = Dataset(X, sa=new_sa, fa=dict(ds.fa), a=dict(ds.a))

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