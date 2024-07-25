import numpy as np
from sekupy.preprocessing.base import Transformer


import logging
logger = logging.getLogger(__name__)


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
    
        selection_mask = np.ones(ds.shape[1], dtype=bool)
        for key, values in selection_dict.items():
            
            logger.info("Selected %s from %s attribute.", str(values), key)
            
            ds_values = ds.fa[key].value
            condition_mask = np.zeros_like(ds_values, dtype=bool)
            
            for value in values:

                if str(value)[0] == '!':
                    array_val = np.array(value[1:]).astype(ds_values.dtype)
                    condition_mask = np.logical_or(condition_mask, ds_values != array_val)
                else:
                    condition_mask = np.logical_or(condition_mask, ds_values == value)
                    
            selection_mask = np.logical_and(selection_mask, condition_mask)
            
        
        return Transformer.transform(self, ds[:, selection_mask])



class SampleSlicer(Transformer):
    """
    Selects only portions of the dataset based on a dictionary
    The dictionary indicates the sample attributes to be used as key and a list
    with conditions to be selected:
    
    selection_dict = {
                        'frame': [1,2,3]
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
            kwargs[k] = "+".join([str(vv) for vv in v])

        return Transformer._set_mapper(self, **kwargs)



    def transform(self, ds):
        
        selection_dict = self._selection
    
        selection_mask = np.ones_like(ds.targets, dtype=bool)
        for key, values in selection_dict.items():
            
            logger.info("Selected %s from %s attribute.", str(values), key)
            
            ds_values = ds.sa[key].value
            condition_mask = np.zeros_like(ds_values, dtype=bool)
            
            for value in values:        
                condition_mask = np.logical_or(condition_mask, ds_values == value)
                
            selection_mask = np.logical_and(selection_mask, condition_mask)
            
        return Transformer.transform(self, ds[selection_mask])


class DatasetMasker(Transformer):
    """
    """

    def __init__(self,
                 mask=None,
                 **kwargs):

        self._mask = mask
        Transformer.__init__(self, name='dataset_masker')    


    def transform(self, ds, axis=0):

        if self._mask is None:
            self._mask = np.ones_like(ds.samples[:, 0])

        if axis == 0:
            ds = ds[self._mask]
        else:
            ds = ds[:, self._mask]
        
        return Transformer.transform(self, ds)


class SampleExpressionSlicer(Transformer):

    def __init__(self, attr, compare_fx=np.greater, attr_transformer=None):
        """This object is used when we want to slice samples based
        on some values and thresholds. For example if we want to 
        exclude samples from subjects with an age greater than the 
        average plus one standard deviation, or on some trials with
        an amplitude smaller than a particular value.

        Parameters
        ----------
        attr : str
            The sample attribute to use for slicing and calculating
            values.
        compare_fx : numpy function or lambda
            This function must take the sample attribute
            and a value/vector as input and return a vector of
            boolean.
        attr_transformer : funcion, optional
            This function can be used to further process the attribute
            for example the np.abs can be used, by default None
        """
        self.attr = attr
        self.compare_fx = compare_fx
        self.attr_transformer = attr_transformer

    
    def transform(self, ds, value=lambda x: np.mean(x)+1.5*np.std(x)):
        """[summary]

        Parameters
        ----------
        ds : pymvpa dataset
            The dataset to be used
        value : int or fx, optional
            The function used to generate a value to be compared
            with the attribute using the compare_fx funtion, 
            by default lambdax:np.mean(x)+1.5*np.std(x)

        Returns
        -------
        ds : pymvpa dataset
            The sliced dataset
        """

        from types import LambdaType

        compare = self.compare_fx
        attributes = ds.sa[self.attr].value.copy()

        if self.attr_transformer is not None:
            attributes = self.attr_transformer(attributes)

        if isinstance(value, LambdaType):
            value = value(attributes)

        mask = compare(attributes, value)

        ds_ = ds[mask]

        return ds_


class FeatureExpressionSlicer(Transformer):

    def __init__(self, fx=np.greater):
        # Update doc
        """This object is used when we want to slice samples based
        on some values and thresholds. For example if we want to 
        exclude features with some common characteristics, 
        for example those with nans.

        Parameters
        ----------
        attr : str
            The sample attribute to use for slicing and calculating
            values.
        compare_fx : numpy function or lambda
            This function must take the sample attribute
            and a value/vector as input and return a vector of
            boolean.
        attr_transformer : funcion, optional
            This function can be used to further process the attribute
            for example the np.abs can be used, by default None
        """

        if fx is None:
            def fx(x): np.logical_not(np.isnan(x).sum(0))

        self._fx = fx

    
    def transform(self, ds):
        """[summary]

        Parameters
        ----------
        ds : pymvpa dataset
            The dataset to be used
        value : int or fx, optional
            The function used to generate a value to be compared
            with the attribute using the compare_fx funtion, 
            by default lambdax:np.mean(x)+1.5*np.std(x)

        Returns
        -------
        ds : pymvpa dataset
            The sliced dataset
        """

        mask = self._fx(ds.samples)

        ds_ = ds[:, mask]

        return ds_
