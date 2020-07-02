
import logging
import numpy as np
from mvpa_itab.conn.operations import array_to_matrix, copy_matrix
from mvpa2.base.collections import SampleAttributesCollection
from mvpa2.datasets.base import Dataset
from pyitab.io.base import add_attributes
from pyitab.preprocessing.base import Transformer
from scipy.spatial.distance import euclidean

logger = logging.getLogger(__name__)

# TODO: Document better!
class SingleRowMatrixTransformer(Transformer):
    """This transformer change a row dataset representing a matrix
    in a square matrix dataset.       
    """

    def __init__(self, name='upper_matrix', **kwargs):

        Transformer.__init__(self, name=name, **kwargs)  


    def transform(self, ds):
        """This function performs the transformation into
        a square matrix dataset
        
        Parameters
        ----------
        ds : pymvpa Dataset
            The dataset to be transformed.
        
        Returns
        -------
        ds : pymvpa Dataset
            The transformed dataset
        """

                
        data = np.dstack([copy_matrix(array_to_matrix(a)) for a in ds.samples])
        data = np.hstack([d for d in data[:,:]]).T
        
        attr = self._edit_attr(ds, data.shape)
        
        ds_ = Dataset.from_wizard(data)
        ds_ = add_attributes(ds_, attr)
                
        return Transformer.transform(self, ds_)
        
        
        
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
    


class SpeedEstimator(Transformer):
    # TODO: Is it a transformer or an estimator??

    def __init__(self, name='speed_tc', **kwargs):
        Transformer.__init__(self, name=name, **kwargs) 

    def transform(self, ds):

        ds_ = ds.copy()

        trajectory = [euclidean(ds.samples[i+1], ds.samples[i]) 
                         for i in range(ds.shape[0]-1)]

        ds_.samples = np.array(trajectory)

        return Transformer.transform(self, ds_)
    


class AverageEstimator(Transformer):

    def __init__(self, name='average_estimator', **kwargs):
        Transformer.__init__(self, name=name, **kwargs)

    def transform(self, ds):

        ds_ = ds.copy()

        ds_.samples = np.mean(ds_.samples, axis=1, keepdims=True)
        
        return Transformer.transform(self, ds_)