

import numpy as np
from sekupy.utils.matrix import array_to_matrix, copy_matrix
from sekupy.io.base import add_attributes
from sekupy.preprocessing.base import Transformer

from scipy.spatial.distance import pdist
from scipy import signal

from sekupy.dataset.collections import SampleAttributesCollection
from sekupy.dataset.base import Dataset

import itertools

import logging
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
        data = np.hstack([d for d in data[:, :]]).T

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

    def transform(self, ds, metric='euclidean'):

        ds_ = ds.copy()

        X = ds.samples

        trajectory = [pdist(np.vstack((X[i+1], X[i])), metric) \
                         for i in range(ds.shape[0]-1)]

        ds_.samples = np.array(trajectory)

        return Transformer.transform(self, ds_)
    


class AverageEstimator(Transformer):

    def __init__(self, name='average_estimator', **kwargs):
        Transformer.__init__(self, name=name, **kwargs)

    def transform(self, ds):
        ds_ = ds.copy()

        ds_.samples = np.mean(ds_.samples, axis=1, keepdims=True)

        return super().transform(ds)()



class SlidingWindowConnectivity(Transformer):


    def __init__(self, window_length=1, shift=1, overlap=0):
        # Units are in seconds ?
        self.window_length = window_length
        self.shift = shift
        self.overlap = overlap
        Transformer.__init__(self, name='sliding_window')


    def transform(self, ds):
        """Connectivity"""

        # TODO: Replace with get_data_ds
        data = ds.samples
        n_edges = int(ds.shape[1] * (ds.shape[1] - 1) * 0.5)
        edges = [e for e in itertools.combinations(np.arange(ds.shape[1]), 2)]

        window_length = self.window_length * ds.a.sample_frequency
        window_start = np.arange(0, (data.shape[0] - window_length + 1), 
                                 self.shift)

        connectivity_lenght = len(window_start)
        timewise_connectivity = np.zeros((connectivity_lenght, n_edges))

        for w in window_start:

            data_window = data[w:w+window_length, :]

            # From here must be included in a function.
            phi = np.angle(signal.hilbert(data_window, axis=0))

            for e, (x, y) in enumerate(edges):
                coh = np.imag(np.exp(1j*(phi[:, x] - phi[:, y])))
                iplv = np.abs(np.mean(coh))
                timewise_connectivity[w, e] = iplv

        ds.samples = timewise_connectivity
        ds = self.update_ds(ds, window_start)

        return ds


    def update_ds(self, ds, windows_start):
        sa = {}
        for k in ds.sa.keys():
            sa.update({k: ds.sa[k].value[windows_start]})

        ds_ = Dataset(ds.samples, sa=sa, a=ds.a)

        return ds_

