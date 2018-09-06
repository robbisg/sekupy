from mvpa_itab.preprocessing.functions import Node
from scipy.spatial.distance import euclidean
from mvpa_itab.pipeline.searchlight import get_seeds

import numpy as np

from mvpa_itab.io.utils import get_ds_data

import logging
from mvpa2.datasets.base import Dataset
from mvpa_itab.pipeline.connectivity.utils import get_roi_adjacency,\
    _parallel_trajectory
logger = logging.getLogger(__name__)



class TrajectoryTransformer(Node):

    
    def __init__(self, dist_fx=euclidean, radius=3.0, kind='voxelwise', n_jobs=-1, verbose=1, **kwargs):
        self.dist = dist_fx
        self.radius = radius
        self.kind = kind
        self.n_jobs = n_jobs
        self.verbose = verbose
        Node.__init__(self, name='trajectory_transformer')
                
    
    
    def _get_adjacency(self, ds, rois):
        
        if self.kind=='voxelwise':
            return get_seeds(ds, self.radius), np.arange(ds.shape[1])
        
        return get_roi_adjacency(ds, rois)
    
        
    def transform(self, ds, rois=None):
        
        logger.info("Building adjacency...")
        A, roi_list = self._get_adjacency(ds, rois)

        dist = self.dist
        
        X, _ = get_ds_data(ds)
        
        logger.info("Applying trajectory transformation to dataset...")
        samples = _parallel_trajectory(X, dist, A, self.n_jobs, self.verbose)
        

        ds_ = Dataset.from_wizard(samples)
        ds_.fa['roi_indices'] = roi_list

        return ds_
    
    
    


    