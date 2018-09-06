from mvpa_itab.pipeline import Transformer
from mvpa2.mappers.fx import mean_group_feature, mean_sample, FxMapper
from mvpa_itab.preprocessing.functions import FeatureSlicer, SampleSlicer
import numpy as np
from itertools import product
from mvpa2.base.dataset import vstack, hstack
from mvpa_itab.preprocessing.pipelines import PreprocessingPipeline
from sklearn.decomposition.pca import PCA
from scipy.spatial.distance import euclidean




class VoxelAverager(Transformer):
    
    def __init__(self, roi='all', name='voxel_averager', **kwargs):
        
        self._roi = roi
        Transformer.__init__(self, name=name, **kwargs)
            
        
        
    def transform(self, ds):
        
        if self._roi == 'all':
            self._roi = [roi for roi in ds.fa.keys() if roi != 'voxel_indices']
        
        
        return mean_group_feature(self._roi).forward(ds)
    
    
    
    
    
    
class PCAVoxelTransformer(Transformer):
    
    
    def __init__(self, roi='all', name='pca_transformer', **kwargs):
        
        self._roi = roi
        Transformer.__init__(self, name=name, **kwargs)
            
            
    
    def _pca(self, axis, data):
        data = axis
        return PCA(n_components=1).fit_transform(data)
    
    
    
    def transform(self, ds):
        
        if self._roi == 'all':
            self._roi = [roi for roi in ds.fa.keys() if roi != 'voxel_indices']

        return FxMapper(axis='features', uattrs=self._roi, fx=self._pca).forward(ds)
        




class VelocityTransformer(Transformer):
    
    
    def __init__(self, roi='all', name='velocity', distance_fx=euclidean, **kwargs):
        
        self._roi = roi
        self._distance = distance_fx
        Transformer.__init__(self, name=name, **kwargs)
            
            
    
    def _fx(self, axis, data):
        
        data = axis
        dist = self._distance
        
        velocity = np.array([dist(data[i+1], data[i]) for i in range(data.shape[0]-1)])
        
        return velocity
    
    
    
    def transform(self, ds):
        
        if self._roi == 'all':
            self._roi = [roi for roi in ds.fa.keys() if roi != 'voxel_indices']

        return FxMapper(axis='features', uattrs=self._roi, fx=self._fx).forward(ds)    
    
    
    
    
    
    
    
    