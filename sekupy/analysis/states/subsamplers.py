import numpy as np
from scipy.signal import argrelextrema
from scipy.spatial.distance import squareform, pdist
from sekupy.preprocessing.base import Transformer
import logging

logger = logging.getLogger(__name__)

def less_equal(x1, x2):
    return np.logical_and(x1 <= x2, x1 != 0)


def greater_equal(x1, x2):
    return np.logical_and(x1 >= x2, x1 != 0)




class Subsampler(Transformer):
    
    def __init__(self, peak, order):

        mapper = {'min': less_equal,
                  'max': greater_equal}

        self.peak = mapper[peak]
                
        self.order = order
 
    
    def transform(self, ds):
        
        # Check if it has been fitted       
        arg = argrelextrema(np.array(self.measure), 
                            self.peak,
                            axis=0,
                            order=self.order)
        
        self.arg = arg
        
        ds.samples = ds.samples[arg]
        
        return ds

    


class SpeedSubsampler(Subsampler):
    
    def __init__(self, peak='min', order=5, distance='euclidean'):
        self.order = order
        self.distance = distance
        Subsampler.__init__(self, peak, order)
    
    
    def fit(self, ds):
        """
        From the data it extract the points with low local velocity 
        and returns the arguments of these points and the 
        speed for each point.    
        """
    
        subj_speed = []
        for i in range(ds.shape[0]):
            distance_ = squareform(pdist(ds.samples[i], self.distance))
            
            speed_ = [distance_[i, i+1] for i in range(distance_.shape[0]-1)]
            subj_speed.append(np.array(speed_))
        
        self.measure = np.vstack(subj_speed)
                        
        return Subsampler.transform(self, ds)
    
    

class VarianceSubsampler(Subsampler):
    
    def __init__(self, peak='max', order=5):
        Subsampler.__init__(self, peak, order)
    
    def transform(self, ds):

        ds_ = ds.copy()
        self.measure = ds_.samples.std(axis=1)
        
        return Subsampler.transform(self, ds_)            
        