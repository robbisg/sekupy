from mvpa_itab.preprocessing.functions import FeatureSlicer
from sklearn.metrics.scorer import _check_multimetric_scoring
import numpy as np
from sklearn.svm import SVC
from mvpa_itab.io.utils import get_ds_data
from sklearn.preprocessing.label import LabelEncoder
from sklearn.model_selection._split import LeaveOneGroupOut
from sklearn.model_selection._validation import cross_validate


import logging
from mvpa_itab.pipeline import Analyzer, Transformer
from scipy.io.matlab.mio import savemat
from nitime.analysis.correlation import CorrelationAnalyzer
from nitime.timeseries import TimeSeries
logger = logging.getLogger(__name__)


class ConnectivityAnalysis(Analyzer):
    """Implement decoding analysis using an arbitrary type of classifier.

    Parameters
    -----------

    estimator : 'svr', 'svc', or an estimator object implementing 'fit'
        The object to use to fit the data

   
    Attributes
    -----------

    scores : dict.
            The dictionary of results for each roi selected.
            The key is the union of the name of the roi and the value(s).
            The value is a list of values, the number is equal to the permutations.
            
    """

    def __init__(self, estimator=CorrelationAnalyzer, score='corrcoef'):
        
        self.estimator = estimator
        self.score = score
        
        Analyzer.__init__(self, name='connectivity')
        


    def fit(self, ds):
        """Fits the decoding of the dataset.

        Parameters
        -----------
    
        ds : PyMVPA dataset
            The dataset to be used to fit the data
            
        """

        ts = TimeSeries(ds.samples.T, sampling_interval=np.float(ds.a.tr))
        C = self.estimator(ts)
        self._score = getattr(C, self.score)
        
        return self
    
    
    
    
    def save(self, path=None, full_save=False):
        
        return
        
        
        
        
        
        
        
        