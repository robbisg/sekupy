from pyitab.analysis.base import Analyzer
from pyitab.utils.math import z_fisher, partial_correlation
from pyitab.preprocessing.connectivity import SpeedEstimator
from pyitab.preprocessing.base import Transformer
from pyitab.preprocessing import FeatureSlicer
from pyitab.analysis.utils import get_rois
from nitime.analysis import CorrelationAnalyzer
from nitime.timeseries import TimeSeries
from scipy.stats import zscore
from scipy.io import savemat

import numpy as np
import os
import logging
logger = logging.getLogger(__name__)

class TrajectoryConnectivity(Analyzer):
    

    def __init__(self, name='mvfc', **kwargs):
        """[summary]

        Parameters
        ----------
        name : str, optional
            [description], by default 'mvfc'
        """
        
        
        Analyzer.__init__(self, name, **kwargs)


    def fit(self,
            ds,
            roi='all',
            roi_values=None,
            estimator=SpeedEstimator(),
            use_partialcorr=False,
            ):
        """Fits the multivariate connectivity of the dataset.

        Parameters
        -----------
    
        ds : PyMVPA dataset
            The dataset to be used to fit the data
    
        cv_attr : string. Default is 'chunks'.
            The attribute to be used to separate data in the cross validation.
            If cv attribute is specified this parameter is ignored.
            
    
        roi : list of strings. Default is 'all'
            The list of rois to be selected for the analysis. 
            Each string must correspond to a key in the dataset feature attributes.

            
        roi_values : list of tuple, optional. Default is None
            The list of tuple must have as first element the name of roi to be used,
            which should be in the feature attribute of the dataset.
            The second element of the tuple must be a list of values, corresponding to
            the value of the specific roi 
            (e.g. roi_values = [('lateral_ips', [2,4,6]), ('left_precuneus', [10,12])] 
             performs two analysis on lateral_ips and left_precuneus with the
             union of rois with values of 2,4,6 and 10,12 )
             
             
        prepro : Node or PreprocessingPipeline implementing transform, optional.
            A transformation of series of transformation to be performed
            before the decoding analysis is performed.
        
        """


        if roi_values == None:
            roi_values = get_rois(ds, roi)
                
        self._tc = dict()

        if use_partialcorr:
            self._tc['full_brain'] = estimator.transform(ds)
        
        for r, value in roi_values:
            
            ds_ = FeatureSlicer(**{r:value}).transform(ds)
            
            logger.info("Dataset shape %s" % (str(ds_.shape)))
            
            # TODO: Fit or transform?
            roi_timecourse = estimator.transform(ds_)
            
            string_value = "_".join([str(v) for v in value])
            self._tc["%s_%s" % (r, string_value)] = roi_timecourse.samples

        self.scores = self._fit(self._tc, use_partialcorr)

        self._info = self._store_info(ds, 
                                      roi=roi,
                                      prepro=estimator)
        
        return self


    def _fit(self, tc, use_partialcorr):
                 
        if use_partialcorr:
            X = np.array([t for k, t in tc.items() if k != 'full_brain'])
            Z = np.expand_dims(tc['full_brain'].samples, axis=0)
            return partial_correlation(X, Z)

        timecourses = [t for k, t in tc.items()]

        ts = TimeSeries(timecourses, sampling_interval=1.)
        C = CorrelationAnalyzer(ts)
        
        matrix = C.corrcoef
        matrix[np.isinf(matrix)] = 1.
        matrix[np.isnan(matrix)] = 1.

        return matrix


    def save(self, path=None, **kwargs):

        # TODO: Demean / minus_chance
        path = Analyzer.save(self, path, attributes=[])

        data = dict()

        data['timecourses'] = self._tc
        data['matrix'] = self.scores
        data['labels'] = list(self._tc.keys())

        filename = "connectivity_data.mat" 
        logger.info("Saving %s" %(filename))
        
        savemat(os.path.join(path, filename), data)