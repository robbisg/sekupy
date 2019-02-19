from pyitab.analysis.base import Analyzer
from pyitab.utils.math import z_fisher, partial_correlation
from pyitab.preprocessing.connectivity import SpeedEstimator
from pyitab.preprocessing.functions import FeatureSlicer, Transformer
from nitime.analysis import CorrelationAnalyzer
from nitime.timeseries import TimeSeries
from scipy.stats import zscore
from scipy.io import savemat

import numpy as np
import os
import logging
logger = logging.getLogger(__name__)

class TrajectoryConnectivity(Analyzer):
    # TODO: Function documentation

    def __init__(self, name='mvfc', tc_estimator=SpeedEstimator(), **kwargs):
        
        self.tc_estimator = tc_estimator
        Analyzer.__init__(self, name, **kwargs)


    def fit(self,
            ds,
            cv_attr='chunks',
            roi='all',
            roi_values=None,
            prepro=Transformer(),
            use_partialcorr=False,
            ):
        """Fits the connectivity of the dataset.

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
            roi_values = self._get_rois(ds, roi)
                
        self._tc = dict()

        if use_partialcorr:
            self._tc['full_brain'] = self.tc_estimator.transform(ds)
        
        for r, value in roi_values:
            
            ds_ = FeatureSlicer(**{r:value}).transform(ds)
            ds_ = prepro.transform(ds_)
            
            logger.info("Dataset shape %s" % (str(ds_.shape)))
            
            # TODO: Fit or transform?
            roi_timecourse = self.tc_estimator.transform(ds_)
            
            string_value = "_".join([str(v) for v in value])
            self._tc["%s_%s" % (r, string_value)] = roi_timecourse.samples
        


        self.scores = self._fit(self._tc, use_partialcorr)

        self._info = self._store_ds_info(ds, 
                                         cv_attr=cv_attr,
                                         roi=roi,
                                         prepro=prepro)
        
        return self
    
    
    
    def _get_rois(self, ds, roi):
        """Gets the roi list if the attribute is all"""
        
        rois = [r for r in ds.fa.keys() if r != 'voxel_indices']
        
        if roi != 'all':
            rois = roi
        
        rois_values = []
        
        for r in rois:
            value = [(r, [v]) for v in np.unique(ds.fa[r].value) if v != 0]
            rois_values.append(value)
            
        return list(*rois_values)  


    def _fit(self, tc, use_partialcorr):
                 
        if use_partialcorr:
            X = np.array([zscore(t) for k, t in tc.items() if k != 'full_brain'])
            Z = np.expand_dims(zscore(tc['full_brain'].samples), axis=0)
            return partial_correlation(X, Z)

        timecourses = [zscore(t) for k, t in tc.items()]

        ts = TimeSeries(timecourses, sampling_interval=1.)
        C = CorrelationAnalyzer(ts)
        
        matrix = z_fisher(C.corrcoef)
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