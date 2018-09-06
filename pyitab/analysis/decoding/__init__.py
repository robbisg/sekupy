import logging
import os
from mvpa_itab.pipeline import Analyzer
from mvpa_itab.results import get_time, make_dir
from collections import Counter
logger = logging.getLogger(__name__)

#from guppy import hpy


class AnalysisPipeline(Analyzer):
    
    def __init__(self, configurator, name="base", **kwargs):
        
        self._configurator = configurator
        self._name = name
        
        
    
    def fit(self, ds, **kwargs):
        
        self._transformer, self._estimator = self._configurator.fit()

        ds_ = self._transform(ds)
        
        self._estimator.fit(ds_, **kwargs)     
        
        #logger.debug(hpy().heap())
        
        return self
    
    
    def save(self):
        

        params = self._configurator._get_fname_info()
        params.update(self._estimator._get_fname_info())
        
        
        logger.info(params)
        
        path = params.pop("path")
        dir_ = "%s_%s_%s_%s_%s" %(
                                  get_time(),
                                  self._name,
                                  params.pop("analysis"),
                                  params.pop("experiment"),
                                  "_".join(["%s_%s" %(k, v) for k, v in params.items()])
                                  )
                               
        full_path = os.path.join(path, "0_results", dir_)
        make_dir(full_path)
        
        # Save results
        self._configurator.save(path=full_path)
        self._estimator.save(path=full_path)        
        
        return
    
    
    def _transform(self, ds):
        
        self._configurator._default_options['ds__target_count_pre'] = Counter(ds.targets)
        self._configurator._default_options['ds__preprocessing'] = ds.a['prepro']
        
        for node in self._transformer.nodes:
            ds = node.transform(ds)
            if node.name in ['balancer', 'target_transformer']:
                key = 'ds__target_count_%s' % (node.name)
                self._configurator._default_options[key] = Counter(ds.targets)
        
        return ds
        
        
    