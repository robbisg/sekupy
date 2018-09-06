from sklearn.pipeline import Pipeline
from mvpa_itab.pipeline.searchlight import SearchLight
from sklearn.model_selection._split import GroupShuffleSplit
from sklearn.svm.classes import SVR
from mvpa_itab.preprocessing.mapper import function_mapper
from mvpa_itab.preprocessing.pipelines import PreprocessingPipeline

import logging
from mvpa_itab.io.configuration import save_configuration
logger = logging.getLogger(__name__)


class ScriptConfigurator(object):
    


    
    def __init__(self, **kwargs):
        
        self._default_options = {
               
                                   'prepro':['sampleslicer', 'featurenorm', 'targettrans'],
                                   #'sampleslicer__band': ['alpha'], 
                                   #'sampleslicer__condition' : ['vipassana'],
                                   #'targettrans__target':"expertise_hours",
                                   
                                   'estimator': [('clf', SVR(C=1, kernel='linear'))],
                                   'estimator__clf__C':1,
                                   'estimator__clf__kernel':'linear',
                                   
                                   'cv': GroupShuffleSplit,
                                   #'cv__n_splits': 10,
                                   #'cv__test_size': 0.25,
                                   
                                   #'scores' : ['r2', 'explained_variance'],
                                   
                                   'analysis': SearchLight,
                                   #'cv_attr': 'subject'
                                   
                                }
        

        self._default_options.update(kwargs)
        
        
       
    
        
    def fit(self):
        
        return self._get_transformer(), self._get_analysis()
    
    
    def set_params(self, **params):
        logger.debug(params)
        self._default_options.update(params)
    
    
    
    def _get_params(self, keyword):
        
        params = dict()
        for key in self._default_options.keys():
            idx = key.find(keyword)
            if idx == 0 and len(key.split("__")) > 1:
                idx += len(keyword)+2
                params[key[idx:]] = self._default_options[key]
    
        logger.debug("%s %s" % (keyword, str(params)))
        return params
    
    
    def _get_transformer(self):
        
        transformer = []
        for key in self._default_options['prepro']:
            class_ = function_mapper(key)
            
            arg_dict = self._get_params(key)
                
            object_ = class_(**arg_dict)
            transformer.append(object_)
        
        logger.debug(transformer)
        return PreprocessingPipeline(nodes=transformer)
    
    
    
    def _get_estimator(self):
       
        estimator = Pipeline(steps=self._default_options["estimator"])
        
        params = self._get_params("estimator")
       
        estimator.set_params(**params)
        
        return estimator
    
    
    
    def _get_cross_validation(self):
        
        cv_params = self._get_params("cv")
        cross_validation = self._default_options['cv']
                
        return cross_validation(**cv_params)
    
    
    
    def _get_analysis(self):
        
        params = self._get_params("analysis")
        params['estimator'] = self._get_estimator()
        params['cv'] = self._get_cross_validation()
        params['scoring'] = self._default_options['scores']
        
        analysis = self._default_options['analysis']        

        
        return analysis(**params)      
    
    
    
    def _get_fname_info(self):
        
        params = dict()
        
        for keyword in ["sample_slicer", "target_trans"]:
            if keyword in self._default_options['prepro']:
                
                params_ = self._get_params(keyword)
                if keyword == "sample_slicer":
                    params_ = {k:"_".join([str(v) for v in value]) for k, value in params_.items()}
                    
                    
                params.update(params_)

            
        logger.debug(params)
        return params
        
    
    
    def _get_kwargs(self):
        params = self._get_params("kwargs")
        
        if 'prepro' in params.keys():
            params['prepro'] = PreprocessingPipeline(nodes=params['prepro'])
        return params
    
    
    
    def save(self, path=None, **kwargs):
                
        if path == None:
            logger.error("path should be setted!")
        
            return
        
        
        save_configuration(path, self._default_options)
        




        
        
        
        