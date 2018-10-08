from sklearn.pipeline import Pipeline
from sklearn.model_selection._split import GroupShuffleSplit
from sklearn.svm.classes import SVC

from pyitab.preprocessing.mapper import function_mapper
from pyitab.preprocessing.pipelines import PreprocessingPipeline
from pyitab.analysis.searchlight import SearchLight
from pyitab.io.configuration import save_configuration

import logging
logger = logging.getLogger(__name__)


class ScriptConfigurator(object):
    
  
    def __init__(self, 
                 prepro=['none'], 
                 estimator=[('clf', SVC(C=1, kernel='linear'))],
                 analysis=SearchLight,
                 cv=GroupShuffleSplit,
                 scores=['accuracy'],
                 **kwargs):

        """The configurator is used to store all the information on 
        the analysis to be performed.

        The parameters used by prepro, estimator, analysis and cv must be
        specified as in this example configuration dict:

        For example if prepro is set on ['sample_slicer'], 
        estimator is 'estimator': [('clf', SVR(C=1, kernel='linear'))]

        example_configuration = {
                                'sample_slicer__band': ['alpha'], 
                                'sample_slicer__condition' : ['vipassana'],
                                
                                ,
                                'estimator__clf__C':1,
                                'estimator__clf__kernel':'linear',
                                
                                'cv': GroupShuffleSplit,
                                'cv__n_splits': 10,
                                'cv__test_size': 0.25,
                                
                                'scores' : ['r2', 'explained_variance'],
                                
                                'analysis': SearchLight,
                                'cv_attr': 'subject'
                            
                        }


        Parameters
        ----------
        prepro : list, optional
            A list of ``Transformers`` to be performed on the dataset before the analysis.
            The list is composed by string, for more details see pyitab.preprocessing.mapper
             (the default is ['none'])
        estimator : list, optional
            The list of sklearn Estimators, to be used in the sklearn Pipeline. 
            (the default is [('clf', SVR(C=1, kernel='linear'))])
        analysis : Analyzer, optional
            Is the analysis to be performed on the dataset, specified by the class 
            that should inherit from Analyzer. 
            (the default is SearchLight)
        cv : [type], optional
            [description] (the default is GroupShuffleSplit, which [default_description])
        scores : list, optional
            [description] (the default is ['accuracy'], which [default_description])

        """

        self._default_options = {}

        self._default_options['prepro'] = prepro
        self._default_options['estimator'] = estimator
        self._default_options['analysis'] = analysis
        self._default_options['cv'] = cv
        self._default_options['scores'] = scores
        self._default_options['num'] = 1

        self._default_options.update(kwargs)

        if not 'id' in list(self._default_options.keys()):
            import uuid
            self._default_options['id'] = uuid.uuid4()

        
        
        
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
                key_split = key[idx:]
                logger.debug(key_split)
                if key_split == "%s":
                    if 'target_trans__target' in self._default_options.keys():
                        key_split = key_split %(self._default_options['target_trans__target'])
                        
                params[key_split] = self._default_options[key]
    
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

        params.update({'id': self._default_options['id']})
        params.update({'num': self._default_options['num']})
        logger.debug(params)
        return params
        
    
    
    def _get_kwargs(self):
        params = self._get_params("kwargs")
        
        if 'prepro' in params.keys():
            params['prepro'] = PreprocessingPipeline(nodes=params['prepro'])
        return params
    
    
    
    def save(self, path=None, **kwargs):
                
        if path is None:
            logger.error("path should be setted!")
            return
        
        save_configuration(path, self._default_options)
