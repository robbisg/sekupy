from sklearn.pipeline import Pipeline
from sklearn.model_selection._split import GroupShuffleSplit
from sklearn.svm.classes import SVC
from pyitab.preprocessing.mapper import function_mapper
from pyitab.preprocessing.pipelines import PreprocessingPipeline
from pyitab.analysis.searchlight import SearchLight
from pyitab.io.configuration import save_configuration
from pyitab.io.loader import DataLoader
from pyitab.analysis.utils import get_params
from pyitab.utils import get_id

import logging
logger = logging.getLogger(__name__)



class AnalysisConfigurator(object):
    
  
    def __init__(self, 
                 prepro=['none'], 
                 estimator=[('clf', SVC(C=1, kernel='linear'))],
                 analysis=SearchLight,
                 cv=GroupShuffleSplit,
                 #scores=['accuracy'],
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

                                'loader': DataLoader,
                                'loader__conf_file':"/media/robbis/DATA/fmri/working_memory/working_memory.conf",
                                'loader__loader':'simulations',
                                'loader__task':'simulations'
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
        # TODO: If we are not classifying?
        self._default_options['estimator'] = estimator
        self._default_options['analysis'] = analysis
        self._default_options['cv'] = cv
        #self._default_options['scores'] = scores
        self._default_options['num'] = 1

        self._default_options.update(kwargs)

        if not 'id' in list(self._default_options.keys()):
            self._default_options['id'] = get_id()

        
    def fit(self):

        objects = {}
        objects['loader'] = self._get_loader()
        objects['transformer'] = self._get_transformer()
        objects['estimator'] = self._get_analysis()
        
        return objects
    
    
    def set_params(self, **params):
        logger.debug(params)

        if 'estimator' in params.keys():
            keys = list(self._default_options.keys()).copy()
            for k in keys:
                if k.find('estimator') != -1:
                    _ = self._default_options.pop(k)

        self._default_options.update(params)
    
    
    def _get_params(self, keyword):
        return get_params(self._default_options, keyword)


    def _get_loader(self):
        
        klass = DataLoader
        if 'loader' in self._default_options.keys():
            klass = self._default_options['loader']

        params = self._get_params("loader")

        if params == {}:
            return None
        
        return klass(**params)
    
    
    def _get_transformer(self):
        
        transformer = []
        for key in self._default_options['prepro']:
            class_ = function_mapper(key)
            
            arg_dict = self._get_params(key)

            if key == 'sample_slicer' and 'attr' in arg_dict.keys():
                arg_dict = arg_dict['attr']

            object_ = class_(**arg_dict)
            transformer.append(object_)
        
        logger.debug(transformer)
        return PreprocessingPipeline(nodes=transformer)
    
    
    def _get_estimator(self):

        estimator = Pipeline(steps=self._default_options["estimator"])
        
        params = self._get_params("estimator")
        _ = estimator.set_params(**params)
        
        return estimator
    
    
    
    def _get_cross_validation(self):
        
        cv_params = self._get_params("cv")
        cross_validation = self._default_options['cv']

        logger.debug(cross_validation)
        logger.debug(cv_params)
                
        return cross_validation(**cv_params)
    
    
    
    def _get_analysis(self):
        
        params = self._get_params("analysis")
        params['id'] = self._default_options['id']
        params['num'] = self._default_options['num']
        
        keys = list(self._default_options.keys())
        if 'estimator' in keys:
            params['estimator'] = self._get_estimator()
        
        if 'cv' in keys:
            params['cv'] = self._get_cross_validation()
        
        analysis = self._default_options['analysis']
        logger.debug(params)
        
        return analysis(**params)
  
    
    
    def _get_fname_info(self):
        
        params = dict()
        
        for keyword in ["sample_slicer", "target_transformer", "sample_transformer"]:

            if keyword in self._default_options['prepro']:
                
                params_ = self._get_params(keyword)
                if keyword == "sample_slicer":
                    params_ = {k: "+".join([str(v) for v in value]) for k, value in params_.items()}

                if keyword == "sample_transformer":
                    params_ = {k: "+".join([str(v) for v in value]) for k, value in params_['attr'].items()}

                params.update(params_)

        params.update({'id': self._default_options['id']})
        params.update({'num': self._default_options['num']})
        logger.debug(params)
        return params
        

    def _get_function_kwargs(self, function='fit'):
        params = self._get_params(function)
        
        if 'prepro' in params.keys():
            params['prepro'] = PreprocessingPipeline(nodes=params['prepro'])
        
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


    def __str__(self):
        line = ""
        for k, v in self._default_options.items():
            line += "\n%s\t\t%s" %(k, str(v))
        
        line += "\n"
        return line