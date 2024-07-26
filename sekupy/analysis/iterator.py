from sekupy.utils import get_id
import numpy as np
import logging
import itertools
from sekupy.analysis.utils import get_params
logger = logging.getLogger(__name__)


class AnalysisIterator(object):


    def __init__(self, options, configurator, kind='combination', config_kwargs={}):
        """This class allows to configure different analysis to be
        iterated using a set of options.

        Options can have different formats (see ```kind``` attributes).
        - ```combination```:
            _default_options = 
                    {'sample_slicer__condition' : [['vipassana'], ['samatha']],
                    'estimator__svr__C': [1, 10],                          
                    'cv__n_splits': [10, 15],
                    'analysis__radius': [9,18]}

        Above config runs every combination of each single dictionary entry.
        
        - ```configurations```:
            _default_options = [
                {'sample_slicer__condition' : ['vipassana'],
                'estimator__svr__C': 1, 
                'cv__n_splits': 10,
                'analysis__radius': 9},
                {'sample_slicer__condition' : ['samatha'],
                'estimator__svr__C': 10,
                'cv__n_splits': 15,
                'analysis__radius': 18}
            ]
        
        This runs two analyses, with configurations specified in elements of the list.

        - ```list```
            _default_options = 
                    {'sample_slicer__condition' : [['vipassana'], ['samatha']],
                    'estimator__svr__C': [1, 10],                          
                    'cv__n_splits': [10, 15],
                    'analysis__radius': [9,18]}
        
        In this case, two configurations are built. The first with elements in 
        position 0 of each list and the other with elements in position 1. 
        The result is the same as the example with ```configurations```.
        Elements in the lists must be equal otherwise combination will be performed.

     
                
        Parameters
        ----------
        options : dict | list of dictionaries
            A dictionary that include all different values that must be 
            iterated in the analysis to be performed.
                        
        configurator : [type]
            [description]

        kind : str | 'combination' or 'list' or 'configurations'
            Indicates the type of datum given in options field.
            (values must be 'combination', 'list' or 'configuration')
            if 'combination' all possible combination of items in options will be performed
            as a cartesian product of lists.
            if 'list' elements of dictionary lists must have the same length
            if 'configuration' the elements are single configuration to be used
            
        
        """
        
        if kind == 'list':
            fx = self._list_setup
        elif kind == 'configuration':
            fx = self._configuration_setup
            list_opt = options[:]
            options = dict()
            options['options'] = list_opt
        elif kind == 'combined':
            fx = self._combined_setup
        else:
            fx = self._setup
        
        self.n_subjects = 1

        self.configurations, self.i, self.n = fx(**options)
        self._configurator = configurator
        self._config_params = config_kwargs
        self._id = get_id()
        logger.info("No. of iterations: %s" % (str(self.n)))
        


    def _configuration_setup(self, **kwargs):
        # TODO: Check kwargs
        configurations, i, n = kwargs['options'], 0, len(kwargs['options']) 
        return configurations, i, n


    def _list_setup(self, **kwargs):
        
        
        set_ = set([len(value) for key, value in kwargs.items()])

        if len(set_) != 1:
            return self._setup(**kwargs)

        n_elements = set_.pop()

        keys = list(kwargs.keys())

        configurations = []
        for i in range(n_elements):
            conf = dict()
            for key in keys:
                conf[key] = kwargs[key][i]

            configurations.append(conf)
            
        i = 0
        n = len(configurations)
        
        return configurations, i, n


    def _combined_setup(self, **kwargs):
        # This may be replaced by _build_combinations
        kwargs = self._check_subject_keywords(**kwargs)
        logger.debug(kwargs)
        return self._build_combinations(key='estimator', **kwargs)
        

    def _build_combinations(self, key='estimator', **kwargs):
        keyword_list = {k: v for k, v in kwargs.items() if k.find(key) != -1}
        excluded_params = {k: v for k, v in kwargs.items() if k.find(key) == -1}

        objects = keyword_list.pop(key)
        
        params = get_params(keyword_list, key)
        logger.debug(params)
        
        combination_list = []
        for obj in objects:
            # This is the line that breaks tests!!!
            name = obj[0][0]
            logger.debug(obj)
            
            est_params = get_params(params, name)
            logger.debug(est_params)

            if len(est_params) == 0:
                combination_list += [{'estimator': obj}]
                continue
            
            param_list = [v for k, v in est_params.items()]
            
            list_ = list(itertools.product(obj, *param_list))
            keys_ = ['estimator']
            keys_ += ["estimator__%s__%s" % (name, p) for p in est_params.keys()]
            combination_list += [dict(zip(keys_, elem)) for elem in list_]
        
        excluded_list, _, _ = self._setup(**excluded_params)

        configurations = []

        for option_excluded in excluded_list:
            for option_included in combination_list:
                keys = list(option_excluded.keys()) + list(option_included.keys())
                values = list(option_excluded.values()) + list(option_included.values())
                configurations.append(dict(zip(keys, values)))

        return configurations, 0, len(configurations)


    def _check_subject_keywords(self, **kwargs):
        import itertools
        subject_keys = ['sample_slicer__subject', 
                        'fetch__subject_names']

        for k in subject_keys:
            if k in kwargs.keys():
                v = kwargs.pop(k)
                kwargs[k] = v
                self.n_subjects = len(v)
            
        logger.debug(self.n_subjects)

        return kwargs


   
    def _setup(self, **kwargs):
        
        kwargs = self._check_subject_keywords(**kwargs)

        args = [arg for arg in kwargs]
        logger.debug(kwargs)
        combinations_ = list(itertools.product(*[kwargs[arg] for arg in kwargs]))
        configurations = [dict(zip(args, elem)) for elem in combinations_]
        i = 0
        n = len(configurations)
        
        return configurations, i, n
        
   
    def __iter__(self):
        return self


    def next(self):
        return self.__next__()
    
    
    def __next__(self):
        
        if self.i < self.n:
            value = self.configurations[self.i]
            self.i += 1
            logger.info("Iteration %d/%d" % (self.i, self.n))

            configurator = self._configurator(**self._config_params)

            configurator.set_params(**value)
            
            configurator.set_params(id=self._id)

            num = self.i
            
            #if self.n_subjects > 1:
            #    num = np.floor((self.i-1) / self.n_subjects) + 1
            #    logger.debug(num)
            configurator.set_params(num=int(num))
        
        else:
            raise StopIteration()
        
        return configurator
    
    
    