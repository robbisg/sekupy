import logging
logger = logging.getLogger(__name__)

_default_options = {
       
                       'sampleslicer__condition' : [['vipassana'], ['samatha']],
                       'estimator__svr__C': [1, 10],                          
                       'cv__n_splits': [10, 15],
                       'analysis__radius':[9,18],
                           
                        }
     

class AnalysisIterator(object):

    
    def __init__(self, options, configurator, kind='combination'):
        """This class allows to configure different analysis to be
        iterated using a set of options.
                
        Parameters
        ----------
        options : dict | list of dictionaries
            A dictionary that include all different values that must be 
            iterated in the analysis to be performed.
                        
        configurator : [type]
            [description]

        kind : str
            Indicates the type of datum given in options field.
            (values must be 'combination', 'list' or 'configuration')
            if 'combination' all possible combination of items in options will be performed
            as a cartesian product of lists.
            if 'list' elements of dictionary lists must have the same length
            if 'configuration' the elements are single configuration to be used
            'combination' or 'list' or 'configurations'
        
        """
        
        if kind == 'list':
            fx = self._list_setup
        elif kind == 'configuration':
            fx = self._configuration_setup
            list_opt = options[:]
            options = dict()
            options['options'] = list_opt
        else:
            fx = self._setup
    
        self.configurations, self.i, self.n = fx(**options)
        self._configurator = configurator
        import uuid
        self._id = uuid.uuid4()


    def _configuration_setup(self, **kwargs):
        # TODO: Check kwargs
        configurations, i, n = kwargs['options'], 0, len(kwargs['options']) 
        return configurations, i, n


    def _list_setup(self, **kwargs):
        
        import itertools
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


   
    def _setup(self, **kwargs):
        
        import itertools
            
        args = [arg for arg in kwargs]
        logger.info(kwargs)
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
            logger.info("Iteration %d/%d" %(self.i, self.n))
            self._configurator.set_params(**value)
            self._configurator.set_params(id=self._id)
            self._configurator.set_params(num=self.i)
        else:
            raise StopIteration()
        
        return self._configurator
    
    
    