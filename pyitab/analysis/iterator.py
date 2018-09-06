import logging
logger = logging.getLogger(__name__)

_default_options = {
       
                       'sampleslicer__condition' : [['vipassana'], ['samatha']],
                       'estimator__svr__C': [1, 10],                          
                       'cv__n_splits': [10, 15],
                       'analysis__radius':[9,18],
                           
                        }
     
    


class AnalysisIterator(object):
    
    
    
    def __init__(self, options, configurator, name=None):
        
        
        self._configurations = self._setup(**options)
        self._configurator = configurator

   
    
    
    
    def _setup(self, **kwargs):
        
        import itertools
            
        args = [arg for arg in kwargs]
        logger.info(kwargs)
        combinations_ = list(itertools.product(*[kwargs[arg] for arg in kwargs]))
        self.configurations = [dict(zip(args, elem)) for elem in combinations_]
        self.i = 0
        self.n = len(self.configurations)

    
    
    def __iter__(self):
        return self
    
        
    
    def next(self):
        
        if self.i < self.n:
            value = self.configurations[self.i]
            self.i += 1
            logger.info("Iteration %d/%d" %(self.i, self.n))
            self._configurator.set_params(**value)
            return self._configurator
        else:
            raise StopIteration()
        
    
    
    
    