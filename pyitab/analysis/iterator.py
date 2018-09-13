import logging
logger = logging.getLogger(__name__)

_default_options = {
       
                       'sampleslicer__condition' : [['vipassana'], ['samatha']],
                       'estimator__svr__C': [1, 10],                          
                       'cv__n_splits': [10, 15],
                       'analysis__radius':[9,18],
                           
                        }
     
    


class AnalysisIterator(object):

    
    def __init__(self, options, configurator):
        """This class allows to configure different analysis to be
        iterated using a set of options.
                
        Parameters
        ----------
        options : dict
            A dictionary that include all different values that must be 
            iterated in the analysis to be performed.
            
        configurator : [type]
            [description]
        
        """

        
        
        self.configurations, self.i, self.n = self._setup(**options)
        self._configurator = configurator

   
    def _setup(self, **kwargs):
        
        import itertools
            
        args = [arg for arg in kwargs]
        logger.info(kwargs)
        combinations_ = list(itertools.product(*[kwargs[arg] for arg in kwargs]))
        configurations = [dict(zip(args, elem)) for elem in combinations_]
        i = 0
        n = len(self.configurations)
        
        return configurations, i, n
        

    def __iter__(self):
        return self
    
        
    
    def next(self):
        
        if self.i < self.n:
            value = self.configurations[self.i]
            self.i += 1
            logger.info("Iteration %d/%d" %(self.i, self.n))
            self._configurator.set_params(**value)
        else:
            raise StopIteration()
        
        return self._configurator
    
    
    