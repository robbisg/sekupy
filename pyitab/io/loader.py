from pyitab.preprocessing.pipelines import StandardPreprocessingPipeline
from pyitab.io.base import load_subject_ds
from pyitab.io.configuration import read_configuration

import logging
logger = logging.getLogger(__name__)


class DataLoader(object):
    
    
    def __init__(self, 
                 configuration_file, 
                 task, 
                 reader=load_subject_ds, 
                 prepro=StandardPreprocessingPipeline(),
                 **kwargs):
        
        
        """
        This sets up the loading, a configuration file and a task is needed
        the task should be a section of the configuration file.        
        """        
        
        self._loader = reader
        self._configuration_file = configuration_file
        self._task = task
        self._prepro = prepro
        
        self._conf = read_configuration(configuration_file, task)
        self._conf.update(**kwargs)
            
        self._data_path = self._conf['data_path']
        
        object.__init__(self, **kwargs)


        
        
    def fetch(self, prepro=None):
        
        if prepro != None:
            self._prepro = prepro
            
        logger.debug(self._prepro)
            
        ds =  self._loader(self._configuration_file, 
                           self._task, 
                           prepro=self._prepro)
        

        ds = self._update_ds(ds)
        
        #logger.debug(hpy().heap())
        
        return ds
    
    
    
    def _update_ds(self, ds):
        
        ds.a.update(self._conf)
        ds.a['task'] = self._task
        
        return ds
    
    
        