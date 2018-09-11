from pyitab.preprocessing.pipelines import StandardPreprocessingPipeline
from pyitab.io.base import load_dataset, load_subject_ds
from pyitab.io.configuration import read_configuration

import logging
logger = logging.getLogger(__name__)


class DataLoader(object):
    
    
    def __init__(self, 
                 configuration_file, 
                 task, 
                 loader=load_dataset, 
                 prepro=StandardPreprocessingPipeline(),
                 **kwargs):
        
        
        """
        This sets up the loading, a configuration file and a task is needed
        the task should be a section of the configuration file.        
        """        
        
        self._loader = loader
        self._configuration_file = configuration_file
        self._task = task
        self._prepro = prepro
        
        self._conf = read_configuration(configuration_file, task)
        self._conf.update(**kwargs)
            
        self._data_path = self._conf['data_path']
        
        
        
    def fetch(self, prepro=None, n_subjects=None):
        
        if prepro is not None:
            self._prepro = prepro
            
        logger.debug(self._prepro)
            
        ds = load_subject_ds(self._configuration_file,
                             self._task,
                             loader=self._loader,
                             prepro=self._prepro,
                             n_subjects=n_subjects)
        

        ds = self._update_ds(ds)
        
        return ds
    





    
    
    def _update_ds(self, ds):
        
        ds.a.update(self._conf)
        ds.a['task'] = self._task
        
        return ds
    
