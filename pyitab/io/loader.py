from pyitab.preprocessing.pipelines import StandardPreprocessingPipeline
from pyitab.io.base import load_dataset
from pyitab.io import load_ds
from pyitab.io.mapper import get_loader

import logging
logger = logging.getLogger(__name__)

# TODO : Documentation
class DataLoader(object):
    """
    This class sets up the loading, a configuration file and a task is needed
    the task should be a section in the configuration file.

    Configuration file should be like this example below:

    [path]
    data_path=/
    subjects=subjects.csv
    experiment=episodic_memory
    types=fmri
    img_dim=4
    TR=1.7

    [fmri]
    sub_dir=bold
    event_file=attributes
    event_header=True
    img_pattern=data.nii.gz
    runs=1
    mask_dir=masks
    brain_mask=lateral_ips.nii.gz

    [roi_labels]
    lateral_ips=/media/robbis/DATA/fmri/carlo_ofp/1_single_ROIs/lateral_ips.nii.gz
    
    
    Parameters
    ----------
    configuration_file : [type]
        [description]
    task : [type]
        [description]
    loader : [type], optional
        [description] (the default is load_dataset, which [default_description])
    prepro : [type], optional
        [description] (the default is StandardPreprocessingPipeline(), which [default_description])
    """      
    
    def __init__(self,
                 configuration_file,
                 task,
                 loader='base',
                 prepro=StandardPreprocessingPipeline(),
                 **kwargs):

        # TODO: Use a loader mapper?
        
        self._loader = get_loader(loader)
        self._configuration_file = configuration_file
        self._task = task
        self._prepro = prepro
        # TODO: Check configuration based on loader
        self._conf = {}
        self._conf.update(**kwargs)
        
        
        
    def fetch(self, prepro=None, n_subjects=None, subject_names=None):
        """[summary]
        
        Parameters
        ----------
        prepro : [type], optional
            [description] (the default is None, which [default_description])
        n_subjects : [type], optional
            [description] (the default is None, which [default_description])
        subject_names : [type], optional
            [description] (the default is None, which [default_description])
        
        Returns
        -------
        [type]
            [description]
        """
   
        if prepro is not None:
            self._prepro = prepro
            
        logger.debug(self._prepro)
            
        ds = load_ds(self._configuration_file,
                     self._task,
                     loader=self._loader,
                     prepro=self._prepro,
                     n_subjects=n_subjects,
                     selected_subjects=subject_names,
                     **self._conf
                     )
        
        return ds
    

    
