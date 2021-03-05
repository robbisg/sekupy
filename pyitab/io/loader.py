from pyitab.io.base import load_dataset
from pyitab.io.configuration import read_configuration
from pyitab.io.subjects import load_subjects
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
    configuration_file : str
        The path of the configuration file
    task : [type]
        [description]
    loader : [type], optional
        [description] (the default is load_dataset, which [default_description])
    **kwargs : arguments dictionary, optional
        Arguments passed to loading functions. They override the configuration.

        data_path : str, the path where data is stored
        subjects : str, the path to subject file
        experiment : str, pipeline name, this will be discarded in future
        types : list of str, list of subsections of configuration file
        sub_dir : str, sub directory where data is stored.
        event_file : str, path or name of the event file
        mask_dir : str, path of mask/ROIs directories
        brain_mask : str, name of the mask/ROI to use for reduce voxels
        roi_labels : dict, a dictionary with ROI name as key and path to ROI as value.
        
    """      
    
    def __init__(self,
                 configuration_file,
                 task,
                 loader='base',
                 **kwargs):

        # TODO: Use a loader mapper?
        
        self._loader = get_loader(loader)
        self._configuration_file = configuration_file
        self._task = task
        # TODO: Check configuration based on loader
        self._conf = {}
        self._conf.update(**kwargs)
        
        
        
    def fetch(self, prepro=None, n_subjects=None, subject_names=None):
        """This function starts to load data given the information provided
        in the constructor.
        
        Parameters
        ----------
        prepro : :class:`~pyitab.preprocessing.pipelines.PreprocessingPipeline`
        or list of strings, optional
            Preprocessing steps to be perfrormed at subject level (the default is None)
        n_subjects : int, optional
            Number of subjects to load in the order provided by the participants.csv file
             (the default is None)
        subject_names : list of strings, optional
            The list of subject names to be loaded (the default is None)
        
        Returns
        -------
        ds: :class:`~mvpa2.dataset.Dataset`
            The loaded dataset.
        """
   
        from pyitab.preprocessing.pipelines import StandardPreprocessingPipeline, \
            PreprocessingPipeline
        if prepro is None:
            prepro = StandardPreprocessingPipeline()
        else:
            prepro = PreprocessingPipeline(nodes=prepro)
            
        logger.debug(prepro)
            
        ds = load_ds(self._configuration_file,
                     self._task,
                     loader=self._loader,
                     prepro=prepro,
                     n_subjects=n_subjects,
                     selected_subjects=subject_names,
                     **self._conf
                     )
        
        return ds
    

    def get_subjects(self):
        """Returns the subject list

        Returns
        -------
        subjects : list of strings
            The subject list provided by participants.csv
        """

        conf = read_configuration(self._configuration_file, 
                                  self._task)
        
        subjects, _ = load_subjects(conf)

        return subjects

    
