from sekupy.io.base import load_dataset
from sekupy.io.configuration import read_configuration
from sekupy.io.subjects import load_subjects
from sekupy.dataset.dataset import vstack

import os

import logging
logger = logging.getLogger(__name__)


def load_ds(conf_file, task, 
            extra_sa=None,
            loader=load_dataset, 
            prepro=None,
            n_subjects=None, 
            selected_subjects=None,
            **kwargs):
    """This is function loads a PyMVPA dataset given
    the configuration file and a loader.

    Parameters
    ----------
    conf_file : str
        Path of the configuration file 
        (see more in ```sekupy.io.configuration.read_configuration```)
    task : str
        name of the task that is used, this should be contatined in
        configuration file
    extra_sa : dictionary, optional
        set of extra sample attributes to be attached to the dataset, by default None
    loader : function, optional
        The function used to load the data in a correct way, by default load_dataset
    prepro : ```sekupy.preprocessing.Pipeline``` object 
                or list of ```sekupy.preprocessing.Transformer```, optional
        Preprocessing pipeline to be performed at dataset-level, by default None
    n_subjects : int, optional
        number of subjects to be loaded, by default None
    selected_subjects : list of string, optional
        name of the subjects to be loaded, by default None

    Returns
    -------
    ```sekupy.dataset.base.Dataset```
        The loaded dataset
    """

    # TODO: conf file should include the full path
    conf = read_configuration(conf_file, task)
           
    conf.update(kwargs)
    logger.debug(conf)
    
    data_path = conf['data_path']
    if len(data_path) == 1:
        data_path = os.path.abspath(os.path.join(conf_file, os.pardir))
        conf['data_path'] = data_path
    

    # Subject file should be included in configuration
    # TODO: Keep in mind BIDS
    
    subjects, extra_sa = load_subjects(conf, selected_subjects, n_subjects)
    logger.debug(subjects)

    logger.info('Merging %s subjects from %s' % (str(len(subjects)), data_path))
    
    ds_merged = []

    for i, subj in enumerate(subjects):
        
        # TODO: Keep in mind BIDS
        ds = loader(data_path, subj, task, **conf)
        
        ds = prepro.transform(ds)
        
        # add extra samples
        if extra_sa is not None:
            for k, v in extra_sa.items():
                if len(v) == len(subjects):
                    ds.sa[k] = [v[i] for _ in range(ds.samples.shape[0])]
               
        ds_merged.append(ds)
        del ds
    
    # per-subject dataset attributes before vstack drops them: most dataset
    # attrs (imgaffine, imghdr, ...) are identical across subjects, so we
    # just keep the first subject's value; a few (snr/states/time) are
    # genuinely per-subject and are kept as a tuple, one entry per subject
    per_subject_a = {}
    if len(subjects) > 1:
        all_a_keys = set().union(*(ds.a.keys() for ds in ds_merged))
        for k in all_a_keys:
            values = tuple(ds.a.get(k) for ds in ds_merged)
            per_subject_a[k] = values if k in ('snr', 'states', 'time', 'mapper') else values[0]
    else:
        per_subject_a = dict(ds_merged[0].a)

    ds_merged = vstack(ds_merged)
    ds_merged.a.update(per_subject_a)

    ds_merged.a.update(conf)
    ds_merged.a['task'] = task

    if 'name' in ds_merged.sa.keys():
        ds_merged.sa['subject'] = ds_merged.sa.pop('name')
    
    return ds_merged