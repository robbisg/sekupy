from pyitab.io.base import load_dataset
from pyitab.io.configuration import read_configuration
from pyitab.io.subjects import load_subjects
from pyitab.preprocessing.pipelines import StandardPreprocessingPipeline

from mvpa2.suite import vstack

import os
import numpy as np

import logging
logger = logging.getLogger(__name__)



def load_ds(conf_file, task, extra_sa=None,
            loader=load_dataset, 
            prepro=StandardPreprocessingPipeline(),
            n_subjects=None, selected_subjects=None,
            **kwargs):

    # TODO: Documentation
    
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
    
    for i, subj in enumerate(subjects):
        
        # TODO: Keep in mind BIDS
        ds = loader(data_path, subj, task, **conf)
        
        if ds is None:
            continue
        
        ds = prepro.transform(ds)
        
        # add extra samples
        if extra_sa is not None:
            for k, v in extra_sa.items():
                if len(v) == len(subjects):
                    ds.sa[k] = [v[i] for _ in range(ds.samples.shape[0])]
        
        
        # First subject
        if i == 0:
            ds_merged = ds.copy()
        else:
            ds_merged = vstack((ds_merged, ds))
            ds_merged.a.update(ds.a)
            
        del ds
    
    ds_merged.a.update(conf)
    ds_merged.a['task'] = task
    
    return ds_merged