from pyitab.io.base import load_dataset
from pyitab.io.configuration import read_configuration
from pyitab.io.subjects import load_subjects
from mvpa2.datasets import vstack

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
        (see more in ```pyitab.io.configuration.read_configuration```)
    task : str
        name of the task that is used, this should be contatined in
        configuration file
    extra_sa : dictionary, optional
        set of extra sample attributes to be attached to the dataset, by default None
    loader : function, optional
        The function used to load the data in a correct way, by default load_dataset
    prepro : ```pyitab.preprocessing.Pipeline``` object 
                or list of ```pyitab.preprocessing.Transformer```, optional
        Preprocessing pipeline to be performed at dataset-level, by default None
    n_subjects : int, optional
        number of subjects to be loaded, by default None
    selected_subjects : list of string, optional
        name of the subjects to be loaded, by default None

    Returns
    -------
    ```mvpa2.suite.Dataset```
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
        try:
            ds = loader(data_path, subj, task, **conf)
        except Exception as e:
            logger.info(e)
            print(e)
            continue
        
        ds = prepro.transform(ds)
        
        # add extra samples
        if extra_sa is not None:
            for k, v in extra_sa.items():
                if len(v) == len(subjects):
                    ds.sa[k] = [v[i] for _ in range(ds.samples.shape[0])]
               
        ds_merged.append(ds)
        del ds
    
    ds_merged = vstack(ds_merged, a='all')

    if len(subjects) > 1:
        for k in ds_merged.a.keys():
            if k not in ['snr', 'states', 'time', 'mapper']:
                ds_merged.a[k] = ds_merged.a[k].value[0]

    
    ds_merged.a.update(conf)
    ds_merged.a['task'] = task

    if 'name' in ds_merged.sa.keys():
        ds_merged.sa['subject'] = ds_merged.sa.pop('name')
    
    return ds_merged


def dataset_wizard(X, y=None, **kwargs):

    from mvpa2.base.collections import SampleAttributesCollection, \
        DatasetAttributesCollection, FeatureAttributesCollection
    from mvpa2.datasets.base import Dataset
    import numpy as np


    sa = SampleAttributesCollection({
        'targets': y,
        'subject': np.ones(X.shape[0]),
        'file': ["foo.mat" for _ in range(X.shape[0])]
    })

    fa = FeatureAttributesCollection({'matrix_values':np.ones(X.shape[1])})
    a = DatasetAttributesCollection({'data_path':'/media/robbis/DATA/meg/hcp/', 
                                    'experiment':'hcp', 
                                    })

    ds = Dataset(X, sa=sa, a=a, fa=fa)

    return ds