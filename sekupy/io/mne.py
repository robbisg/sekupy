from bids import BIDSLayout
from sekupy.io.bids import get_bids_kwargs
from sekupy.utils.bids import filter_bids, filter_files, get_dictionary
from sekupy.dataset.collections import SampleAttributesCollection, \
     DatasetAttributesCollection, FeatureAttributesCollection
from sekupy.dataset.base import Dataset
from sekupy.dataset.dataset import vstack

from sekupy.io._loaders import mambo_mapper

import h5py
import os
import numpy as np

import logging
logger = logging.getLogger(__name__)


def load_bids_mne_data(path, subj, task, **kwargs):
    ''' Load a 2d dataset given the image path, the subject and the main folder of 
    the data.

    Parameters
    ----------
    path : string
       specification of filepath to load
    subj : string
        the id of the subject to load
    task : string
        the experiment name
    kwargs : keyword arguments
        Keyword arguments to format-specific load

    Returns
    -------
    ds : ``Dataset``
       Instance of ``sekupy.dataset.base.Dataset``
    '''
    
    roi_labels = dict()
    derivatives = False

    logger.debug(kwargs)

    if 'roi_labels' in kwargs.keys():             # dictionary of mask {'mask_label': string}
        roi_labels = kwargs['roi_labels']
    
    if 'bids_derivatives' in kwargs.keys():
        if kwargs['bids_derivatives'] == 'True':
            derivatives = True
        else:
            derivatives = os.path.join(path, kwargs['bids_derivatives'])
    
    if 'load_fx' in kwargs.keys():
        load_fx = mambo_mapper(kwargs['load_fx'])
    else:
        load_fx = load_hcp_motor

    # TODO: Use kwargs to get derivatives etc.
    logger.debug(derivatives)
    layout = BIDSLayout(path, derivatives=derivatives)

    logger.debug(layout.get())

    # Load the filename list
    kwargs_bids = get_bids_kwargs(kwargs)
    
    # Raise exception if it is a integer
    if isinstance(subj, str) and subj.find("-") != -1:
        subj = subj.split('-')[1]
    
    logger.debug((kwargs_bids, task, subj))

    file_list = layout.get(return_type='file', 
                           extension='mat', 
                           subject=subj,
                           suffix='conn'
                           )

    logger.debug(file_list)

    file_list = filter_files(file_list, **kwargs_bids)

    datasets = []
    for f in file_list:
        logger.info(f['filename'])
        data, sa, a, fa = load_fx(f['filename'], subject=subj)
        logger.debug(data.shape)
        ds = Dataset(data, sa=sa, a=a, fa=fa)
        datasets.append(ds)

    dataset = vstack(datasets, a='all')

    return dataset
