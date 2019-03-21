from pyitab.io.base import load_fmri, add_attributes, add_events, add_filename
from pyitab.io.base import load_mask, load_roi_labels
from pyitab.io.subjects import add_subjectname

from mvpa2.suite import SampleAttributesCollection, fmri_dataset

from bids import BIDSLayout

import nibabel as ni
import os
import numpy as np

import logging
logger = logging.getLogger(__name__)



def load_bids_dataset(path, subj, task, **kwargs):
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
       Instance of ``mvpa2.datasets.Dataset``
    '''
    
    skip_vols = 0
    roi_labels = dict()
    derivatives = False

    if 'skip_vols' in kwargs.keys():              # no. of canceled volumes
        skip_vols = np.int(kwargs['skip_vols'])
    if 'roi_labels' in kwargs.keys():             # dictionary of mask {'mask_label': string}
        roi_labels = kwargs['roi_labels']
    
    if 'bids_derivatives' in kwargs.keys():
        if kwargs['bids_derivatives'] == 'True':
            derivatives = True
        else:
            derivatives = os.path.join(path, kwargs['bids_derivatives'])

    # TODO: Use kwargs to get derivatives etc.
    layout = BIDSLayout(path, derivatives=derivatives)

    #logger.debug(layout.get())


    # Load the filename list
    kwargs_bids = get_bids_kwargs(kwargs)
    
    subj = int(subj[5:])

    file_list = layout.get(return_type='file', 
                           task=task, 
                           extensions='nii.gz', 
                           subject=subj,
                           **kwargs_bids
                           )
    logger.debug(file_list)
    # Load data
    try:
        fmri_list = load_fmri(file_list, skip_vols=skip_vols)
        
    except IOError as err:
        logger.error(err)
        return
    
    run_lengths = [img.shape[-1] for img in fmri_list]

    # Loading attributes
    attr = load_bids_attributes(path, subj, task, 
                                run_lengths=run_lengths, 
                                layout=layout)
    
    if (attr is None) and (len(file_list) == 0):
        return None            

    # Loading mask 
    mask = load_bids_mask(path, subject=subj, task=task, **kwargs)
    roi_labels['brain'] = mask
    
    # Check roi_labels
    roi_labels = load_roi_labels(roi_labels)

    # Load the pymvpa dataset.    
    try:
        logger.info('Loading dataset...')
        
        ds = fmri_dataset(fmri_list, 
                          targets=attr.targets, 
                          chunks=attr.chunks, 
                          mask=mask,
                          add_fa=roi_labels)
        
        logger.debug('Dataset loaded...')
    except ValueError as e:
        logger.error("ERROR: %s (%s)", e, subj)
        del fmri_list
    
    
    # Add filename attributes for detrending purposes   
    ds = add_filename(ds, fmri_list)
    del fmri_list
    
    # Update Dataset attributes
    ds = add_events(ds)
    
    # Name added to do leave one subject out analysis
    ds = add_subjectname(ds, subj)
    
    # If the attribute file has more fields than chunks and targets    
    ds = add_attributes(ds, attr)
         
    return ds 



def load_bids_attributes(path, subj, task, **kwargs):
    # TODO: parameters are for compatibility
    # TODO: Test with different bids datasets

    layout = kwargs['layout']
    run_lengths = kwargs['run_lengths']

    event_files = layout.get(return_type='file',
                             task=task,
                             extensions='tsv',
                             suffix='events',
                             subject=subj)

    logger.debug(event_files)

    tr = layout.get_tr()
    
    attribute_list = []

    for i, eventfile in enumerate(event_files):

        attributes = dict()
        events = np.recfromcsv(eventfile, delimiter='\t', encoding='utf-8')
        
        length = run_lengths[i]

        attributes['chunks'] = np.ones(length) * i

        events_names = list(events.dtype.names)
        events_names.remove('onset')
        events_names.remove('duration')

        for field in events_names:
            attributes[field] = add_bids_attributes(field, events, length, tr)

        attributes['targets'] = attributes['trial_type'].copy()

        attribute_list.append(attributes.copy())
    
    logger.debug(attribute_list)

    attribute_dict = {k: np.hstack([dic[k] for dic in attribute_list]) for k in attribute_list[0]}
    sa = SampleAttributesCollection(attribute_dict)

    return sa



def add_bids_attributes(event_key, events, length, tr):
        
    from itertools import groupby

    labels = events[event_key]
    dtype = events.dtype[event_key]

    targets = np.zeros(length, dtype=dtype)
    if dtype.kind is "U":
        targets[:] = 'rest'


    event_onsets = events['onset']
    event_duration = events['duration']

    
    group_events = [[key, len(list(group))] for key, group in groupby(labels)]

    for j, (label, no_events) in enumerate(group_events):
        idx = np.nonzero(labels == label)
        
        event_onset = event_onsets[idx][0]
        event_end = event_onsets[idx][-1] + event_duration[idx][-1]

        duration = event_end - event_onset

        volume_duration = np.int(np.rint(duration / tr))
        volume_onset = np.int(np.ceil(event_onset / tr))

        targets[volume_onset:volume_onset+volume_duration] = label

    return targets


def get_bids_kwargs(kwargs):

    bids_kw = {}
    for arg in kwargs:
        if arg.find("bids_") != -1:
            key = arg[5:]
            bids_kw[key] = kwargs[arg]

    _ = bids_kw.pop('derivatives')

    return bids_kw


def load_bids_mask(path, subject=None, task=None, **kwargs):

    if 'bidsmask' in kwargs:
        layout = kwargs['layout']
        kw_bids = get_bids_kwargs(kwargs)
        kw_bids['desc'] = 'brain'
        mask_list = layout.get(return_type='file', 
                               task=task, 
                               extensions='nii.gz', 
                               subject=subject, 
                               **kw_bids)

        return ni.load(mask_list[0])
    
    else:
        return load_mask(path, **kwargs)
