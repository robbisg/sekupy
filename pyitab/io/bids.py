from pyitab.io.base import load_fmri, add_attributes, add_events, add_filename
from pyitab.io.base import load_mask, load_roi_labels
from pyitab.io.subjects import add_subjectname

from mvpa2.base.collections import SampleAttributesCollection
from mvpa2.datasets.mri import fmri_dataset

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
    
    roi_labels = dict()
    derivatives = False

    logger.debug(kwargs)

    if 'roi_labels' in kwargs.keys():
        roi_labels = kwargs['roi_labels']
    
    if 'bids_derivatives' in kwargs.keys():
        if kwargs['bids_derivatives'] == 'True':
            derivatives = True
        elif kwargs['bids_derivatives'] == 'False':
            derivatives = False        
        else:
            derivatives = os.path.join(path, kwargs['bids_derivatives'])
    
    tr = None
    if 'tr' in kwargs.keys():
        tr = kwargs['tr']

    logger.debug(derivatives)
    layout = BIDSLayout(path, derivatives=derivatives)

    logger.debug(layout.get())

    # Load the filename list
    kwargs_bids = get_bids_kwargs(kwargs)
    
    if subj.find("-") != -1:
        try:
            subj = int(subj.split('-')[1])
        except Exception as err:
            subj = subj.split('-')[1]

    if 'task' not in kwargs_bids.keys():
        kwargs_bids['task'] = task

    if 'suffix' not in kwargs_bids.keys():
        kwargs_bids['suffix'] = 'bold'

    logger.debug((kwargs_bids, task, subj))

    file_list = layout.get(return_type='file', 
                           extension='.nii.gz', 
                           subject=subj,
                           **kwargs_bids)

    logger.debug(file_list)

    file_list = [f for f in file_list if f.find('pipeline') == -1]

    # Load data
    fmri_list = load_fmri(file_list)

    # Loading attributes
    run_lengths = [img.shape[-1] for img in fmri_list]

    onset_offset = 0
    if 'onset_offset' in kwargs.keys():
        onset_offset = kwargs['onset_offset']

    extra_duration = 0
    if 'extra_duration' in kwargs.keys():
        extra_duration = kwargs['extra_duration']
    
    attr = load_bids_attributes(path, subj, run_lengths=run_lengths, 
                                layout=layout, tr=tr, 
                                onset_offset=onset_offset, 
                                extra_duration=extra_duration, 
                                **kwargs_bids)
               

    # Loading mask
    mask = load_bids_mask(path, subject=subj, 
                          task=task, layout=layout, **kwargs)
    roi_labels['brain'] = mask
    
    # Check roi_labels
    roi_labels = load_roi_labels(roi_labels)

    logger.debug(roi_labels)

    # Load the pymvpa dataset.    
    logger.info('Loading dataset...')
    
    ds = fmri_dataset(fmri_list, 
                      targets=attr.targets, 
                      chunks=attr.chunks, 
                      mask=mask,
                      add_fa=roi_labels)
    
    logger.debug('Dataset loaded...')

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


def load_bids_attributes(path, subj, **kwargs):
    """[summary]

    Parameters
    ----------
    path : [type]
        [description]
    subj : [type]
        [description]
    **kwargs : dictionary
        run_lenghts, tr, layout, onset_offset, extra_duration
    Returns
    -------
    [type]
        [description]

    Raises
    ------
    Exception
        [description]
    """
    
    # TODO: parameters are for compatibility
    # TODO: Test with different bids datasets
    
    for k in ['desc', 'scope']:
        if k in kwargs.keys():
            _ = kwargs.pop(k)

    onset_offset = kwargs.pop('onset_offset')
    extra_duration = kwargs.pop('extra_duration')
    layout = kwargs.pop('layout')
    run_lengths = kwargs.pop('run_lengths')

    tr = None
    if 'tr' in kwargs.keys():
        tr = float(kwargs.pop('tr'))

    try:
        tr = layout.get_tr()
    except Exception as err:
        if tr is None:
            raise Exception("tr must be set in configuration file")

    kwargs['suffix'] = 'events'

    event_files = layout.get(return_type='file',
                             extension='.tsv',
                             subject=subj,
                             **kwargs
                             )

    logger.debug(event_files)
    logger.debug(kwargs)

    event_files = [e for e in event_files if e.find('stimlast') == -1]

    attribute_list = []
    
    for i, eventfile in enumerate(event_files):
        #logger.info(eventfile)

        attributes = dict()
        events = np.recfromcsv(eventfile, delimiter='\t', encoding='utf-8')
        
        length = run_lengths[i]

        attributes['chunks'] = np.ones(length) * i

        events_names = list(events.dtype.names)
        events_names.remove('onset')
        events_names.remove('duration')

        for field in events_names:
            attributes[field] = add_bids_attributes(field, 
                                                    events, 
                                                    length, 
                                                    tr,
                                                    onset_offset=onset_offset,
                                                    extra_duration=extra_duration
                                                    )

        attributes['targets'] = attributes['trial_type'].copy()

        attribute_list.append(attributes.copy())
    
    #logger.debug(attribute_list)

    columns = set([k for item in attribute_list for k in item.keys()])

    attribute_dict = {k:[] for k in list(columns)}
    for i, attr in enumerate(attribute_list):
        for k in attr.keys():
            attribute_dict[k] = np.hstack((attribute_dict[k], attr[k]))
            nelem = run_lengths[i]
        
        if len(attr.keys()) != len(columns):
            for c in list(columns):
                if c not in attr.keys():
                    attribute_dict[c] = np.hstack((attribute_dict[c], -1*np.ones(nelem)))


    #attribute_dict = {k: np.hstack([dic[k] for dic in attribute_list]) for k in attribute_list[11]}
    sa = SampleAttributesCollection(attribute_dict)

    return sa



def add_bids_attributes(event_key, events, length, tr, onset_offset=0, extra_duration=0):

    #logger.debug((event_key, events, length))
    # TODO: Add frame field
    from itertools import groupby

    labels = events[event_key]
    
    # This is to avoid 0-shaped event
    labels = labels.reshape(labels.size)
    dtype = events.dtype[event_key]

    targets = np.zeros(length, dtype=dtype)
    if dtype.kind is "U":
        targets[:] = 'rest'

    event_onsets = events['onset']
    event_onsets = np.hstack((event_onsets, [length * tr]))
    event_duration = events['duration']
    event_duration = event_duration.reshape(event_duration.size)

    group_events = [[key, len(list(group))] for key, group in groupby(labels)]

    for j, (label, no_events) in enumerate(group_events):
        idx = np.nonzero(labels == label)[0]
        
        for i in idx:
            event_onset = event_onsets[i]
            event_end = event_onset + event_duration[i]
           
            volume_onset = np.int(np.floor(event_onset / tr))
            volume_duration = np.int(np.rint(event_end / tr))

            volume_onset += onset_offset
            volume_duration += extra_duration

            targets[volume_onset:volume_duration] = label

    return targets


def get_bids_kwargs(kwargs):

    bids_kw = {}
    for arg in kwargs:
        if arg.find("bids_") != -1:
            key = arg[5:]

            if isinstance(kwargs[arg], str):
                bids_kw[key] = kwargs[arg].split(',')
            else:
                bids_kw[key] = kwargs[arg]
            
    
        if arg == 'bids_derivatives':
            bids_kw.pop('derivatives')

        if arg == 'bids_desc':
            bids_kw.pop('desc')

    return bids_kw


def load_bids_mask(path, subject=None, task=None, **kwargs):

    if 'brain_mask' in kwargs.keys():
        return load_mask(path, **kwargs)

    layout = kwargs['layout']
    kw_bids = get_bids_kwargs(kwargs)
    kw_bids['suffix'] = 'mask'

    if 'task' not in kw_bids.keys():
        kw_bids['task'] = task

    logger.debug(kw_bids)

    if 'run' in kw_bids.keys():
        _ = kw_bids.pop('run')

    mask_list = layout.get(return_type='file',
                           extension='.nii.gz',
                           subject=subject,
                           **kw_bids)

    logger.debug(mask_list)
    
    if len(mask_list) == 0:
        return None

    logger.info("Mask used: %s" % (mask_list[0]))

    return ni.load(mask_list[0])

