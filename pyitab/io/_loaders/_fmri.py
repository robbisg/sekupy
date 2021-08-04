import nibabel as ni
import numpy as np
from mvpa2.base.collections import SampleAttributesCollection
from pyitab.io.base import load_roi_labels, load_mask, load_fmri


import logging
logger = logging.getLogger(__name__)

def load_bids_fmri(filename, subject, layout, **kwargs):

    logger.debug(kwargs)

    if 'roi_labels' in kwargs.keys():
        roi_labels = kwargs.pop('roi_labels')

    if 'filetype' in kwargs.keys():
        _ = kwargs.pop('filetype')

    # Load data
    logger.info('Now loading '+filename)
    img = ni.load(filename)
    run_lengths = [img.shape[-1]]

    onset_offset = 0
    if 'onset_offset' in kwargs.keys():
        onset_offset = kwargs['onset_offset']

    extra_duration = 0
    if 'extra_duration' in kwargs.keys():
        extra_duration = kwargs['extra_duration']

    attr = load_bids_attributes(subject, run_lengths=run_lengths,
                                layout=layout,
                                onset_offset=onset_offset,
                                extra_duration=extra_duration,
                                **kwargs)

    # Loading mask
    mask = load_bids_mask(subject=subject, layout=layout, **kwargs)
    roi_labels['brain'] = mask

    # Check roi_labels
    roi_labels = load_roi_labels(roi_labels)

    # Load the pymvpa dataset.
    # TODO: Store the rest in a function
    ds = load_fmri([img], subject, attr, mask, roi_labels=roi_labels)

    return ds


def load_bids_mask(subject=None, task=None, **kwargs):

    logger.debug(kwargs)

    if 'brain_mask' in kwargs.keys():
        return load_mask(path, **kwargs)

    layout = kwargs['layout']
    kw_bids = dict()
    kw_bids['suffix'] = 'mask'

    if 'task' not in kw_bids.keys():
        kw_bids['task'] = task

    if 'run' in kwargs.keys():
       kw_bids['run'] = kwargs['run']

    logger.debug(kw_bids)

    mask_list = layout.get(return_type='file',
                           extension='.nii.gz',
                           subject=subject,
                           **kw_bids)

    logger.debug(mask_list)
    logger.info("Mask used: %s" % (mask_list[0]))

    return ni.load(mask_list[0])


def load_bids_attributes(subject, **kwargs):
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

    logger.debug(kwargs)

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
    kwargs['extension'] = 'tsv'

    event_files = layout.get(return_type='file',
                             subject=subject,
                             **kwargs
                             )

    logger.debug(event_files)
    logger.debug(kwargs)
    
    event_files = [e for e in event_files if e.find('stimlast') == -1]

    attribute_list = []

    for i, eventfile in enumerate(event_files):
        # logger.info(eventfile)

        attributes = dict()
        events = np.recfromcsv(eventfile, delimiter='\t', encoding='utf-8')

        length = run_lengths[i]

        attributes['chunks'] = np.ones(length) * i

        events_names = list(events.dtype.names)
        events_names.remove('onset')
        events_names.remove('duration')

        for field in events_names:
            attributes[field] = add_bids_attributes(field, events,
                                                    length, tr,
                                                    onset_offset=onset_offset,
                                                    extra_duration=extra_duration
                                                    )

        attributes['targets'] = attributes['trial_type'].copy()

        attribute_list.append(attributes.copy())

    # logger.debug(attribute_list)

    columns = set([k for item in attribute_list for k in item.keys()])

    attribute_dict = {k: [] for k in list(columns)}
    for i, attr in enumerate(attribute_list):
        for k in attr.keys():
            attribute_dict[k] = np.hstack((attribute_dict[k], attr[k]))
            nelem = run_lengths[i]

        if len(attr.keys()) != len(columns):
            for c in list(columns):
                if c not in attr.keys():
                    attribute_dict[c] = np.hstack((attribute_dict[c],
                                                   -1*np.ones(nelem)))

    sa = SampleAttributesCollection(attribute_dict)

    return sa


def add_bids_attributes(event_key, events, length, tr,
                        onset_offset=0, extra_duration=0):

    # logger.debug((event_key, events, length))

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