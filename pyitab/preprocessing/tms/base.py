from scipy.stats import kurtosis, entropy

import mne
from mne.utils import _check_preload
import numpy as np
import matplotlib.pyplot as pl

import logging
logger = logging.getLogger(__name__)


def mark_tms_bad(raw, events, tmin=-0.002, tmax=0.01):
    ""
    #eog_events = mne.preprocessing.find_eog_events(raw)
    onsets = events[:, 0] / raw.info['sfreq'] + tmin
    durations = [tmax] * len(events)
    descriptions = ['bad artifact'] * len(events)
    tms_annot = mne.Annotations(onsets, durations, descriptions,
                                orig_time=raw.info['meas_date'])
    raw.set_annotations(tms_annot)

    return raw


def interpolate_tms_pulse(inst, tmin=-0.002, tmax=0.01, order=3, points=1):
    """[summary]

    Parameters
    ----------
    inst : [type]
        [description]
    tmin : float, optional
        [description], by default -0.002
    tmax : float, optional
        [description], by default 0.01
    order : int, optional
        [description], by default 3
    points : int, optional
        [description], by default 1

    Returns
    -------
    [type]
        [description]
    """

    _check_preload(inst, 'interpolate')

    t = inst.times
    mask_artifact = np.logical_and(t > tmin, t < tmax)

    t_interp_min = tmin - points / inst.info['sfreq']
    t_interp_max = tmax + points / inst.info['sfreq']

    mask_interp = np.logical_and(t >= t_interp_min, 
                                 t <= t_interp_max)

    mask_interp = np.logical_and(np.logical_not(mask_artifact), 
                                 mask_interp)

    epoch_data = inst.get_data()
    inst._data = epoch_data

    for epoch in epoch_data:
        for ch, data in enumerate(epoch):
            coeffs = np.polyfit(t[mask_interp], data[mask_interp], deg=order)
            poly = np.poly1d(coeffs)
            x_fit = poly(t[mask_artifact])

            epoch[ch, mask_artifact] = x_fit

    return inst


def read_bids_events(fname):

    events_file = np.recfromcsv(fname, delimiter='\t')

    events_name = np.unique(events_file['trial_type'])
    events_id = {k: j for j, k in enumerate(events_name)}

    events = list()
    for field in ['sample', 'duration', 'trial_type']:
        if field != 'trial_type':
            events.append(events_file[field])
        else:
            arr = [events_id[k] for k in events_file[field]]
            events.append(np.array(arr))
    
    events = np.int_(np.vstack(events).T)
    

    return events, events_id


def reject_electrodes(raw, method='kurtosis', threshold=5):
    """[summary]

    TODO: Insert the possibility to avoid normalization.

    Parameters
    ----------
    raw : [type]
        [description]
    method : str, optional
        [description], by default 'kurtosis'
    threshold : int, optional
        [description], by default 5

    Returns
    -------
    [type]
        [description]
    """

    _check_preload(raw, 'reject_electrodes')

    if method == 'kurtosis':
        metric = kurtosis(raw._data, axis=1)

    elif method == 'probability':
        metric = entropy(raw._data, axis=1)

    tmp_metric = np.sort(metric)
    idx = int(np.round(metric.shape[0] * .1))
    tmp_metric = tmp_metric[idx:-idx]

    metric = (metric - np.mean(tmp_metric)) / np.std(tmp_metric)
    
    ch_mask = np.abs(metric) > threshold

    # TODO: Check for multiple times that this function is called
    raw.info['bads'] = list(np.array(raw.info['ch_names'])[ch_mask])

    logger.info('Channels removed: '+' '.join(raw.info['bads']))

    return raw

def reject_epochs(epochs, method='jointprob', threshold=5):
    return



def reject_jointprob(epochs, selected=None, 
                     local_threshold=5, 
                     global_threshold=5):
    """[summary]

    Parameters
    ----------
    epochs : [type]
        [description]
    selected : [type], optional
        [description], by default None
    local_threshold : int, optional
        [description], by default 5
    global_threshold : int, optional
        [description], by default 5

    Returns
    -------
    [type]
        [description]
    """
    
    _check_preload(epochs, 'reject_jointprob')
    data = epochs._data
    nepochs, nchan, ntimes = data.shape

    if selected is None:
        # Channels from bads must be deleted
        selected = np.arange(nchan)

    channel_data = np.moveaxis(data, 0, -1)

    joint_prob_e, rejected_e = jointprob(channel_data, local_threshold, jp=None, normalize=True)
    _, rejected_tmp = jointprob(data[:, selected], local_threshold, jp=joint_prob_e[selected], normalize=True)
    
    rejected_e = np.zeros((nchan, nepochs))
    rejected_e[selected, :] = rejected_tmp

    epoch_data = data.reshape(nepochs, -1)
    joint_prob, rejected = jointprob(epoch_data, global_threshold, jp=None, normalize=True)

    rejected = np.logical_or(rejected, np.sum(rejected_e, axis=1) != 0)

    rejected = np.logical_or(rejected, np.sum(rejected_e[selected], axis=1) != 0)

    return rejected


def jointprob(data, threshold=0, jp=None, normalize=True, nbins=1000):

    nchan, ntimes, nevents = data.shape
    if jp is None:
        jp = np.zeros((nchan, nevents))
    
        for ch in range(nchan):
            data_prob, pdf = real_probability(data[ch], nbins)

            for e in range(nevents):

                data_ = data_prob[e*ntimes:(e+1)*ntimes]
                jp[ch, e] = -1 * np.sum(np.log(data_))

        if normalize:
            jp = (jp - np.mean(jp)) / np.std(jp)

    rejected = np.abs(jp) > threshold

    return jp, rejected


def real_probability(data, nbins):

    if nbins > 0:
        size = data.shape[0] * data.shape[1]
        histogram = np.zeros((nbins))
        max_ = np.max(data)
        min_ = np.min(data)
        data = np.floor((data - min_) / (max_ - min_) * (nbins - 1))
        data = np.int_(data.flatten())
        
        for i in range(size):
            histogram[data[i]] += 1
        
        data_prob = histogram[data] / size
        histogram /= size
        pl.plot(histogram)
    else:
        data = (data - data.mean()) / data.std()
        data_prob = np.exp(-.5 * data * data) / (2 * np.pi)
        data_prob /= np.sum(data_prob)
        histogram = data_prob

    return data_prob, histogram


def remove_data(epoch, tmin, tmax, replacement='average'):
    """Removes data between defined timepoints and replaces data with
    different approaches:

    Parameters
    ----------
    epoch : [type]
        [description]
    tmin : [type]
        [description]
    tmax : [type]
        [description]
    replacement : str, optional
        [description], by default 'average'
    """
    return
