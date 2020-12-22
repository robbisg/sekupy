from bids import BIDSLayout
from pyitab.io.bids import get_bids_kwargs
from pyitab.utils.bids import filter_bids, filter_files, get_dictionary
from mvpa2.base.collections import SampleAttributesCollection, \
     DatasetAttributesCollection, FeatureAttributesCollection
from mvpa2.datasets.base import Dataset
from mvpa2.datasets import vstack

import h5py
import os
import numpy as np

import logging
logger = logging.getLogger(__name__)



def load_reftep_sensor(filename, **kwargs):
    mat = h5py.File(filename, 'r')
    data = np.float32(mat['iPLVmat'].value[:, 1, :])
    #data /= np.nanmean(data)
    y = mat['AmpsMclean'][:].T

    sa_dict = get_dictionary(filename)
    sa_dict.pop('extension')
    sa_dict.pop('filename')
    subject = kwargs['subject']
    sa = {k:[v for _ in range(y.shape[0])] for k, v in sa_dict.items()}
    
    sa.update({'targets':   y[:, 0],
               'mep-right': y[:, 0],
               'mep-left':  y[:, 1],
               'subject':   [subject for _ in range(y.shape[0])],
               'file':      [filename for _ in range(y.shape[0])],
               'chunks' :   np.arange(y.shape[0])
                                     })

    a = DatasetAttributesCollection({})
    fa = FeatureAttributesCollection({'matrix_values':np.ones(data.shape[1])})
    sa = SampleAttributesCollection(sa)

    mat.close()

    return data, sa, a, fa


def load_reftep_power(filename, **kwargs):
    mat = h5py.File(filename, 'r')
    data = np.float32(mat['powerbox'][:])
    data /= np.nanmean(data)
    y = mat['AmpsMclean'][:].T

    sa_dict = get_dictionary(filename)
    sa_dict.pop('extension')
    sa_dict.pop('filename')
    subject = kwargs['subject']
    sa = {k:[v for _ in range(y.shape[0])] for k, v in sa_dict.items()}
    
    sa.update({'targets':   y[:,0],
               'mep-right': y[:,0],
               'mep-left':  y[:,1],
               'subject':   [subject for _ in range(y.shape[0])],
               'file':      [filename for _ in range(y.shape[0])],
               'chunks' :   np.arange(y.shape[0])
                                     })

    a = DatasetAttributesCollection({})
    fa = FeatureAttributesCollection({'matrix_values':np.ones(data.shape[1])})
    sa = SampleAttributesCollection(sa)

    mat.close()

    return data, sa, a, fa




def load_reftep_iplv(filename, **kwargs):
    mat = h5py.File(filename, 'r')
    data = np.float32(mat['iPLV'][:])
    y = mat['AmpsMclean'][:].T

    sa_dict = get_dictionary(filename)
    sa_dict.pop('extension')
    sa_dict.pop('filename')
    subject = kwargs['subject']
    sa = {k:[v for _ in range(y.shape[0])] for k, v in sa_dict.items()}
    
    sa.update({'targets':   y[:,0],
               'mep-right': y[:,0],
               'mep-left':  y[:,1],
               'subject':   [subject for _ in range(y.shape[0])],
               'file':      [filename for _ in range(y.shape[0])],
               'chunks' :   np.arange(y.shape[0])
                                     })

    a = DatasetAttributesCollection({})
    fa = FeatureAttributesCollection({'matrix_values':np.ones(data.shape[1])})
    sa = SampleAttributesCollection(sa)

    mat.close()

    return data, sa, a, fa


def load_hcp_motor(filename, **kwargs):

    targets = {
        1: 'LH',
        2: 'LF',
        4: 'RH',
        5: 'RF',
        6: 'FIX'
    }
      
    mat = h5py.File(filename)
    data = mat['powerbox'][:]
    data /= np.nanmean(data)
    # Trials x Sources x Times
    data = np.float32(data.swapaxes(1, 2))
    
    labels = [targets[t] for t in mat['trialvec'][:][0]]
    limb = [t[1] for t in labels]
    side = [t[0] for t in labels]
    
    times = mat['timevec'][:].squeeze()
    rt = mat['trailinfo'][:][5]
    subject = kwargs['subject']

    sa_dict = get_dictionary(filename)
    sa_dict.pop('extension')
    sa_dict.pop('filename')
    sa = {k:[v for _ in range(rt.shape[0])] for k, v in sa_dict.items()}
    
    sa.update({'targets': labels,
               'chunks': np.arange(rt.shape[0]),
               'limb': limb,
               'side': side,
               'rt':rt,
               'subject':[subject for _ in range(rt.shape[0])],
               'file':   [filename for _ in range(rt.shape[0])]
                                     })

    sa = SampleAttributesCollection(sa)

    a = DatasetAttributesCollection({'times': times})
    fa = FeatureAttributesCollection({'matrix_values':np.ones(data.shape[1])})

    mat.close()

    return data, sa, a, fa



def load_bids_mambo_dataset(path, subj, task, **kwargs):
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

    if 'roi_labels' in kwargs.keys():             # dictionary of mask {'mask_label': string}
        roi_labels = kwargs['roi_labels']
    
    if 'bids_derivatives' in kwargs.keys():
        if kwargs['bids_derivatives'] == 'True':
            derivatives = True
        else:
            derivatives = os.path.join(path, kwargs['bids_derivatives'])
    
    if 'load_fx' in kwargs.keys():
        load_fx = mambo_mapper[kwargs['load_fx']]
    else:
        load_fx = load_hcp_motor

    # TODO: Use kwargs to get derivatives etc.
    logger.debug(derivatives)
    layout = BIDSLayout(path, derivatives=derivatives)

    logger.debug(layout.get())

    # Load the filename list
    kwargs_bids = get_bids_kwargs(kwargs)
    
    # Raise exception if it is a integer
    if subj.find("-") != -1:
        subj = int(subj.split('-')[1])

    if 'task' not in kwargs_bids.keys():
        kwargs_bids['task'] = [task]

    logger.debug((kwargs_bids, task, subj))

    file_list = layout.get(return_type='file', 
                           extension='mat', 
                           subject=subj,
                           )

    logger.debug(file_list)

    file_list = filter_files(file_list, **kwargs_bids)

    datasets = []
    for f in file_list:
        logger.info(f['filename'])
        data, sa, a, fa = load_fx(f['filename'], subject=subj)
        ds = Dataset(data, sa=sa, a=a, fa=fa)
        datasets.append(ds)

    dataset = vstack(datasets, a='all')

    return dataset



mambo_mapper = {
    'hcp-motor': load_hcp_motor,
    'reftep-iplv': load_reftep_iplv,
    'reftep-power': load_reftep_power,
    'reftep-sensor': load_reftep_sensor
}