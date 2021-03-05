from mvpa2.base.collections import SampleAttributesCollection
from mvpa2.datasets.base import Dataset
from mvpa2.datasets import vstack

from sklearn.preprocessing import LabelEncoder

from pyitab.io.configuration import read_configuration
from pyitab.base import Node
from pyitab.io.base import add_attributes, load_attributes, load_filelist
from pyitab.io.subjects import load_subject_file, add_subjectname
from pyitab.utils.atlas import get_atlas_info

import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

# TODO : Document 

def conn_transformer(data):
    
    data = np.rollaxis(np.vstack(data), 0, 3)
    parcels = data.shape[0]
    data = data[np.triu_indices(data.shape[0], k=1)].T
    logger.debug(data.shape)
    
    return data, parcels


def time_transformer(data):
    data = np.vstack(data)
    return data, data.shape[1]


def power_transformer(data):
    data = np.vstack(data)
    return data, data.shape[1]
    

def load_mat_data(path, subj, folder, **kwargs):
    
    from scipy.io import loadmat
    
    meg_transformer = {'connectivity': conn_transformer,
                       'power': power_transformer,
                       'timevarying': time_transformer
                       }
    
    key = kwargs['mat_key']
    transformer = meg_transformer[kwargs['transformer']]
    
    # load data from mat
    filelist = load_filelist(path, subj, folder, **kwargs)

    if filelist is None or len(filelist) == 0:
        return None, None
    
    data = []
    info = {'file': []}
    for f in filelist:
        
        logger.info("Loading %s..." %(f))
        mat = loadmat(f)
        datum = mat[key]
        logger.debug(datum.shape)
        data.append(mat[key])
        info['file'] += [f for _ in range(datum.shape[0])]

    data, info['parcels'] = transformer(data)

    return data, info


def load_mat_ds(path, subj, folder, **kwargs):   
    
    logger.debug(kwargs)
    data, info = load_mat_data(path, subj, folder, **kwargs)
    
    # load attributes
    attr = load_attributes(path, subj, folder, **kwargs)

    if (attr is None) or (data is None):
        return None
    
    attr, labels = edit_attr(attr, data.shape)

    logger.debug(data.shape)
    logger.debug(attr)
    
    ds = Dataset.from_wizard(data, attr.targets, flatten=False)
    ds = add_subjectname(ds, subj)
    ds = add_attributes(ds, attr)

    ds = add_labels(ds, info['parcels'], **kwargs)

    #ds.fa['roi_labels'] = labels
    ds.fa['matrix_values'] = np.ones(data.shape[1], dtype=np.int8)
    
    ds.sa['chunks'] = LabelEncoder().fit_transform(ds.sa['name'])
    ds.sa['file'] = info['file']
    
    return ds


def add_labels(ds, parcels, **kwargs):

    info = get_atlas_info(kwargs['atlas'])
    labels = info[4]
    labels = labels[:parcels]

    if kwargs['transformer'] == 'connectivity':
        idx_from, idx_to = np.triu_indices(parcels, k=1)
        nodes_from = [labels[i] for i in idx_from]
        nodes_to = [labels[i] for i in idx_to]
        ds.fa['nodes_1'] = nodes_from
        ds.fa['nodes_2'] = nodes_to
    elif kwargs['transformer'] == 'power':
        ds.fa['nodes'] = labels
    elif kwargs['transformer'] == 'timevarying':
        n_nodes = len(labels)
        idx_from, idx_to = np.triu_indices(n_nodes, k=1)
        nodes_from = [labels[i] for i in idx_from]
        nodes_to = [labels[i] for i in idx_to]
        network_from = []
        ds.fa['nodes_1'] = nodes_from
        ds.fa['nodes_2'] = nodes_to
        
        pass

    # TODO: Power labels

    return ds


def edit_attr(attr, shape):
        
    factor = int(shape[0]/len(attr.targets))

    attr_ = dict()
    for key in attr.keys():
        attr_[key] = []
        for label in attr[key]:
            attr_[key] += [label for i in range(factor)]
            
    """    
    attr_['roi_labels'] = []
    for j in range(len(attr.targets)):
        for i in range(shape[1]):
            attr_['roi_labels'] += ["roi_%02d" % (i+1)]
    """
    
    #attr_['roi_labels'][:shape[1]]
    return SampleAttributesCollection(attr_), None
    