from mvpa2.base.collections import SampleAttributesCollection
from mvpa2.suite import Dataset, vstack

from sklearn.preprocessing.label import LabelEncoder

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


def power_transformer(data):
    data = np.vstack(data)
    return data, data.shape[1]
    

def load_mat_data(path, subj, folder, **kwargs):
    
    from scipy.io import loadmat
    
    meg_transformer = {'connectivity': conn_transformer,
                       'power': power_transformer}
    
    key = kwargs['mat_key']
    transformer = meg_transformer[kwargs['transformer']]
    
    # load data from mat
    filelist = load_filelist(path, subj, folder, **kwargs)

    if len(filelist) == 0:
        return None
    
    data = []
    for f in filelist:
        
        logger.info("Loading %s..." %(f))
        mat = loadmat(f)
        datum = mat[key]
        logger.debug(datum.shape)
        data.append(mat[key])
    
    return transformer(data)


def load_mat_ds(path, subj, folder, **kwargs):   
        
    data, parcels = load_mat_data(path, subj, folder, **kwargs)
    
    # load attributes
    attr = load_attributes(path, subj, folder, **kwargs)

    if (attr is None) or (data is None):
        return None
    
    attr, labels = edit_attr(attr, data.shape)
    
    ds = Dataset.from_wizard(data, attr.targets)
    ds = add_subjectname(ds, subj)
    ds = add_attributes(ds, attr)

    ds = add_labels(ds, parcels, **kwargs)

    #ds.fa['roi_labels'] = labels
    ds.fa['matrix_values'] = np.ones_like(data[0])
    
    ds.sa['chunks'] = LabelEncoder().fit_transform(ds.sa['name'])
    
    return ds


def add_labels(ds, parcels, **kwargs):

    labels = get_atlas_info(kwargs['atlas'])[4]
    labels = labels[:parcels]

    if kwargs['transformer'] == 'connectivity':
        idx_from, idx_to = np.triu_indices(parcels, k=1)
        nodes_from = [labels[i] for i in idx_from]
        nodes_to = [labels[i] for i in idx_to]
        ds.fa['nodes_1'] = nodes_from
        ds.fa['nodes_2'] = nodes_to
    else:
        ds.fa['nodes'] = labels

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
    