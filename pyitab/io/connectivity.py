import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

from mvpa2.base.collections import SampleAttributesCollection
from mvpa2.suite import Dataset, vstack

from sklearn.preprocessing.label import LabelEncoder

from pyitab.io.configuration import read_configuration
from pyitab.analysis import Node
from pyitab.io.base import load_subject_file, add_attributes, load_attributes,\
    add_subjectname, load_filelist

# TODO : Document 

def conn_transformer(data):
    
    data = np.rollaxis(np.vstack(data), 0, 3)
    data = data[np.triu_indices(data.shape[0], k=1)].T
    logger.debug(data.shape)
    
    return data
    

def load_mat_data(path, subj, folder, **kwargs):
    
    from scipy.io import loadmat
    
    meg_transformer = {'connectivity': conn_transformer,
                       'power': lambda data: np.vstack(data)}
    
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
        
    data = load_mat_data(path, subj, folder, **kwargs)
    
    # load attributes
    attr = load_attributes(path, subj, folder, **kwargs)

    if (attr is None) or (data is None):
        return None
    
    attr, labels = edit_attr(attr, data.shape)
    
    ds = Dataset.from_wizard(data, attr.targets)
    ds = add_subjectname(ds, subj)
    ds = add_attributes(ds, attr)
    
    #ds.fa['roi_labels'] = labels
    ds.fa['matrix_values'] = np.ones_like(data[0])
    
    ds.sa['chunks'] = LabelEncoder().fit_transform(ds.sa['name'])
    
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
    