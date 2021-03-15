import h5py
import numpy as np

from mvpa2.base.collections import SampleAttributesCollection, \
     DatasetAttributesCollection, FeatureAttributesCollection

from pyitab.utils.bids import get_dictionary
from scipy.io import loadmat


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
    sa = {k: [v for _ in range(rt.shape[0])] for k, v in sa_dict.items()}
    
    sa.update({'targets': labels,
               'chunks': np.arange(rt.shape[0]),
               'limb': limb,
               'side': side,
               'rt': rt,
               'subject': [subject for _ in range(rt.shape[0])],
               'file':   [filename for _ in range(rt.shape[0])]
               })

    sa = SampleAttributesCollection(sa)

    a = DatasetAttributesCollection({'times': times})
    fa = FeatureAttributesCollection({'matrix_values': np.ones(data.shape[1])})

    mat.close()

    return data, sa, a, fa


def load_hcp_blp(filename, **kwargs):
    mat = loadmat(filename)
    data = mat['data']

    parcels = data.shape[0]

    data = data[np.triu_indices(data.shape[0], k=1)]
    data = np.expand_dims(data, axis=0)

    subject = kwargs['subject']

    sa_dict = get_dictionary(filename)
    sa_dict.pop('extension')
    sa_dict.pop('filename')
    sa = {k: [v for _ in range(data.shape[0])] for k, v in sa_dict.items()}

    sa.update({
               'subject': [subject for _ in range(data.shape[0])],
               'file':   [filename for _ in range(data.shape[0])]
               })

    idx_from, idx_to = np.triu_indices(parcels, k=1)

    labels = np.recfromcsv("/media/robbis/DATA/meg/viviana-hcp/atlas.csv")
    args = np.argsort(labels['node'])
    labels = labels[args]

    nodes_from = [labels[i]['abbr'].decode().strip() for i in idx_from]
    nodes_to = [labels[i]['abbr'].decode().strip() for i in idx_to]

    fa = dict(
        nodes_1=nodes_from,
        nodes_2=nodes_to
    )
    fa = FeatureAttributesCollection(fa)
    a = DatasetAttributesCollection({})

    return data, sa, a, fa

    
