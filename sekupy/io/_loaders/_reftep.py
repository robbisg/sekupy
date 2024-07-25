import h5py
import numpy as np

from sekupy.dataset.collections import SampleAttributesCollection, \
     DatasetAttributesCollection, FeatureAttributesCollection
from sekupy.dataset.base import Dataset

from sekupy.utils.bids import get_dictionary

from scipy.io import loadmat


def load_reftep_sensor(filename, subject=None, layout=None, **kwargs):
    mat = h5py.File(filename, 'r')
    data = np.float32(mat['iPLVmat'].value[:, 1, :])
    # data /= np.nanmean(data)
    y = mat['AmpsMclean'][:].T

    sa_dict = get_dictionary(filename)
    sa_dict.pop('extension')
    sa_dict.pop('filename')
    sa = {k: [v for _ in range(y.shape[0])] for k, v in sa_dict.items()}
    
    sa.update({'targets':   y[:, 0],
               'mep-right': y[:, 0],
               'mep-left':  y[:, 1],
               'subject':   [subject for _ in range(y.shape[0])],
               'file':      [filename for _ in range(y.shape[0])],
               'chunks':    np.arange(y.shape[0])
               })

    a = DatasetAttributesCollection({})
    fa = FeatureAttributesCollection({'matrix_values': np.ones(data.shape[1])})
    sa = SampleAttributesCollection(sa)

    mat.close()

    ds = Dataset(data, sa=sa, a=a, fa=fa)

    return data, sa, a, fa


def load_reftep_power(filename, subject, layout=None, **kwargs):
    mat = h5py.File(filename, 'r')
    data = np.float32(mat['powerbox'][:])
    data /= np.nanmean(data)
    y = mat['AmpsMclean'][:].T

    sa_dict = get_dictionary(filename)
    sa_dict.pop('extension')
    sa_dict.pop('filename')
    sa = {k: [v for _ in range(y.shape[0])] for k, v in sa_dict.items()}
    
    sa.update({'targets':   y[:, 0],
               'mep-right': y[:, 0],
               'mep-left':  y[:, 1],
               'subject':   [subject for _ in range(y.shape[0])],
               'file':      [filename for _ in range(y.shape[0])],
               'chunks':    np.arange(y.shape[0])
               })

    a = DatasetAttributesCollection({})
    fa = FeatureAttributesCollection({'matrix_values': np.ones(data.shape[1])})
    sa = SampleAttributesCollection(sa)

    mat.close()

    ds = Dataset(data, sa=sa, a=a, fa=fa)

    return data, sa, a, fa


def load_reftep_iplv(filename, subject=None, layout=None, **kwargs):
    mat = h5py.File(filename, 'r')
    data = np.float32(mat['iPLV'][:])
    y = mat['AmpsMclean'][:].T

    sa_dict = get_dictionary(filename)
    sa_dict.pop('extension')
    sa_dict.pop('filename')
    sa = {k: [v for _ in range(y.shape[0])] for k, v in sa_dict.items()}
    
    sa.update({'targets':   y[:, 0],
               'mep-rapb':  y[:, 0],
               'mep-rfdi':  y[:, 1],
               'subject':   [subject for _ in range(y.shape[0])],
               'file':      [filename for _ in range(y.shape[0])],
               'chunks':    np.arange(y.shape[0])
               })

    a = DatasetAttributesCollection({})
    fa = FeatureAttributesCollection({'matrix_values': np.ones(data.shape[1])})
    sa = SampleAttributesCollection(sa)

    mat.close()

    ds = Dataset(data, sa=sa, a=a, fa=fa)

    return data, sa, a, fa


def load_reftep_conn(filename, subject=None, layout=None, **kwargs):

    mat = h5py.File(filename, 'r')
    data = np.float32(mat['oPLV'][:])
    y = mat['MEP_Amp'][:]

    meps = mat['MEP'][:]
    phases = mat['phases1'][:].squeeze()
    amps = mat['amps1'][:].squeeze()

    sa_dict = get_dictionary(filename)
    sa_dict.pop('extension')
    sa_dict.pop('filename')
    sa = {k: [v for _ in range(y.shape[0])] for k, v in sa_dict.items()}
    
    sa.update({'targets':   y[:, 0],
               'mep-rapb':  meps[:, 0],
               'mep-rfdi':  meps[:, 1],
               'phases':    phases,
               'amps':      amps,
               'subject':   [subject for _ in range(y.shape[0])],
               'file':      [filename for _ in range(y.shape[0])],
               'chunks':    np.arange(y.shape[0])
               })

    a = DatasetAttributesCollection({})
    fa = FeatureAttributesCollection({'matrix_values': np.ones(data.shape[1])})
    sa = SampleAttributesCollection(sa)

    mat.close()

    ds = Dataset(data, sa=sa, a=a, fa=fa)

    return data, sa, a, fa



def load_mtms_iplv(filename, subject=None, layout=None, **kwargs):
    
    mat = loadmat(filename, 'r')
    data = np.float32(mat['oPLV'][:])
    y = mat['MEP_Amp'][:]

    meps = mat['MEP'][:]
    phases = mat['phases1'][:].squeeze()
    amps = mat['amps1'][:].squeeze()

    sa_dict = get_dictionary(filename)
    sa_dict.pop('extension')
    sa_dict.pop('filename')
    sa = {k: [v for _ in range(y.shape[0])] for k, v in sa_dict.items()}
    
    sa.update({'targets':   y[:, 0],
               'mep-rapb':  meps[:, 0],
               'mep-rfdi':  meps[:, 1],
               'phases':    phases,
               'amps':      amps,
               'subject':   [subject for _ in range(y.shape[0])],
               'file':      [filename for _ in range(y.shape[0])],
               'chunks':    np.arange(y.shape[0])
               })

    a = DatasetAttributesCollection({})
    fa = FeatureAttributesCollection({'matrix_values': np.ones(data.shape[1])})
    sa = SampleAttributesCollection(sa)

    mat.close()

    ds = Dataset(data, sa=sa, a=a, fa=fa)

    return data, sa, a, fa