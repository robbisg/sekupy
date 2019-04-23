from pyitab.io.loader import DataLoader
from pyitab.io.bids import load_bids_dataset, load_bids_mask
from pyitab.preprocessing.pipelines import PreprocessingPipeline

from bids import BIDSLayout

import numpy as np
import nibabel as ni
import os
import pytest

currdir = os.path.dirname(os.path.abspath(__file__))
currdir = os.path.abspath(os.path.join(currdir, os.pardir))


def test_bids_data():

    datadir = os.path.join(currdir, 'data', 'bids')
    configuration_file = os.path.join(datadir, 'ds105.conf')

    loader = DataLoader(configuration_file=configuration_file,
                        loader='bids',
                        task="objectviewing")

    ds = loader.fetch()

    assert 'name' not in ds.sa.keys()
    assert len(np.unique(ds.sa.subject)) == 2
    assert ds.shape[0] == 121 * 2 * 2


def test_bids_mask():
    bids_dir = os.path.join(currdir, 'data', 'bids')
    layout = BIDSLayout(bids_dir, derivatives=True)

    mask = load_bids_mask(bids_dir, 
                          subject='1', 
                          task='objectviewing',
                          layout=layout,
                          bids_derivatives='True',
                          bidsmask="mask",
                          scope='derivatives')

    assert isinstance(mask, ni.Nifti1Image)
    assert mask.shape == (60, 72, 60)



def test_bids_events():
    pass