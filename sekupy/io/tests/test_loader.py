from sekupy.io.loader import DataLoader
from sekupy.io.connectivity import load_mat_ds
from sekupy.preprocessing.pipelines import PreprocessingPipeline
import numpy as np
import os
import pytest

currdir = os.path.dirname(os.path.abspath(__file__))
currdir = os.path.abspath(os.path.join(currdir, os.pardir))


def test_fmri_data():

    datadir = os.path.join(currdir, 'data', 'fmri')
    configuration_file = os.path.join(datadir, 'fmri.conf')

    loader = DataLoader(configuration_file=configuration_file, 
                        task="fmri")

    ds = loader.fetch()

    assert len(np.unique(ds.sa.subject)) == 4
    assert ds.shape[0] == 120

    ds = loader.fetch(n_subjects=1)
    assert len(np.unique(ds.sa.subject)) == 1

    ds = loader.fetch()
    assert len(np.unique(ds.sa.subject)) == 4



def test_meg_data():

    datadir = os.path.join(currdir, 'data', 'meg')
    configuration_file = os.path.join(datadir, 'meg.conf')

    loader = DataLoader(configuration_file=configuration_file,
                        task='connectivity', 
                        loader='mat')

    ds = loader.fetch(prepro=None)
    assert len(np.unique(ds.sa.subject)) == 4
    assert ds.shape[0] == 48
