from sekupy.io.loader import DataLoader
from sekupy.io.bids import load_bids_mask
from sekupy.preprocessing.pipelines import PreprocessingPipeline
from sekupy.preprocessing import SampleSlicer
from sekupy.preprocessing.base import Transformer

import numpy as np
import nibabel as ni
import os
import pytest

def test_carlo_ofp():

    data_path = '/media/robbis/DATA/fmri/carlo_ofp/'

    if not os.path.exists(data_path):
        pytest.skip("Local test")

    roi_labels = {
        'within': data_path+'1_single_ROIs/within_conjunction_mask.nii.gz', 
        'across': data_path+'1_single_ROIs/across_conjunction_mask.nii.gz'
        }

    loader = DataLoader(configuration_file="/media/robbis/DATA/fmri/carlo_ofp/ofp.conf",
                        data_path=data_path,
                        subjects='/media/robbis/DATA/fmri/carlo_ofp/subjects.csv',
                        loader='base',
                        mask_dir=data_path+'1_single_ROIs/',
                        brain_mask='glm_atlas_mask_333.nii.gz',
                        roi_labels=roi_labels,
                        task='RESIDUALS')

    ds = loader.fetch(n_subjects=2)

    assert ds.shape == (3936, 65549)
    assert len(np.unique(ds.sa.subject)) == 2


def test_carlo_mdm():

    data_path = '/media/robbis/DATA/fmri/carlo_mdm/'

    if not os.path.exists(data_path):
        pytest.skip("Local test")

    conf_file = "/media/robbis/DATA/fmri/carlo_mdm/memory.conf"
    loader = DataLoader(configuration_file=conf_file,
                        data_path='/media/robbis/DATA/fmri/carlo_mdm/',
                        subjects='/media/robbis/DATA/fmri/carlo_mdm/subjects.csv',
                        mask_dir=data_path,
                        brain_mask='glm_atlas_mask_333.nii.gz',
                        loader='base',
                        task='BETA_MVPA')
    ds = loader.fetch(n_subjects=3)

    assert ds.shape == (1080, 65549)
    assert len(np.unique(ds.sa.subject)) == 3
    assert ds.samples.min() != ds.samples.max()

def test_haxby():

    data_path = '/media/robbis/DATA/fmri/ds105/'

    if not os.path.exists(data_path):
        pytest.skip("Local test")

    conf_file = "/media/robbis/DATA/fmri/ds105/ds105.conf"
    loader = DataLoader(configuration_file=conf_file, 
                        task='objectviewing',
                        loader='bids',
                        bids_derivatives='True',
                        bids_suffix='bold'
                        )

    ds = loader.fetch(n_subjects=1)
    assert ds.shape == (1452, 67683)
    assert len(np.unique(ds.sa.subject)) == 1
    assert ds.samples.min() != ds.samples.max()



def test_egg():

    data_path = '/media/robbis/DATA/fmri/EGG/'

    if not os.path.exists(data_path):
        pytest.skip("Local test")

    conf_file = "/media/robbis/DATA/fmri/EGG/bids.conf"
    loader = DataLoader(configuration_file=conf_file, 
                        task='plain',
                        loader='bids',
                        bids_suffix='preproc',
                        onset_offset=1,
                        extra_duration=2,
                        )

    ds = loader.fetch(n_subjects=1)

    assert ds.shape == (510, 190031)
    assert len(np.unique(ds.sa.subject)) == 1
    assert ds.samples.min() != ds.samples.max()


def test_movie():

    data_path = '/media/robbis/DATA/fmri/movie_viviana/meg/'

    if not os.path.exists(data_path):
        pytest.skip("Local test")

    conf_file = '/media/robbis/DATA/fmri/movie_viviana/meg/movie.conf'
    loader = DataLoader(configuration_file=conf_file,
                        data_path=data_path,  
                        loader='mat', 
                        task='conn')

    ds = loader.fetch(n_subjects=2, prepro=[Transformer()])

    assert ds.shape == (24, 990, 1383)
    assert len(np.unique(ds.sa.subject)) == 2
    assert ds.samples.min() != ds.samples.max()


def test_sherlock():

    data_path = '/home/robbis/mount/permut1/bids/'

    if not os.path.exists(data_path):
        pytest.skip("Local test")


    conf_file = "/home/robbis/mount/permut1/bids/bids.conf"
    loader = DataLoader(configuration_file=conf_file,
                        data_path=data_path,
                        subjects='participants.tsv',
                        loader='bids', 
                        task='preproc',
                        bids_task=['encoding'],
                        bids_run=['01'])

    ds = loader.fetch(subject_names=['matsim'],
                      prepro=[SampleSlicer(trial_type=np.arange(1, 32))])

    assert ds.shape == (259, 70236)
    assert len(np.unique(ds.sa.subject)) == 1
    assert ds.samples.min() != ds.samples.max()


def test_reftep():

    data_path = '/media/robbis/DATA/meg/reftep/'

    if not os.path.exists(data_path):
        pytest.skip("Local test")

    conf_file = "/media/robbis/DATA/meg/reftep/bids.conf"
    loader = DataLoader(configuration_file=conf_file,
                        loader='bids-meg',
                        bids_band='alphalow',
                        bids_atlas='aal',
                        task='reftep',
                        load_fx='reftep-iplv')

    ds = loader.fetch(subject_names=['sub-001'],
                      prepro=[Transformer()])

    assert ds.shape == (1008, 3160)
    assert len(np.unique(ds.sa.subject)) == 1
    assert ds.samples.min() != ds.samples.max()


def test_viviana_hcp():

    data_path = '/media/robbis/DATA/meg/viviana-hcp/'

    if not os.path.exists(data_path):
        pytest.skip("Local test")

    conf_file = "/media/robbis/DATA/meg/viviana-hcp/bids.conf"

    loader = DataLoader(configuration_file=conf_file,
                        data_path=data_path,
                        subjects="/media/robbis/DATA/meg/viviana-hcp/participants.tsv",
                        loader='bids-meg',
                        task='blp',
                        bids_task='rest',
                        bids_band='alpha',
                        bids_atlas="complete",
                        bids_derivatives='True',
                        load_fx='hcp-blp')

    ds = loader.fetch(prepro=[Transformer()])


    assert ds.shape == (94, 13366)
    assert not np.all(ds.fa.nodes_1 == ds.fa.nodes_2)
    assert ds.samples.min() != ds.samples.max()
