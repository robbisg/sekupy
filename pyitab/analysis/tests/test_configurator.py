from pyitab.analysis.configurator import AnalysisConfigurator
from sklearn.model_selection import StratifiedShuffleSplit
from pyitab.analysis.decoding.roi_decoding import RoiDecoding
from pyitab.io.loader import DataLoader
from pyitab.io.base import load_dataset
from pyitab.tests import get_datadir

import os
import numpy as np

example_configuration = {   
                            'prepro': ['sample_slicer', 'target_transformer'],
                            'sample_slicer__subject': ['subj01'], 
                            'sample_slicer__decision': ['L', 'F'],
                            'target_transformer__attr': 'decision',
                            
                            'estimator__clf__C': 1,
                            'estimator__clf__kernel': 'linear',
                            
                            'cv': StratifiedShuffleSplit,
                            'cv__n_splits': 2,
                            'cv__test_size': 0.2,
                                                        
                            'analysis': RoiDecoding,
                            'cv_attr': 'chunks'
                        }


def test_configurator_fit():

    conf = AnalysisConfigurator(**example_configuration)
    obj = conf.fit()

    assert isinstance(obj, dict)
    assert obj['loader'] is None
    assert isinstance(obj['estimator'], example_configuration['analysis'])


def test_configurator_getparams():
    extra_conf = {}
    example_configuration.update(extra_conf)
    conf = AnalysisConfigurator(**example_configuration)
    
    pass


def test_configurator_getloader(get_datadir):
    
    conf = AnalysisConfigurator(**example_configuration)
    loader = conf._get_loader()

    assert loader is None

    datadir = get_datadir
    configuration_file = os.path.join(datadir, 'fmri.conf')
    extra_conf = {'loader__configuration_file': configuration_file, 
                  'loader__task': 'fmri',
                  'loader__loader': 'base'}
    
    example_configuration.update(extra_conf)

    conf = AnalysisConfigurator(**example_configuration)
    loader = conf._get_loader()

    assert isinstance(loader, DataLoader)
    assert loader._configuration_file == configuration_file
    assert loader._task == 'fmri'
    assert loader._loader == load_dataset


def test_configurator_getanalysis():
    extra_conf = {}
    example_configuration.update(extra_conf)
    conf = AnalysisConfigurator(**example_configuration)
    
    pass


def test_configurator_getfunctionkwargs_fetch(get_datadir):

    datadir = get_datadir
    configuration_file = os.path.join(datadir, 'fmri.conf')

    n_subjects = 2

    extra_conf = {'loader__configuration_file': configuration_file, 
                  'loader__task': 'fmri',
                  'loader__loader': 'base',
                  'fetch__n_subjects': n_subjects}
    
    example_configuration.update(extra_conf)
    conf = AnalysisConfigurator(**example_configuration)

    loader = conf._get_loader()
    fetch_kw = conf._get_function_kwargs(function='fetch')
    
    ds = loader.fetch(**fetch_kw)

    assert len(np.unique(ds.sa.subject)) == n_subjects

