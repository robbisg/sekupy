from pyitab.analysis.configurator import AnalysisConfigurator
from pyitab.analysis.pipeline import AnalysisPipeline
from pyitab.analysis.iterator import AnalysisIterator
from sklearn.model_selection import StratifiedShuffleSplit, LeaveOneGroupOut
from pyitab.analysis.decoding.roi_decoding import RoiDecoding
from pyitab.analysis.utils import get_params


from pyitab.tests import fetch_ds, get_datadir, tmpdir

import pytest
import numpy as np

def test_iterator_combination():
    pass

def test_iterator_list():
    pass

def test_iterator_dict():
    pass

def test_iterator_subjectwise():
    pass


def test_save_multisubject_decoding(fetch_ds, tmpdir):

    import os
    options = {
            'sample_slicer__subject': [['subj01'], ['subj02']],

    }
    example_configuration = {   
                                'prepro': ['sample_slicer', 'target_transformer'],
                                 
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
    ds = fetch_ds

    iterator = AnalysisIterator(options, 
                                AnalysisConfigurator(**example_configuration))


    path = tmpdir
    for conf in iterator:
        kwargs = conf._get_kwargs()

        a = AnalysisPipeline(conf, name='test')
        a.fit(ds, **kwargs)
        a.save(path=path)


    print(a._estimator._test_id)
    
    experiment = conf._default_options['ds.a.experiment']
    experiment = experiment.replace("_", "+")

    pipeline_folder = "pipeline-%s_analysis-%s_experiment-%s_roi-%s_id-%s" % \
        ('test', 'roi+decoding', experiment, 
            'all', str(a._estimator._test_id))
    print(pipeline_folder)
    
    expected_folder = os.path.join(path, 'derivatives', pipeline_folder)
    print(os.listdir(expected_folder))
    assert os.path.exists(expected_folder)
    assert len(os.listdir(expected_folder)) == 2 + 1
    
    subject_folder = os.path.join(expected_folder, 'subj02')
    assert os.path.exists(subject_folder)
    assert os.path.exists(os.path.join(expected_folder, 'subj01'))
 
    params = get_params(conf._default_options, 'sample_slicer')
    
    slicers = "_".join(["%s-%s" % (k, "+".join(v)) for k, v in params.items()])
    fname = "bids_%s_mask-%s_value-%s_perm-%s_data.%s" % \
                (slicers, 'brain', '2.0', '0000', 'mat')
    
    assert os.path.exists(os.path.join(subject_folder, fname))
    # assert os.path.exists(os.path.join(subject_folder, conf_fname))
    
    assert len(os.listdir(subject_folder)) == 26 + 1
