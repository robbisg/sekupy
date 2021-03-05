from pyitab.analysis.configurator import AnalysisConfigurator
from pyitab.analysis.pipeline import AnalysisPipeline
from sklearn.model_selection import StratifiedShuffleSplit, LeaveOneGroupOut
from pyitab.analysis.decoding.roi_decoding import RoiDecoding
from pyitab.analysis.utils import get_params


from pyitab.tests import fetch_ds, get_datadir, tmpdir

import pytest
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



def test_fit(fetch_ds):
    ds = fetch_ds

    conf = AnalysisConfigurator(**example_configuration)
    a = AnalysisPipeline(conf, name='test')

    a.fit(ds)

    scores = a._estimator.scores
    assert len(scores.keys()) == 26 # No. of ROI
    
    roi_result = scores['mask-brain_value-2.0']
    assert roi_result[0]['test_accuracy'].shape == (2,)


def test_fit_without_ds():

    with pytest.raises(Exception):
        conf = AnalysisConfigurator(**example_configuration)
        a = AnalysisPipeline(conf, name='test')

        a.fit()


def test_fit_with_ds(fetch_ds, get_datadir):
    import os
    datadir = get_datadir
    ds = fetch_ds
    configuration_file = os.path.join(datadir, 'fmri.conf')

    n_subjects = 2

    assert len(np.unique(ds.sa.subject)) != n_subjects
    assert len(np.unique(ds.sa.subject)) == 4

    extra_conf = {'loader__configuration_file': configuration_file, 
                  'loader__task': 'fmri',
                  'loader__loader': 'base',
                  'fetch__n_subjects': n_subjects,
                  'cv': LeaveOneGroupOut,
                  'cv_attr': 'subject'
                  }
    
    example_configuration.update(extra_conf)
    example_configuration.pop('cv__n_splits')
    example_configuration.pop('cv__test_size')
    example_configuration.pop('sample_slicer__subject')
    conf = AnalysisConfigurator(**example_configuration)

    a = AnalysisPipeline(conf, name='test')
    a.fit(ds, cv_attr='subject')

    scores = a._estimator.scores    
    roi_result = scores['mask-brain_value-2.0']
    assert roi_result[0]['test_accuracy'].shape == (4,)


def test_save_pipeline_decoding(fetch_ds, tmpdir):

    import os
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
    ds = fetch_ds
    conf = AnalysisConfigurator(**example_configuration)
    a = AnalysisPipeline(conf, name='test')
    a.fit(ds)

    path = tmpdir
    a.save(path=path)

    print(path)
    print(a._estimator._test_id)
    experiment = conf._default_options['ds.a.experiment']
    experiment = experiment.replace("_","+")

    pipeline_folder = "pipeline-%s_analysis-%s_experiment-%s_roi-%s_id-%s" % \
        ('test', 'roi+decoding', experiment, 
            'all', str(a._estimator._test_id))
    print(pipeline_folder)
    
    expected_folder = os.path.join(path, 'derivatives', pipeline_folder)
    assert os.path.exists(expected_folder)
    assert os.path.exists(os.path.join(expected_folder, "dataset_description.json"))
    
    subject_folder = os.path.join(expected_folder, 'subj01')
    assert os.path.exists(subject_folder)

    print(conf._default_options) 
    params = get_params(conf._default_options, 'sample_slicer')
    
    slicers = "_".join(["%s-%s" % (k, "+".join(v)) for k, v in params.items()])
    fname = "bids_%s_mask-%s_value-%s_perm-%s_data.%s" % \
                (slicers, 'brain', '2.0', '0000', 'mat')
    print(os.listdir(os.path.join(path, 'derivatives', pipeline_folder, 'subj01')))
    assert os.path.exists(os.path.join(subject_folder, fname))
    # assert os.path.exists(os.path.join(subject_folder, conf_fname))
    
    assert len(os.listdir(subject_folder)) == 26 + 1

