from pyitab.analysis.configurator import AnalysisConfigurator
from pyitab.analysis.pipeline import AnalysisPipeline
from pyitab.analysis.iterator import AnalysisIterator
from sklearn.model_selection import StratifiedShuffleSplit, LeaveOneGroupOut
from pyitab.analysis.decoding.roi_decoding import RoiDecoding
from pyitab.analysis.states.base import Clustering
from sklearn import cluster, mixture
from pyitab.analysis.utils import get_params


from pyitab.tests import fetch_ds, get_datadir, tmpdir

import pytest
import numpy as np


def test_iterator_combination():
    
    options = {
            'sample_slicer__subject': [['subj01'], ['subj02']],
            'estimator__clf__C': [1, 2, 3],
            'cv__n_splits': [2, 5]

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

    iterator = AnalysisIterator(options, 
                                AnalysisConfigurator,
                                config_kwargs=example_configuration)

    n_options = np.prod([len(v) for k, v in options.items()])

    assert len(list(iterator)) == n_options



def test_iterator_list():

    options = {
            'sample_slicer__subject': [['subj01'], ['subj02']],
            'estimator__clf__C': [1, 2],
            'cv__n_splits': [2, 5]

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

    iterator = AnalysisIterator(options, 
                                AnalysisConfigurator,
                                config_kwargs=example_configuration,
                                kind='list'
                                )

    n_options = len(options['cv__n_splits'])
    assert len(list(iterator)) == n_options

    # Failing
    options = {
            'sample_slicer__subject': [['subj01'], ['subj02']],
            'estimator__clf__C': [1, 2],
            'cv__n_splits': [2, 5, 3]

    }
    
    iterator = AnalysisIterator(options, 
                                AnalysisConfigurator,
                                config_kwargs=example_configuration,
                                kind='list'
                                )
                                
    bad_n_options = len(options['cv__n_splits'])
    n_options = len(list(iterator))
    assert n_options != bad_n_options

    good_n_options = np.prod([len(v) for k, v in options.items()])
    assert n_options == good_n_options
    
def test_iterator_combined():
    options = {
        'estimator': [
            [('clf1', cluster.MiniBatchKMeans())],
            [('clf2', cluster.KMeans())],
            [('clf5', mixture.GaussianMixture())]
        ],
        'estimator__clf1__n_clusters'  : range(3, 5),
        'estimator__clf5__n_components': range(3, 5)
    }

    example_configuration = {'analysis': Clustering}

    iterator = AnalysisIterator(options, 
                                AnalysisConfigurator,
                                config_kwargs=example_configuration,
                                kind='combined'
                                )
    
    n_options = len(list(iterator))
    assert n_options == 5

    options = {
        'estimator': [
            [('clf1', cluster.MiniBatchKMeans())],
            [('clf2', cluster.KMeans())],
            [('clf5', mixture.GaussianMixture())]
        ],
        'estimator__clf1__n_clusters': range(3, 5),
        'estimator__clf1__tol': [.1, .3, .5],
        'estimator__clf5__n_components': range(3, 5),
        'prepro': [['sample_slicer'], ['feature_slicer']]
    }

    iterator = AnalysisIterator(options, 
                                AnalysisConfigurator, 
                                config_kwargs=example_configuration,
                                kind='combined'
                                )
    n_options = len(list(iterator))
    assert n_options == 2*((2*3*1) + (2*1) + 1)

    options = {
                'estimator': [
                    [('clf1', cluster.MiniBatchKMeans())],
                    [('clf1', cluster.KMeans())],
                    [('clf1', cluster.SpectralClustering())],
                    [('clf1', cluster.AgglomerativeClustering())],
                    [('clf5', mixture.GaussianMixture())]
                ],
                'estimator__clf1__n_clusters': range(3, 5),
                'estimator__clf5__n_components': range(2, 4),
            } 

    iterator = AnalysisIterator(options,
                                AnalysisConfigurator,
                                config_kwargs=example_configuration,
                                kind='combined'
                                )

    n_options = len(list(iterator))
    assert n_options == (2*4) + 2

    
def test_iterator_subjectwise():
    pass

@pytest.mark.skip()
def test_save_multisubject_decoding(fetch_ds, tmp_path):

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
                                AnalysisConfigurator,
                                config_kwargs=example_configuration)


    path = str(tmp_path)
    for conf in iterator:
        kwargs = conf._get_kwargs()

        a = AnalysisPipeline(conf, name='test')
        a.fit(ds, **kwargs)
        a.save(path=path)
    
    experiment = conf._default_options['ds.a.experiment']
    experiment = experiment.replace("_", "+")

    pipeline_folder = "pipeline-%s_analysis-%s_experiment-%s_roi-%s_id-%s" % \
        ('test', 'roi+decoding', experiment, 'all', str(a._estimator._test_id))

    
    expected_folder = os.path.join(path, 'derivatives', 'pipeline-test', pipeline_folder)

    assert os.path.exists(expected_folder)
    assert len(os.listdir(expected_folder)) == 2 + 1
    
    subject_folder = os.path.join(expected_folder, 'subj02')
    assert os.path.exists(subject_folder)
    assert os.path.exists(os.path.join(expected_folder, 'subj01'))
 
    params = get_params(conf._default_options, 'sample_slicer')
    params.update(get_params(conf._default_options, 'target_transformer'))
    
    slicers = list()
    for k, v in params.items():
        if isinstance(v, str):
            v = [v]
        slicers.append("%s-%s" % (k, "+".join(v)))

    slicers = "_".join(slicers)
    fname = "bids_%s_mask-%s_value-%s_perm-%s_data.%s" % \
                (slicers, 'brain', '2.0', '0000', 'mat')
    
    assert os.path.exists(os.path.join(subject_folder, fname))
    # assert os.path.exists(os.path.join(subject_folder, conf_fname))
    
    assert len(os.listdir(subject_folder)) == 26 + 1
