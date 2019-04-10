from pyitab.io.subjects import load_subject_file, load_subjects
from pyitab.io.configuration import read_configuration
import os
import pytest

currdir = os.path.dirname(os.path.abspath(__file__))
currdir = os.path.abspath(os.path.join(currdir, os.pardir))
datadir = os.path.join(currdir, 'data', 'fmri')


def test_load_subject_file():

    subject_file = os.path.join(datadir, 'subjects.csv')

    # Testing main functionality
    subjects, extra_sa = load_subject_file(subject_file)

    assert len(subjects) == 4
    assert 'age' in extra_sa.keys()


def test_load_subjects():

    conf_file = os.path.join(datadir, 'fmri.conf')
    configuration = read_configuration(conf_file, 'fmri')

    data_path = configuration['data_path']
    if len(data_path) == 1:
        data_path = os.path.abspath(os.path.join(conf_file, os.pardir))
        configuration['data_path'] = data_path


    subjects, extra_sa = load_subjects(configuration, n_subjects=2)

    assert len(subjects) == 2
    assert 'subj03' not in list(subjects)
    assert len(extra_sa['age']) == 2
    assert 'subj01' in list(subjects)

    selected_subjects = ['subj02', 'subj04', 'subj03']
    subjects, extra_sa = load_subjects(configuration, 
                                       selected_subjects=selected_subjects)

    assert len(subjects) == len(selected_subjects)
    assert 'subj01' not in list(subjects)
    assert 'subj03' in list(subjects)