from pyitab.io.subjects import load_subject_file
import os
import pytest

currdir = os.path.dirname(os.path.abspath(__file__))
currdir = os.path.abspath(os.path.join(currdir, os.pardir))
datadir = os.path.join(currdir, 'data', 'fmri')



def test_load_subject():

    
    subject_file = os.path.join(datadir, 'subjects.csv')

    # Testing main functionality
    subjects, extra_sa = load_subject_file(subject_file)

    assert len(subjects) == 4
    assert 'age' in extra_sa.keys()

    # Testing n_subject attribute
    subjects, extra_sa = load_subject_file(subject_file, n_subjects=2)

    assert len(subjects) == 2
    assert 'subj03' not in list(subjects)
    assert len(extra_sa['age']) == 2
    assert 'subj01' in list(subjects)
