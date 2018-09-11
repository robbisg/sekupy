from pyitab.io.base import load_subject_file
import os
import unittest

currdir = os.path.dirname(os.path.abspath(__file__))
currdir = os.path.abspath(os.path.join(currdir, os.pardir))

class TestBase(unittest.TestCase):

    def test_load_subject(self):

        datadir = os.path.join(currdir, 'data', 'fmri')
        subject_file = os.path.join(datadir, 'subjects.csv')

        subjects, extra_sa = load_subject_file(subject_file)

        assert len(subjects) == 4
        assert 'age' in extra_sa.keys()

        subjects, extra_sa = load_subject_file(subject_file, n_subjects=2)

        assert len(subjects) == 2
        assert 'subj03' not in list(subjects)
        assert 'subj01' in list(subjects)


if __name__ == '__main__':
    unittest.main()