import os
import numpy as np

import logging
logger = logging.getLogger(__name__)


def load_subjects(configuration, selected_subjects=None, n_subjects=None):
    """[summary]
    
    Parameters
    ----------
    selected_subjects : [type]
        [description]
    n_subjects : [type]
        [description]
    conf : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    """

    delimiter = ','
    data_path = configuration['data_path']
    subject_file = configuration['subjects']

    if subject_file[0] != '/':
        subject_file = os.path.join(data_path, subject_file)
        #conf['subjects'] = subject_file


    if os.path.isfile(subject_file):
        _, ext = os.path.splitext(subject_file)
        if ext == '.tsv':
            delimiter = "\t"

        logger.debug(subject_file)
        
        subjects, extra_sa = load_subject_file(subject_file,
                                               delimiter=delimiter,
                                               n_subjects=n_subjects)

        logger.debug(subjects, extra_sa)
        subjects = subjects[:n_subjects]
        extra_sa = {k: extra_sa[k][:n_subjects] for k in extra_sa.keys()}


    else:
        subjects = os.listdir(data_path)
        j = os.path.join
        subjects = [s for s in subjects if os.path.isdir(j(data_path, s)) and \
                                            not s[0].isnumeric()]
        subjects = np.array(subjects)
        extra_sa = {}

    if selected_subjects is not None:
        subject_mask = [subjects == s for s in selected_subjects]
        subject_mask = np.logical_or.reduce(np.array(subject_mask))
        subjects = subjects[subject_mask]
        extra_sa = {k : v[subject_mask] for k, v in extra_sa.items()}


    return subjects, extra_sa



def load_subject_file(fname, n_subjects=None, delimiter=","):
    ''' Load information about subjects from a file.


    Parameters
    ----------
    fname : string
       The file of subjects information (.csv) 
        An example of subjects file is this:
        
        >> subjects.csv
        subject,group,group_split,age
        s01_160112alefor,1,1,21
        s02_160216micbra,1,1,30
        >>
    
    n_subjects : integer
        The number of subjects to include.

    Returns
    -------
    subjects : string array
       list of subjects name

    extra_sa : dictionary
        a dictionary of extra subject attributes like 
        age or other subject-wise information
        In the example above the dictionary will be:
        {'group':[1,1], 'group_split':[1,1], 'age':[21,30]}

    '''

    subject_array = np.recfromcsv(fname, delimiter=delimiter, encoding='utf-8')
    
    fields = subject_array.dtype.names
    subjects = subject_array[fields[0]]
    extra_sa = {k: subject_array[k] for k in fields}

    return subjects, extra_sa



def add_subjectname(ds, subj):
    """
    This function takes a string (the name of the subject) 
    and add it to the dataset for each element of the dataset
    
    Parameters
    ----------
    
    ds : pymvpa dataset
        the dataset where attribute should be added
        
    subj : string
        the name of the subject
        
    
    Returns
    -------
    ds : pymvpa dataset modified
    """
    
    ds.sa['name'] = [subj for _ in range(len(ds.sa.targets))]
    
    return ds



