
def load_test_dataset(task='fmri', n_subjects=1):
    """Load a test dataset for development and testing purposes.
    
    This function loads sample datasets included with sekupy for
    testing and development. It supports different neuroimaging
    modalities including fMRI and MEG data.
    
    Parameters
    ----------
    task : str, optional
        Type of data to load ('fmri', 'meg'), by default 'fmri'
    n_subjects : int, optional
        Number of subjects to load, by default 1
        
    Returns
    -------
    Dataset
        Loaded test dataset with preprocessing applied
    """
    from sekupy.io.loader import DataLoader
    from sekupy.io.base import load_dataset
    from sekupy.io.connectivity import load_mat_ds
    from sekupy.preprocessing.base import PreprocessingPipeline
    from sekupy.preprocessing.pipelines import StandardPreprocessingPipeline
    import os
    currdir = os.path.dirname(os.path.abspath(__file__))
    currdir = os.path.abspath(os.path.join(currdir, os.pardir))
    if task != 'fmri':
        reader = 'mat'
        prepro = PreprocessingPipeline()
    else:
        reader = 'base'
        prepro = StandardPreprocessingPipeline()

    datadir = os.path.join(currdir, 'io', 'data', task)
    configuration_file = os.path.join(datadir, '%s.conf' %(task))

    loader = DataLoader(configuration_file=configuration_file, 
                        task=task,
                        loader=reader)

    ds = loader.fetch(prepro=prepro, n_subjects=n_subjects)

    return ds


def enable_logging():
    """Enable logging for sekupy with formatted output.
    
    This function sets up logging for the sekupy package with a
    detailed formatter that includes file, line number, and function
    information for debugging purposes.
    
    Returns
    -------
    logging.Logger
        Root logger instance configured for sekupy
    """
    import logging
    root = logging.getLogger()
    form = logging.Formatter('%(name)s - %(levelname)s: %(lineno)d \t %(filename)s \t%(funcName)s \t --  %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(form)
    root.addHandler(ch)
    root.setLevel(logging.INFO)
    
    return root


def get_id():
    """Generate a unique identifier string.
    
    This function creates a unique identifier by using the last 8 characters
    of a temporary directory name, with special characters replaced by '0'
    to ensure compatibility.
    
    Returns
    -------
    str
        8-character unique identifier string
    """
    import tempfile
    id_ = tempfile.mkdtemp()[-8:]
    
    # This avoids important characters to be included
    for r in ["/", "_", "-", "."]:
        id_ = id_.replace(r, "0")
    
    return id_


def make_dict_product(as_filter=True, **kwargs):
    """Create a list of dictionaries from Cartesian product of parameters.
    
    This function generates all possible combinations of parameters,
    useful for parameter grid search in neuroimaging analyses.
    
    Parameters
    ----------
    as_filter : bool, optional
        If True, wrap each parameter value in a list, by default True
    **kwargs : dict
        Parameter names and their possible values
        
    Returns
    -------
    list
        List of dictionaries, each representing one parameter combination
        
    Examples
    --------
    >>> make_dict_product(C=[1, 10], kernel=['linear', 'rbf'])
    [{'C': [1], 'kernel': ['linear']}, {'C': [1], 'kernel': ['rbf']}, ...]
    """
    import itertools
    args = [arg for arg in kwargs]
    combinations_ = list(itertools.product(*[kwargs[arg] for arg in kwargs]))
    configurations = []
    for elem in combinations_:
        if as_filter:
            elem = [[e] for e in elem]
        
        configurations.append(dict(zip(args, elem)))

    return configurations


def setup_analysis(path, analysis, participants_fname=None, **configuration):
    """Set up analysis directory structure and configuration files.
    
    This function creates the necessary directory structure and configuration
    files for a neuroimaging analysis following BIDS conventions.

    Parameters
    ----------
    path : str
        Base path where the analysis directory will be created
    analysis : str
        Name of the analysis (used for directory and config file naming)
    participants_fname : str, optional
        Name of the participants file, by default None (uses 'participants.csv')
    **configuration : dict
        Additional configuration parameters to include in the config file

    Returns
    -------
    str
        Path to the created configuration file
    """

    import os
    import configparser
    
    # Make directory
    analysis_path = os.path.join(path, analysis)
    os.makedirs(analysis_path)

    if participants_fname is None:
        participants_fname = 'participants.csv'

    # Create configuration file
    conf = {
        'path': {
            'data_path': analysis_path,
            'subjects': os.path.join(path, analysis, participants_fname),
            'types': [analysis],
            'experiment': analysis
        },
        analysis: {
            'event_file': 'None',
            'sub_dir': 'None', 
            'event_header': 'None',
            'img_pattern':'None',
            'runs':'None',
            'mask_dir':'None',
            'brain_mask':'None',
            'bids_derivatives':'None',
            'bids_scope':'None', 
            'bids_desc':'None',
        },
    }

    config = configparser.ConfigParser()

    for k in configuration.keys():
        if k in conf[analysis].keys() or k.find("bids_") != -1:
            conf[analysis][k] = str(configuration[k])
        
    for k, v in list(conf[analysis].items()):
        if v == 'None':
            conf[analysis].pop(k)

    config.update(conf)

    conf_fname = os.path.join(analysis_path, analysis+'.conf')
    with open(conf_fname, 'w') as configfile:
        config.write(configfile)

    return conf_fname

