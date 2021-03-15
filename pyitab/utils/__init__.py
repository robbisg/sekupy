
def load_test_dataset(task='fmri', n_subjects=1):
    from pyitab.io.loader import DataLoader
    from pyitab.io.base import load_dataset
    from pyitab.io.connectivity import load_mat_ds
    from pyitab.preprocessing.base import PreprocessingPipeline
    from pyitab.preprocessing.pipelines import StandardPreprocessingPipeline
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
    import logging
    root = logging.getLogger()
    form = logging.Formatter('%(name)s - %(levelname)s: %(lineno)d \t %(filename)s \t%(funcName)s \t --  %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(form)
    root.addHandler(ch)
    root.setLevel(logging.INFO)
    
    return root


def get_id():
    import tempfile
    id_ = tempfile.mkdtemp()[-8:]
    
    # This avoids important characters to be included
    for r in ["/", "_", "-", "."]:
        id_ = id_.replace(r, "0")
    
    return id_


def make_dict_product(as_filter=True, **kwargs):
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
    """This function sets up the analysis files 

    Parameters
    ----------
    path : [type]
        [description]
    analysis : [type]
        [description]
    participants_fname : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]
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

