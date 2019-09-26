
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