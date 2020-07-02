from mvpa2.base.hdf5 import h5load
from pyitab.preprocessing.base import PreprocessingPipeline, Transformer
from pyitab.preprocessing.mapper import function_mapper
from pyitab.analysis.utils import get_params
from pyitab.io.configuration import read_configuration
import os
from mvpa2.datasets import vstack

import logging
logger = logging.getLogger(__name__)


class SimulationLoader(object):

    def __init__(self, conf_file, task, name='simulator', **kwargs):

        conf = read_configuration(conf_file, task)
            
        conf.update(kwargs)
        logger.debug(conf)
        
        data_path = conf['data_path']
        if len(data_path) == 1:
            data_path = os.path.abspath(os.path.join(conf_file, os.pardir))
            conf['data_path'] = data_path

        conf['task'] = task

        self.conf = conf
        self.name = name


    def fetch(self, n_subjects=10, **kwargs):
        
        ds_merged = []
        for i in range(n_subjects):
            logger.info("Creating %d/%d dataset" %(i+1, n_subjects))
            pipeline = self._get_transformer(**kwargs)
            ds = pipeline.transform(ds=None)
            ds.sa['subject'] = [i+1 for _ in range(ds.shape[0])]
            name = 'subj_simulated-%s'
            ds.sa['file'] = [name % (str(i+1)) for _ in range(ds.shape[0])]
            ds_merged.append(ds)

        ds_merged = vstack(ds_merged, a='all')
        ds_merged.a.update(self.conf)

        self._ds = ds_merged

        return ds_merged


    def _get_transformer(self, **options):

        if 'pipeline' not in options.keys():
            return Transformer()
        
        transformer = []
        for key in options['pipeline']:
            klass = function_mapper(key)           
            arg_dict = get_params(options, key)

            if key == 'sample_slicer' and 'attr' in arg_dict.keys():
                arg_dict = arg_dict['attr']

            logger.debug(klass)
            objekt = klass(**arg_dict)
            transformer.append(objekt)
        
        logger.debug(transformer)
        return PreprocessingPipeline(nodes=transformer)

    def save(self, fname):

        if hasattr(self, '_ds'):
            ds = self._ds
            fname = os.path.join(ds.a.data_path, fname)
            _ = ds.save(fname+".gzip", compression='gzip')

        return


def load_simulations(path, subj, folder, **kwargs):
    
    ds = h5load(os.path.join(path, subj+'.gzip'))
    ds.sa['file'] = [subj for _ in range(ds.shape[0])]

    return ds