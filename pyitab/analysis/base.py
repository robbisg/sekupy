import logging
import os
import numpy as np
from pyitab.utils.files import make_dir
from pyitab.utils.time import get_time
from pyitab.utils import get_id
from pyitab.io.configuration import save_configuration
from pyitab.base import Node
logger = logging.getLogger(__name__)


class Analyzer(Node):
    
    def __init__(self, name='analyzer', **kwargs):


        self.id = get_id()
        if 'id' in kwargs.keys():
            self.id = kwargs['id']


        self.num = 1
        if 'num' in kwargs.keys():
            self.num = kwargs['num']
        
        
        Node.__init__(self, name=name, **kwargs)
        
        
    def fit(self, ds, **kwargs):
        # TODO: Must implement a store_info?
        return self


    def save(self, path=None, **kwargs):
        """Basic function for saving information about the analysis.
        Basically it should be overriden in subclasses. 

        This implementation creates the folder in which results are
        stored, following BIDS specification.
        
        Parameters
        ----------
        path : str, optional
            The pathname where results are stored, if None is passed
            it creates the directory
             (the default is None, which [default_description])
        
        **kwargs : dict, optional
            Dictionary of keywords used for directory creation.
        
        Returns
        -------
        path : str
            The directory created or the path passed as parameter.
        """


        # TODO: Keep in mind other analyses
        if not hasattr(self, "scores"):
            logger.error("Please run fit() before saving results.")
            
            return None
        
        # Build path and make dir
        path_info = self._get_analysis_info()

        if path is not None:
            path_info['path'] = path

        if 'pipeline' in kwargs.keys():
            path_info['pipeline'] = kwargs.pop('pipeline')

        logger.debug(path_info)
        path = self._build_path(**path_info)

        make_dir(path)
        logger.info("Result directory is: %s" % path)
        
        prefix = self._get_prefix()

        # Filter some params ?
        # Do it in the subclasses ?
        # Add self._info ?
        logger.debug(kwargs)
        
        kwargs.update(self._info)
        save_configuration(path, kwargs)

        self._save_dataset_description(path)

        return path, prefix


    # Only on Analyzer
    def _build_path(self, **info):

        keys = ['pipeline', 'analysis']

        pipeline_directory = []
        for k in keys:
            if k in info.keys():
                value = info.pop(k)
                if isinstance(value, list):
                    value = "+".join([str(item) for item in value])
                value = value.replace("_", "+")
                pipeline_directory += ["%s-%s" % (k, value)]
        
        path = info.pop('path')
        id_ = info.pop('id')

        for k, v in info.items():
            v = str(v).replace("_", "+")
            pipeline_directory += ["%s-%s" % (k, str(v))]

        logger.info(pipeline_directory)

        pipeline_directory += ["%s-%s" % ('id', id_)]

        subjects = np.unique(self._info['sa'].subject)
        if len(subjects) != 1:
            subdir = 'group'
        else:
            subdir = str(subjects[0])

        
        result_path = os.path.join(path, 
                                   'derivatives', 
                                   "_".join(pipeline_directory), 
                                   subdir)

        return result_path



    def _get_prefix(self):

        import numpy as np

        fname_list = np.unique(self._info['sa']['file'])

        prefix_list = os.path.basename(fname_list[0]).split("_")

        subjects = np.unique(self._info['sa'].subject)
        if len(subjects) != 1:
            prefix_list[0] = "group"
        
        if len(prefix_list) == 1:
            prefix_list = ["bids", ""]

        return "_".join(prefix_list[:-1])


    # Deprecated maybe
    def _get_fname_info(self):

        info = dict()
        logger.debug(self._info)
        info['path'] = self._info['a'].data_path
        
        info['task'] = self._info['a'].task
        
        info['analysis'] = self.name
        info['subjects'] = self._info['subjects']
        info['is_group'] = len(info['subjects']) != 1
               
        return info
            


    def _store_info(self, ds, **kwargs):

        import numpy as np

        info = dict()
        info['a'] = ds.a.copy()
        info['sa'] = ds.sa.copy()

        info.update({'ds.a.%s' % k: ds.a[k].value for k in ds.a.keys()})
        info.update({'ds.sa.%s' % k: np.unique(ds.sa[k].value) for k in ds.sa.keys()})

        info['targets'] = np.unique(ds.targets)
        info['summary'] = ds.summary()

        for k, v in kwargs.items():

            if isinstance(v, list):
                v = "+".join([str(it) for it in v])

            info[k] = str(v)
            if k == 'prepro':
                info[k] = [v.name]

        if 'subject' in ds.sa.keys():
            info['subjects'] = list(np.unique(ds.sa.subject))

        logger.debug(info)
        return info
    


    def _get_analysis_info(self):

        info = dict()
        info['analysis'] = self.name
        info['path'] = self._info['a'].data_path
        info['id'] = "%s+%04d" % (self.id, self.num)
        info['experiment'] = self._info['a'].experiment

        # Only for testing purposes
        self._test_id = info['id']

        return info

    
    
    def _get_info(self, attributes=['estimator', 'scoring', 
                                    'cv','permutation']):

        # TODO: It may crash whether used with connectivity
        # TODO: maybe it should be performed on subclasses
        
        import numpy as np
        
        info = dict()
        
        for k in attributes:
            info[k] = getattr(self, k)
        info['targets'] = self._info['targets']
        
        for k in self._info['sa'].keys():
            info[k] = np.unique(self._info['sa'][k].value)
        info['summary'] = self._info['summary']

        return info


    # TODO: Look if can be applied to connectivity
    def _get_permutation_indices(self, n_samples):
        """Permutes the indices of the dataset"""
        
        # TODO: Permute labels based on cv_attr
        from sklearn.utils import shuffle
        
        if self.permutation == 0:
            return [range(n_samples)]
        
        indices = [range(n_samples)]
        for r in range(self.permutation):
            idx = shuffle(indices[0], random_state=r)
            indices.append(idx)
        
        return indices


    def _get_test_id(self):
        if '_test_id' in self.__dict__.keys():
            return getattr(self, '_test_id')


    
    def _save_dataset_description(self, path):
        """This function saves a dataset_description.json
        for BIDS dataformat
        
        Parameters
        ----------
        path : str
            The path that will be used to save the file
        """

        info = self._get_analysis_info()

        keys = ['pipeline', 'analysis', 'id']

        description =  {
                        "Name": "PyITAB - Pipelines for neuroimaging",
                        "BIDSVersion": "1.1.1",
                        "PipelineDescription": {
                            "Name": "_".join([info[k] for k in keys if k in info.keys()]),
                        },
                        "CodeURL": "https://github.com/robbisg/pyitab"
                    }

        dataset_desc = os.path.join(os.path.dirname(path), 
                                    "dataset_description.json")
        
        if not os.path.exists(dataset_desc):
            import json
            
            with open(dataset_desc, 'w') as fp:
                json.dump(description, fp)
        



