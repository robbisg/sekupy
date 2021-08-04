import os
import json
import numpy as np
from pyitab.results.base import get_configuration_fields
from pyitab.utils.bids import get_dictionary
from scipy.io import loadmat

import logging
logger = logging.getLogger(__name__)


def get_values_rsa(path, directory, field_list, result_keys, scores=None):

    dir_path = os.path.join(path, directory)

    conf_fname = os.path.join(dir_path, "configuration.json")
    
    with open(conf_fname) as f:
        conf = json.load(f)
    
    fields, _ = get_configuration_fields(conf, *field_list)
    
    files = os.listdir(dir_path)
    files = [f for f in files if f.find(".mat") != -1]
    
    results = []

    for fname in files:
        fname_fields = get_dictionary(fname)
        fields.update(fname_fields)
        logger.debug(fields)
        data = loadmat(os.path.join(dir_path, fname))
        logger.debug(data.keys())
        
        for score in scores:

            test_score = [k.find(score) != -1 for k in list(data.keys())]
            if not np.any(np.array(test_score)):
                score = 'score'
            fields['fx'] = score
            for i, s in enumerate(data['test_%s' % (score)]):
                fields["score_%s" % (score)] = s
                fields['fold'] = i+1
                if result_keys is not None:
                    for k in result_keys:
                        values = data[k].squeeze()
                        fields[k] = values[i].squeeze().copy()
                
                fields_ = fields.copy()

                results.append(fields_)

    return results



def get_values_lm(path, directory, field_list, result_keys, scores=None):
    
    dir_path = os.path.join(path, directory)

    conf_fname = os.path.join(dir_path, "configuration.json")
    
    with open(conf_fname) as f:
        conf = json.load(f)
    
    fields, _ = get_configuration_fields(conf, *field_list)
    
    files = os.listdir(dir_path)
    files = [f for f in files if f.find(".mat") != -1]
    
    results = []

    for fname in files:
        fname_fields = get_dictionary(fname)
        fields.update(fname_fields)
        logger.debug(fields)
        mat = loadmat(os.path.join(dir_path, fname), squeeze_me=True)
        logger.debug(mat.keys())
        
        for field in ['design_info', 'MSE', 'theta', 'r_square']:
            fields[field] = mat[field]

        if 'stats_contrasts' in mat.keys():
            for contrast in mat['stats_contrasts'].dtype.names:
                p_value = mat['stats_contrasts'][contrast][()]['p_values'][()]
                fields[contrast] = p_value

        results.append(fields.copy())
        
    return results
