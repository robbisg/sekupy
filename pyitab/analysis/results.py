import pandas as pd
import json
import os
from scipy.io import loadmat
import numpy as np
from tqdm import tqdm
from itertools import product

import logging

logger = logging.getLogger(__name__)


def get_results(path, dir_id, field_list=['sample_slicer'], result_keys=None, filter=None):
    """This function is used to collect the results from a previous analysis.
    
    Parameters
    ----------
    path : str
        The pathname of the folder in which results are stored
    dir_id : str
        The id / pattern to be used to filter folders. It is often the id of 
        the Analysis Pipeline used.
    field_list : list, optional
        List of different condition used by the AnalysisIterator  
        (the default is ['sample_slicer'], which is a fields of the configuration)
    result_keys : list, optional
        List of strings indicating the other fields to get from the result (e.g. cross_validation folds)
    filter : dictionary, optional
        This is used to filter dataset and include only fields or conditions.
        See ```pyitab.preprocessing.SampleSlicer``` for an example of dictionary
         (the default is None, which [default_description])
    
    Returns
    -------
    dataframe : pandas dataframe
        A table of the results in pandas format
    """
       
    dir_analysis = os.listdir(path)
    dir_analysis = [d for d in dir_analysis if d.find(dir_id) != -1]
    dir_analysis.sort()
    
    results = []
    full_data = []
    
    for d in tqdm(dir_analysis):
        # read json
        conf_fname = os.path.join(path, d, "configuration.json")
        with open(conf_fname) as f:
            conf = json.load(f)
        
        fields, scores = get_configuration_fields(conf, *field_list)
        
        # TODO: Check if permutation is in fields
        
        files = os.listdir(os.path.join(path, d))
        files = [f for f in files if f.find(".mat") != -1]
        
        for fname in tqdm(files):
            
            fname_split = fname.split("_")
            fields['roi'] = "_".join(fname_split[:-4])
            fields['roi_value'] = np.float_(fname_split[-4])
            fields['permutation'] = np.float_(fname_split[-2])
                    
            data = loadmat(os.path.join(path, d, fname))
            
            for score in scores:
                for i, s in enumerate(data['test_%s' % (score)].squeeze()):
                    fields[score] = s
                    fields['fold'] = i+1
                    logger.debug(fields)
                    if result_keys is not None:
                        for k in result_keys:
                            fields[k] = data[k][i].squeeze().copy()
                    fields_ = fields.copy()
                    results.append(fields_)
            

    
    dataframe = pd.DataFrame(results)
    
    if filter is not None:
        dataframe = filter_dataframe(dataframe, filter)
           
    return dataframe

    



def get_permutation_values(dataframe, keys, scores=["accuracy"]):
    
    # TODO: Multiple scores (test)
    
    
    #keys = ['band', 'targets', 'permutation', "C", "k", "n_splits"]

    df_perm = dataframe.loc[dataframe['permutation'] != 0]
    
    table = pd.pivot_table(df_perm, 
                           values=scores, 
                           index=keys, 
                           aggfunc=np.mean).reset_index()
    
    
    options = {k:np.unique(table[k]) for k in keys}
    
    
    n_permutation = options.pop('permutation')[-1]
    
    keys, values = options.keys(), options.values()
    opts = [dict(zip(keys,items)) for items in product(*values)]
    
    p_values = []

    for item in opts:
        
        cond_dict = {k:v for k,v in item.items()}
        item = {k: [v] for k, v in item.items()}
        
        df_ = dataframe.copy()
        data_ = table.copy()
        
        
        data_ = filter_dataframe(data_, item)
        item.update({'permutation':[0]})
        df_ = filter_dataframe(df_, item)
          
        for score in scores:
            
            df_avg = np.mean(df_[score].values)
            
            n_values = (np.count_nonzero(data_[score].values > df_avg) + 1)
            
            p = n_values / float(n_permutation)
            
            cond_dict[score+'_perm'] = np.mean(data_[score].values)
            cond_dict[score+'_true'] = np.mean(df_[score].values)               
            cond_dict[score+'_p'] = p
        
        p_values.append(cond_dict)
        
           
    
    return pd.DataFrame(p_values)

   
    
def get_configuration_fields(conf, *args):
    
    import ast
    
    results = dict()
    
    for k, v in conf.items():

        for arg in args:
            
            if arg == k == 'prepro':
                value = ast.literal_eval(v)
                results[arg] = "_".join(value)
            
            idx_end = len(arg)+2
            
            if k[:idx_end] == arg+"__":
                try:
                    value = ast.literal_eval(v)
                except ValueError as _:
                    if str(k[idx_end:]) != 'prepro':
                        results[str(k[idx_end:])] = v
                    else:
                        results[str(k[idx_end:])] += v
                    continue
                    
                if isinstance(value, list):
                    value = "_".join(value)
                results[str(k[idx_end:])] = value
                
    return results, ast.literal_eval(conf['scores'])




def get_searchlight_results(path, dir_id, field_list=['sample_slicer'], load_cv=False):
    
    dir_analysis = os.listdir(path)
    dir_analysis = [d for d in dir_analysis if d.find(dir_id) != -1]
    dir_analysis.sort()
    
    results = []
    
    for d in tqdm(dir_analysis):
        # read json
        conf_fname = os.path.join(path, d, "configuration.json")
        with open(conf_fname) as f:
            conf = json.load(f)
        
        # TODO: Check if permutation is in fields
        fields, scores = get_configuration_fields(conf, *field_list)
    
        files = os.listdir(os.path.join(path, d))
        files = [f for f in files if f.find(".nii.gz") != -1]
        files.sort()
        
        if not load_cv:
            files = [f for f in files if f.find("avg") != -1]
        
        for fname in tqdm(files):
            
            fname_split = fname.split("_")
            fields['measure'] = "_".join(fname_split[:-3])
            fields['permutation'] = np.float_(fname_split[-2])              
            fields['map'] = os.path.join(path, d, fname)
            
            fields_ = fields.copy()
            results.append(fields_)
    
    dataframe = pd.DataFrame(results)
       
    return dataframe
    



def filter_dataframe(dataframe, selection_dict):
    

    selection_mask = np.ones(dataframe.shape[0], dtype=np.bool)
    for key, values in selection_dict.iteritems():
        
                
        ds_values = dataframe[key].values
        condition_mask = np.zeros_like(ds_values, dtype=np.bool)
        
        for value in values:

            if str(value)[0] == '!':
                array_val = np.array(value[1:]).astype(ds_values.dtype)
                condition_mask = np.logical_or(condition_mask, ds_values != array_val)
            else:
                condition_mask = np.logical_or(condition_mask, ds_values == value)
                
        selection_mask = np.logical_and(selection_mask, condition_mask)
        
    
    return dataframe.loc[selection_mask]

