import pandas as pd
import json
import os
import numpy as np
import logging

from scipy.io import loadmat
from scipy.stats import ttest_1samp
from itertools import product
from joblib import Parallel, delayed
from pyitab.results.base import get_configuration_fields, filter_dataframe
from pyitab.utils.bids import get_dictionary, find_directory

logger = logging.getLogger(__name__)



def get_values_bids(path, directory, field_list, result_keys, scores=None):
    """[summary]

    Parameters
    ----------
    path : [type]
        [description]
    directory : [type]
        [description]
    field_list : [type]
        [description]
    result_keys : [type]
        [description]
    scores : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]
    """


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
            for i, s in enumerate(data['test_%s' % (score)].squeeze()):
                fields["score_%s" % (score)] = s
                fields['fold'] = i+1
                logger.debug(fields)
                if result_keys is not None:
                    for k in result_keys:
                        values = data[k].squeeze()
                        fields[k] = values[i].squeeze().copy()
                
                fields_ = fields.copy()

                results.append(fields_)

    return results


def get_results_bids(path, field_list=['sample_slicer'], 
                    result_keys=[], scores=['score'], 
                    n_jobs=-1,  verbose=1, filter=None,
                    get_function=get_values_bids, 
                    subjects=None,
                    **kwargs):
    """This function is used to collect the results from analysis folders.
    
    Parameters
    ----------
    path : str
        The pathname of the folder in which results are stored
    field_list : list, optional
        List of different condition used by the AnalysisIterator  
        (the default is ['sample_slicer'], which is a fields of the configuration)
    result_keys : list, optional
        List of strings indicating the other fields to get from the result (e.g. cross_validation folds)
    filter : dictionary, optional
        This is used to filter dataset and include only fields or conditions.
        See ```pyitab.preprocessing.SampleSlicer``` for an example of dictionary
         (the default is None, which [default_description])
    scores : list, optional
        Use mse and corr for regression, score for basic decoding
    get_function: fx, optional
        The fucntion used to load data
    subjects : list, optional
        List of string representing subject's results to be loaded.
    **kwargs : dictionary, optional
        List of parameters used to filter BIDS folder by 'pipeline' for example.
    
    Returns
    -------
    dataframe : pandas dataframe
        A table of the results in pandas format
    """
    # TODO: Use function for this snippet

    
    filtered_dirs = find_directory(path, **kwargs)
    logger.debug(filtered_dirs)

    results = []

    for item in filtered_dirs:
        
        # read json
        pipeline_dir = os.path.join(path, item['filename'])
        subject_dirs = os.listdir(pipeline_dir)
        subject_dirs = [d for d in subject_dirs if d.find(".json") == -1]

        if subjects is not None:
            subject_dirs = [d for d in subject_dirs if d in subjects]

        r = [get_function(pipeline_dir, s, field_list, result_keys, scores=scores) \
             for s in subject_dirs]
        """
        r = Parallel(n_jobs=n_jobs, 
            verbose=verbose)(delayed(get_values_bids) \
                            (pipeline_dir, s, field_list, result_keys, scores=scores) \
                                    for s in subject_dirs)
        """
        results.append(r)

    results_ = [i for sublist in results for item in sublist for i in item]
    dataframe = pd.DataFrame(results_)

    if filter is not None:
        dataframe = filter_dataframe(dataframe, **filter)
    
    return dataframe
    


def get_permutation_values(dataframe, keys, scores=["accuracy"]):
    # TODO: Document
    # TODO: Multiple scores (test)
    
    
    #keys = ['band', 'targets', 'permutation', "C", "k", "n_splits"]

    df_perm = dataframe.loc[dataframe['permutation'] != 0]
    
    table_perm = pd.pivot_table(df_perm, 
                                values=scores, 
                                index=keys, 
                                aggfunc=np.mean).reset_index()
    
    
    options = {k:np.unique(table_perm[k]) for k in keys}
    
    n_permutation = options.pop('permutation')[-1]
    
    keys, values = options.keys(), options.values()
    opts = [dict(zip(keys,items)) for items in product(*values)]
    
    p_values = []

    for item in opts:
        
        cond_dict = {k: v for k, v in item.items()}
        item = {k: [v] for k, v in item.items()}
        
        df_true = dataframe.copy()
        df_permutation = table_perm.copy()
        
        df_permutation = filter_dataframe(df_permutation, **item)
        item.update({'permutation': [0]})
        df_true = filter_dataframe(df_true, **item)
          
        for score in scores:
            
            df_avg = np.mean(df_true[score].values)
            
            permutation_values = df_permutation[score].values

            n_values = (np.count_nonzero(permutation_values > df_avg) + 1)
            
            p = n_values / float(n_permutation)
            
            cond_dict[score+'_perm'] = np.mean(df_permutation[score].values)
            cond_dict[score+'_true'] = np.mean(df_true[score].values)            
            cond_dict[score+'_p'] = p
        
        p_values.append(cond_dict)
            
    return pd.DataFrame(p_values)


def get_searchlight_results_bids(path, field_list=['sample_slicer'], **kwargs):
    
    kwargs.update({'analysis':'searchlight'})
    filtered_dirs = find_directory(path, **kwargs)

    results = []
    
    for item in filtered_dirs:
        
        # read json
        pipeline_dir = os.path.join(path, item['filename'])
        subject_dirs = os.listdir(pipeline_dir)
        subject_dirs = [d for d in subject_dirs if d.find(".json") == -1]
        
        
        #r = [get_values_searchlight()]
        for subject in subject_dirs:
            
            conf_fname = os.path.join(pipeline_dir, subject, "configuration.json")
            with open(conf_fname) as f:
                conf = json.load(f)
            
            # TODO: Check if permutation is in fields
            fields, scores = get_configuration_fields(conf, *field_list)
            logger.debug(fields)
            files = os.listdir(os.path.join(pipeline_dir, subject))
            files = [f for f in files if f.find(".nii.gz") != -1]
            files.sort()
            #logger.debug(files)            
            
            # TODO: Use Parallel
            for fname in files:
                fname_fields = get_dictionary(fname)
                fields.update(fname_fields)
                fields['filename'] = os.path.join(pipeline_dir, 
                                                  subject, 
                                                  fields['filename'])      
                fields_ = fields.copy()
                results.append(fields_)
        
        #results.append(r)

    #results_ = [i for sublist in results for item in sublist for i in item]
    dataframe = pd.DataFrame(results)
    
    #if filter is not None:
        #dataframe = filter_dataframe(dataframe, **filter)

    return dataframe
