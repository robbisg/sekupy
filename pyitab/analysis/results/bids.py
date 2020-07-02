import pandas as pd
import json
import os
import numpy as np
import logging

from scipy.io import loadmat
from scipy.stats import ttest_1samp
from itertools import product
from joblib import Parallel, delayed
from pyitab.analysis.results.base import get_configuration_fields, filter_dataframe

logger = logging.getLogger(__name__)



def get_values_bids(path, directory, field_list, result_keys, scores=None):


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
    **kwargs : dictionary, optional
        List of parameters used to filter BIDS folder by 'pipeline' for example.
    
    Returns
    -------
    dataframe : pandas dataframe
        A table of the results in pandas format
    """
    # TODO: Use function for this snippet
    dir_analysis = os.listdir(path)
    dir_analysis.sort()
    dir_analysis = [get_dictionary(f) for f in dir_analysis]

    filtered_dirs = []
    for key, value in kwargs.items():
        for dictionary in dir_analysis:
            if key in dictionary.keys():
                value = value.replace("_", "+")
                if value == dictionary[key]:
                    filtered_dirs.append(dictionary)

    logger.info(filtered_dirs)
    
    results = []

    for item in filtered_dirs:
        
        # read json
        pipeline_dir = os.path.join(path, item['filename'])
        subject_dirs = os.listdir(pipeline_dir)
        subject_dirs = [d for d in subject_dirs if d.find(".json") == -1]

        r = Parallel(n_jobs=n_jobs, 
            verbose=verbose)(delayed(get_values_bids)\
                            (pipeline_dir, s, field_list, result_keys, scores=scores) \
                                    for s in subject_dirs)
        
        results.append(r)
         
    results_ = [i for sublist in results for item in sublist for i in item]
    dataframe = pd.DataFrame(results_)

    if filter is not None:
        dataframe = filter_dataframe(dataframe, **filter)
    
    return dataframe
    

def get_results(path, pipeline_name, field_list=['sample_slicer'], 
                result_keys=None, filter=None, n_jobs=-1, verbose=1):
    """This function is used to collect the results from analysis folders.
    
    Parameters
    ----------
    path : str
        The pathname of the folder in which results are stored
    pipeline_name : str
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
    # TODO: Optimize memory
    dir_analysis = os.listdir(path)
    dir_analysis = [d for d in dir_analysis if d.find(pipeline_name) != -1 and d.find(".") == -1]
    dir_analysis.sort()
    
    logger.info("Loading %d files..." %(len(dir_analysis)))
    results = Parallel(n_jobs=n_jobs, 
                       verbose=verbose)(delayed(get_values)(path, d, field_list, result_keys) \
                    for d in dir_analysis)
    
    results = [item for sublist in results for item in sublist]

    dataframe = pd.DataFrame(results)
    
    if filter is not None:
        dataframe = filter_dataframe(dataframe, **filter)
           
    return dataframe


def ttest_values(dataframe, keys, scores=["accuracy"], popmean=0.5):
    # TODO: Documentation
    # TODO: Multiple scores (test)
        
    
    options = {k: np.unique(dataframe[k]) for k in keys}
    
    keys, values = options.keys(), options.values()
    opts = [dict(zip(keys,items)) for items in product(*values)]
    
    p_values = []

    for item in opts:
        
        cond_dict = {k: v for k, v in item.items()}
        item = {k: [v] for k, v in item.items()}
        
        df_true = dataframe.copy()
        df_true = filter_dataframe(df_true, **item)
          
        for score in scores:
            
            values_score = df_true[score].values
            print(values_score)
            t, p = ttest_1samp(values_score, popmean)

            cond_dict[score+'_avg'] = np.mean(values_score)
            cond_dict[score+'_t'] = t            
            cond_dict[score+'_p'] = p
        
        p_values.append(cond_dict)
        
           
    
    return pd.DataFrame(p_values)





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
    
    # TODO: Mind BIDS!
    dir_analysis = os.listdir(path)
    dir_analysis.sort()
    dir_analysis = [get_dictionary(f) for f in dir_analysis]

    filtered_dirs = []

    if kwargs == {}:
        filtered_dirs = dir_analysis

    for key, value in kwargs.items():
        for dictionary in dir_analysis:
            if key in dictionary.keys():
                value = value.replace("_", "+")
                if value == dictionary[key]:
                    filtered_dirs.append(dictionary)

    results = []

    logger.debug(filtered_dirs)
    
    for item in filtered_dirs:
        
        # read json
        pipeline_dir = os.path.join(path, item['filename'])
        subject_dirs = os.listdir(pipeline_dir)
        subject_dirs = [d for d in subject_dirs if d.find(".json") == -1]
         
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
    
    dataframe = pd.DataFrame(results)
       
    return dataframe

# TODO: Generalize for dirs and files
def get_dictionary(filename):
    dictionary = dict()

    parts = filename.split("_")

    index = [i for i, f in enumerate(parts) if f.find("-") == -1]

    if len(index) == len(parts):
        return dictionary

    new_parts = []
    for i in index:
        part = parts[i]
        logger.debug(part)
        if i == len(parts) - 1:
            pp = part.split(".")

            if len(pp) == 3:
                trailing = pp[0]
                ext = "%s.%s" % (pp[1], pp[2])
            else:
                trailing, ext = pp

            new_parts.append("filetype-%s" % (trailing))
            new_parts.append("extension-%s" % (ext))

        if i == 0:
            new_parts.append("subjecttype-%s" %(part))

    parts += new_parts
    logger.debug(parts)

    for part in parts:
        try:
            key, value = part.split("-")
        except Exception as err:
            continue

        dictionary[key] = value
    
    dictionary['filename'] = filename

    return dictionary