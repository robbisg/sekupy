import pandas as pd
import json
import os
from scipy.io import loadmat
from scipy.stats import ttest_1samp
import numpy as np
from tqdm import tqdm
from itertools import product
from joblib import Parallel, delayed
import logging

logger = logging.getLogger(__name__)



def get_values(path, directory, field_list, result_keys):


    dir_path = os.path.join(path, directory)

    conf_fname = os.path.join(dir_path, "configuration.json")
    
    with open(conf_fname) as f:
        conf = json.load(f)
    
    fields, scores = get_configuration_fields(conf, *field_list)
    
    files = os.listdir(dir_path)
    files = [f for f in files if f.find(".mat") != -1]
    
    results = []

    for fname in files:

        fname_split = fname.split("_")
        fields['roi'] = "_".join(fname_split[:-4])
        fields['roi_value'] = np.float16(fname_split[-4])
        fields['permutation'] = np.float16(fname_split[-2])

        data = loadmat(os.path.join(dir_path, fname))
        logger.debug(data.keys())
        
        for score in scores:

            test_score = [k.find(score) != -1 for k in list(data.keys())]
            if not np.any(np.array(test_score)):
                score = 'score'

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


    

def get_results(path, dir_id, field_list=['sample_slicer'], 
                result_keys=None, filter=None, n_jobs=-1, verbose=1):
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
    # TODO: Optimize memory
    dir_analysis = os.listdir(path)
    dir_analysis = [d for d in dir_analysis if d.find(dir_id) != -1 and d.find(".") == -1]
    dir_analysis.sort()
    
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

   
    
def get_configuration_fields(conf, *args):
    """This function is used to collect fields from the configuration file.
    
    Parameters
    ----------
    conf : dictionary
        The configuration dictionary to be digged.

    args : list
        List of keywords to be found in the configuration file.
    
    Returns
    -------
    [type]
        [description]
    """
    # TODO: Complete documentation
    
    import ast
    results = dict()

    if 'id' in list(conf.keys()):
        results['id'] = conf['id']
    else:
        results['id'] = "None"
    
    for k, v in conf.items():
        for arg in args:

            if arg == k == 'ds__img_pattern':
                results[arg] = v
            
            if arg == k == 'prepro':
                value = ast.literal_eval(v)
                results[arg] = "_".join(value)
            
            idx_end = len(arg) + 2  # len("__")
            
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
                    value = [str(v) for v in value]
                    
                    if len(value) != 1:
                        value = "_".join(value)
                    else:
                        value = value[0]
                    
                results[str(k[idx_end:])] = value
                
    return results, ast.literal_eval(conf['scores'])




def get_searchlight_results(path, dir_id, field_list=['sample_slicer'], load_cv=False):
    
    dir_analysis = os.listdir(path)
    dir_analysis = [d for d in dir_analysis if d.find(dir_id) != -1 and d.find(".") == -1]
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
    

def get_connectivity_results(path, dir_id, field_list=['sample_slicer'], load_cv=False):
    
    dir_analysis = os.listdir(path)
    dir_analysis = [d for d in dir_analysis if d.find(dir_id) != -1 and d.find(".") == -1]
    dir_analysis.sort()
    
    results = []
    
    for d in tqdm(dir_analysis):
        # read json
        conf_fname = os.path.join(path, d, "configuration.json")
        with open(conf_fname) as f:
            conf = json.load(f)
        
        # TODO: Check if permutation is in fields
        fields, scores = get_configuration_fields(conf, *field_list)
        
        data = loadmat(os.path.join(path, d, "connectivity_data.mat"))

        fields['data'] = data['matrix']
        fields_ = fields.copy()           
            
        results.append(fields_)
    
    dataframe = pd.DataFrame(results)
       
    return dataframe



def filter_dataframe(dataframe, **selection_dict):
    # TODO: Documentation

    _symbols = ['!', '<', '>']

    selection_mask = np.ones(dataframe.shape[0], dtype=np.bool)
    for key, values in selection_dict.items():
        
                
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



def aggregate_searchlight(path, dir_id, filter):
    """This should be used for a within subject analysis 
    to collect data from different folders / subjects and
    collect results.

    Be aware of the different parameters of the analysis.

    So the best approach is to use get_searchlight_results
    and then use that to aggregate.
    
    Parameters
    ----------
    path : [type]
        [description]
    dir_id : [type]
        [description]
    
    """
    dataframe = get_searchlight_results(path, dir_id, field_list=['sample_slicer'], load_cv=False)
    
    return


def array2df(dataframe, key):
    # TODO : Documentation
    df = pd.DataFrame(dataframe[key].values.tolist(), 
                      columns=['%s_%d' %(key, i) \
                                    for i in range(dataframe[key].values[0].shape[0])])


    df_keys = pd.DataFrame([row[:-1] for row in dataframe.values.tolist()], 
                                    columns=dataframe.keys()[:-1])

    df_concat = pd.concat(df, df_keys)

    return df_concat


def query_rows(dataframe, keys, attr, fx=np.max):


    df_values = df_fx_over_keys(dataframe, keys, attr=attr, fx=fx)

    queried_df = []

    for i, row in df_values.iterrows():
        mask = np.ones(dataframe.shape[0], dtype=np.bool)
        for k in df_values.keys():
            mask = np.logical_and(mask, dataframe[k].values == row[k])

        queried_df.append(dataframe.loc[mask])


    return pd.concat(queried_df)




def df_fx_over_keys(dataframe, keys, attr='features', fx=lambda x:np.vstack(x).sum(0), **fx_kwargs):
    """This function perform a function on the dataframe, it groups the dataframe
    by using the key parameter and applies a function to values indicated.
    
    Parameters
    ----------
    dataframe : pandas Dataframe
        The dataframe to be processed by the function
    keys : list of string
        The keys that should be used to group the dataframe. These keys are those that
        were preserved in the output.
    attr : str, optional
        The key were values should be found (the default is 'features')
    fx : function, optional
        The function that is applied to values. (the default is lambda x:np.vstack(x).sum(0))
    
    Returns
    -------
    dataframe : The processed dataframe.
    """

    df_sum = dataframe.groupby(keys)[attr].apply(fx, **fx_kwargs)

    return df_sum.reset_index()


def get_weights(dataframe):

    from scipy.stats import zscore

    df_weights = []
    for i, row in dataframe.iterrows():
        matrix = np.zeros_like(row['features'], dtype=np.float)
        mask = np.equal(row['features'], 1)

        matrix[mask] = zscore(row['weights'])

        row['weights'] = matrix
        
        df_weights.append(row)

    return pd.DataFrame(df_weights)



def dataframe_to_afni():
    """This should return a command or similar to perform
    statistics in AFNI
    
    """

    return
