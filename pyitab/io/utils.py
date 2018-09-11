import logging
import os
import nibabel as ni
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_ds_data(ds, target_attribute='targets'):
    """
    Returns X and y data from pymvpa dataset
    """
    
    return ds.samples, ds.sa[target_attribute].value


def add_subdirs(path, name, sub_dirs):
    
    complete_path = []
    
    for d in sub_dirs:
        if d == 'none':
            d = ''
        if d[0] == '/':
            complete_path.append(d)
        
        logger.debug("%s %s %s", path, name, d)
        pathname = os.path.join(path, name, d)
        
        logger.debug(pathname)
        
        if os.path.isdir(pathname):
            complete_path.append(pathname)
    
    
    complete_path.append(path)
    complete_path.append(os.path.join(path, name))
    

    return complete_path




def build_pathnames(path, name, sub_dirs):
            
    
    path_file_dirs = add_subdirs(path, name, sub_dirs)
    
    logger.debug(path_file_dirs)
    logger.info('Loading...')
    
    file_list = []
    # Verifying which type of task I've to classify (task or rest) 
    # and loads filename in different dirs
    for path_ in path_file_dirs:
        dir_list = [os.path.join(path_, f) for f in os.listdir(path_)]
        file_list = file_list + dir_list

    logger.debug('\n'.join(file_list))
    
    return file_list


def save_map(filename, map_np_array, affine=np.eye(4)):
        
    map_zscore = ni.Nifti1Image(map_np_array, affine)
    ni.save(map_zscore, filename)
    


def make_dir(path):
    """ Make dir unix command wrapper """
    #os.mkdir(os.path.join(path))
    command = 'mkdir -p '+os.path.join(path)
    logger.info(command)
    os.system(command)
    


def ds_to_dataframe(ds, keys=['band', 'targets', 'subjects'], melt=False):
    
    df_dict = {}
    df_list = []
    
    for sample in ds:
        for k in keys:
            df_dict[k] = sample.sa[k].value[0]
            
        sample_data = sample.samples.squeeze()
        feature_dict = {"feature_%04d"%(i+1) : v for i, v in enumerate(sample_data)}
            
        df_dict.update(feature_dict)
        dd = df_dict.copy()
        df_list.append(dd)
    
    df = pd.DataFrame(df_list)
    
    if melt:
        df = pd.melt(df, id_vars=keys, value_name='value')
    
    return df
        
