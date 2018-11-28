import numpy as np
import os
import pandas as pd


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


def get_ds_data(ds, target_attribute='targets'):
    """
    Returns X and y data from pymvpa dataset
    """

    return ds.samples, ds.sa[target_attribute].value



def temporal_transformation(X, y, time_attr):
    """[summary]
    
    Parameters
    ----------
    X : [type]
        [description]
    y : [type]
        [description]
    time_attr : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    """

    times = np.unique(time_attr)

    X_ = X.reshape(-1, len(times), X.shape[1])
    X_ = np.rollaxis(X_, 1, 3)

    y_ = temporal_attribute_reshaping(y, time_attr)

    return X_, y_


def temporal_attribute_reshaping(attribute_list, time_attribute):
    """[summary]
    
    Parameters
    ----------
    attribute_list : [type]
        [description]
    time_attribute : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    """

    times = np.unique(time_attribute)

    y = attribute_list.reshape(-1, len(times))
    labels = []
    for yy in y:
        l, c = np.unique(yy, return_counts=True)
        labels.append(l[np.argmax(c)])

    return np.array(labels)