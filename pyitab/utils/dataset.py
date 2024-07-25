import numpy as np
import os
import pandas as pd


def ds_to_dataframe(ds, keys=None, melt=False):
    """[summary]
    
    Parameters
    ----------
    ds : [type]
        [description]
    keys : list, optional
        [description] (the default is ['band', 'targets', 'subjects'], which [default_description])
    melt : bool, optional
        [description] (the default is False, which [default_description])
    
    Returns
    -------
    [type]
        [description]
    """

    
    df_dict = {}
    df_list = []

    if keys is None:
        keys = list(ds.sa.keys())
    
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
    """This function simpy returns X and y for scikit-learn analyses
    starting from a pymvpa dataset.
    
    Parameters
    ----------
    ds : dataset ``pyitab.dataset.base.Dataset``
        The dataset in pymvpa format
    target_attribute : str, optional
        The sample attribute to be used to extract labels
         (the default is 'targets', which [default_description])
    
    Returns
    -------
    X, y
        A tuple with the X data matrix (samples x features) and the y
        array of labels.
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