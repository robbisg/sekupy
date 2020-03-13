import pandas as pd
import numpy as np

import logging
logger = logging.getLogger(__name__)

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




def apply_function(dataframe, keys, attr='features', fx=lambda x:np.vstack(x).sum(0), **fx_kwargs):
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


def clean_dataframe(dataframe, keys=[]):
    """Clean columns that are not informative.
    A list with keys can be provided to delete unuseful
    columns.
    
    Parameters
    ----------
    dataframe : [type]
        [description]
    """
    from collections import Counter

    if keys == []:
        keys = dataframe.keys()
      
    for k in keys:

        if 'score' in k:
            continue

        logger.debug(k)

        unique = Counter(dataframe[k].values).keys()

        if len(unique) == 1:
            dataframe = dataframe.drop(k, axis=1)

    return dataframe
