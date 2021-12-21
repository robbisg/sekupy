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

    df_concat = pd.concat([df, df_keys])

    return df_concat


def query_rows(dataframe, keys, attr, fx=np.max):
    """[summary]
    
    Parameters
    ----------
    dataframe : [type]
        [description]
    keys : [type]
        [description]
    attr : [type]
        [description]
    fx : [type], optional
        [description], by default np.max
    
    Returns
    -------
    [type]
        [description]
    """


    df_values = apply_function(dataframe, keys, attr=attr, fx=fx)

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



def dataframe_slicer(data, row=None, col=None, hue=None):
    """Generator for name indices and data subsets for each unique value
    of row, col, hue.

    Adaptively stolen from `seaborn`.

    Yields
    ------
    (i, j, k), data_ijk : tuple of ints, DataFrame
        The ints provide an index into the {row, col, hue}_names attribute,
        and the dataframe contains a subset of the full data corresponding
        to each facet. The generator yields subsets that correspond with
        the self.axes.flat iterator, or self.axes[i, j] when `col_wrap`
        is None.

    """
    from seaborn._core import categorical_order
    from itertools import product

    # Construct masks for the row variable
    if row is not None:
        row_names = categorical_order(data[row])
        row_masks = [data[row] == n for n in row_names]
    else:
        row_masks = [np.repeat(True, len(data))]

    # Construct masks for the column variable
    if col is not None:
        col_names = categorical_order(data[col])
        col_masks = [data[col] == n for n in col_names]
    else:
        col_masks = [np.repeat(True, len(data))]

    # Construct masks for the hue variable
    if hue is not None:
        hue_names = categorical_order(data[hue])
        hue_masks = [data[hue] == n for n in hue_names]
    else:
        hue_masks = [np.repeat(True, len(data))]

    # Here is the main generator loop
    for (i, row), (j, col), (k, hue) in product(enumerate(row_masks),
                                                enumerate(col_masks),
                                                enumerate(hue_masks)):
        data_ijk = data[row & col & hue ] # Check null
        yield (i, j, k), data_ijk