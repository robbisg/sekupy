import numpy as np
from scipy.stats import pearsonr
from sekupy.utils.math import seed_correlation

def pearsonr_score(X, y):

    out = [pearsonr(x, y) for x in X.T]

    return np.abs(np.vstack(out)[:,0]), np.vstack(out)[:,1]


def positive_correlated(X, y):
    """[summary]

    Parameters
    ----------
    X : [type]
        [description]
    y : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    r, p = seed_correlation(X, y)

    p[r <= 0] = 1
    r[r <= 0] = 0

    return r, p

def negative_correlated(X, y):
    """[summary]

    Parameters
    ----------
    X : [type]
        [description]
    y : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    r, p = seed_correlation(X, y)

    p[r >= 0] = 1
    r[r >= 0] = 0

    return r, p
