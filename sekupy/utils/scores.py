from scipy.stats import pearsonr
from sekupy.utils.math import dot_correlation
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np

def _get_correlation(y_true, y_pred):
    r = dot_correlation(y_pred, y_true)

    r = (r - r.min(axis=1, keepdims=True)) \
            / (r.max(axis=1, keepdims=True) \
              - r.min(axis=1, keepdims=True))

    return r


def aic(y_true, y_pred, k):

    resid = y_pred - y_true
    sse = resid**2

    score = 2*k - 2*np.log(sse)

    return score


def bic(y_true, y_pred, k):
    
    resid = y_pred - y_true
    sse = resid**2
    n = len(y_pred)

    score = n*np.log(sse/n) + k*np.log(n)

    return score


def correlation(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]


def pscore(y_true, y_pred):
    r = _get_correlation(y_true, y_pred)
    return np.mean(np.diag(r))


def identifiability(y_true, y_pred):
    r = _get_correlation(y_true, y_pred)
    i_self = np.mean(np.diag(r))

    id1 = np.triu_indices(r.shape[0], k=1)
    id2 = np.tril_indices(r.shape[0], k=1)

    i_diff = np.mean(np.hstack((r[id1], r[id2])))

    return i_self - i_diff


def fingerprint_accuracy(y_true, y_pred):
    r = _get_correlation(y_true, y_pred)
    
    prediction = np.argmax(r, axis=1)
    true = np.arange(r.shape[0])

    a = np.sum(prediction == true) / r.shape[0]

    return a


def r2_fingerprint(y_pred, y_true):
    print(y_pred[0].shape)
    return np.mean([r2_score(y_pred[i], y_true[i]) for i in range(y_pred.shape[0])])
    

def mse_fingerprint(y_pred, y_true):
    return np.mean([mean_squared_error(y_pred[i], y_true[i]) for i in range(y_pred.shape[0])])