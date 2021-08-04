import numpy as np

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