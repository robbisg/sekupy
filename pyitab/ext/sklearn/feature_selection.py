import numpy as np
from scipy.stats import pearsonr

def pearsonr_score(X, y):

    out = [pearsonr(x, y) for x in X.T]

    return np.abs(np.vstack(out)[:,0]), np.vstack(out)[:,1]
