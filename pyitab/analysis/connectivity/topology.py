import numpy as np

def degree(X):
    return np.sum(X, axis=0)

def n_triangles(X):

    for i in np.arange(X.shape[0]):
        break