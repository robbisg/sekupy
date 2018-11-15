import numpy as np

# TODO: Documentation
def array_to_matrix(array, nan_mask=None, copy=True, **kwargs):
    
    if nan_mask is None:
        # second degree resolution to get matrix dimensions #
        c = -2*array.shape[0]
        a = 1
        b = -1
        det = b*b - 4*a*c
        rows = np.int((-b + np.sqrt(det))/(2*a))
        
        matrix = np.ones((rows, rows))
    else:
        matrix = np.float_(np.logical_not(nan_mask))
    
    il = np.tril_indices(matrix.shape[0])
    matrix[il] = 0
    
    matrix[np.nonzero(matrix)] = array

    if copy:
        copy_matrix(matrix, **kwargs)
    
    return matrix


def flatten_matrix(matrix):
    
    il = np.tril_indices(matrix.shape[0])
    out_matrix = matrix.copy()
    out_matrix[il] = np.nan
    
    out_matrix[range(matrix.shape[0]),range(matrix.shape[0])] = np.nan

    return matrix[~np.isnan(out_matrix)]



def copy_matrix(matrix, diagonal_filler=0):

    iu = np.triu_indices(matrix.shape[0])
    il = np.tril_indices(matrix.shape[0])

    matrix[il] = diagonal_filler

    for i, j in zip(iu[0], iu[1]):
        matrix[j, i] = matrix[i, j]

    return matrix

