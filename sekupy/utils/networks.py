import os
import nibabel as ni
import numpy as np
from sekupy.utils.matrix import copy_matrix
import itertools
from scipy.stats.stats import zscore


def aggregate_networks(matrix, roi_list, aggregation_fx=np.sum):
    """
    Function used to aggregate matrix values using 
    aggregative information provided by roi_list
    
    Parameters
    ----------
    matrix : numpy 2D array, shape n x n
        Connectivity matrix in squared form
    roi_list : list of string, length = n
        List of each ROI's network name. Each element represents
        the network that includes the ROI in that particular position.
        
    Returns
    -------
    aggregate_matrix : numpy 2D array, p x p
        The matrix obtained, by pairwise network sum 
        of nodes within networks.
        
    """
    
    unique_rois = np.unique(roi_list)
    n_roi = unique_rois.shape[0]

    aggregate_matrix = np.zeros((n_roi, n_roi), dtype=np.float16)
    
    network_pairs = itertools.combinations(unique_rois, 2)
    indexes = np.vstack(np.triu_indices(n_roi, k=1)).T
    
    # This is to fill upper part of the aggregate matrix
    for i, (n1, n2) in enumerate(network_pairs):
        
        x = indexes[i][0]
        y = indexes[i][1]
        
        mask1 = roi_list == n1
        mask2 = roi_list == n2
        
        # Build the mask of the intersection between
        mask_roi = np.meshgrid(mask1, mask1)[1] * np.meshgrid(mask2, mask2)[0]
        
        value = aggregation_fx(matrix[np.nonzero(mask_roi)])
        #value /= np.sum(mask_roi)
        
        aggregate_matrix[x, y] = value
    
    # Copy matrix in the lower part
    aggregate_matrix = copy_matrix(aggregate_matrix)
    
    # This is to fill the diagonal with within-network sum of elements
    for i, n in enumerate(unique_rois):
        
        diag_matrix, mask_net = network_connections(matrix, n, roi_list)
        upper_mask = np.triu(mask_net, k=1)
        aggregate_matrix[i, i] = aggregation_fx(diag_matrix[np.nonzero(upper_mask)]) 
        # aggregate_matrix[i, i] = np.mean(diag_matrix) 
    
    return aggregate_matrix


def network_connections(matrix, label, roi_list, method='within'):
    """
    Function used to extract within- or between-networks values
    """
    
    mask1 = roi_list == label
    
    if method == 'within':
        mask2 = roi_list == label
    elif method != 'between':
        mask2 = roi_list == method
    else:
        mask2 = roi_list != label
    
    matrix_hori = np.meshgrid(mask1, mask1)[0] * np.meshgrid(mask2, mask2)[1]
    matrix_vert = np.meshgrid(mask1, mask1)[1] * np.meshgrid(mask2, mask2)[0]

    connections_ = matrix * (matrix_hori + matrix_vert)
    
    return connections_, matrix_hori + matrix_vert