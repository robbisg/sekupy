from nilearn.decoding.searchlight import GroupIterator
from sklearn.externals.joblib.parallel import Parallel, delayed
import numpy as np
import time
from scipy.sparse.coo import coo_matrix
import sys


def _parallel_trajectory(X, dist, A, n_jobs=-1, verbose=0):
    """Function for computing a searchlight trajectory"""
    group_iter = GroupIterator(A.shape[0], n_jobs)
    scores = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_trajectory)(
                            A.rows[list_i],
                            dist, X, 
                            thread_id + 1, 
                            A.shape[0], 
                            verbose)
        for thread_id, list_i in enumerate(group_iter))
    return np.concatenate(scores).T



def _trajectory(list_rows, dist, X, thread_id, total, verbose=0):
    """Function for grouped iterations of trajectory searchlight

    Parameters
    -----------
    list_rows : array of arrays of int
        adjacency rows. For a voxel with index i in X, list_rows[i] is the list
        of neighboring voxels indices (in X).

    dist : distance measure
        object to use to fit the data

    X : array-like of shape at least 2D
        data to fit.

    thread_id : int
        process id, used for display.

    total : int
        Total number of voxels, used for display

    verbose : int, optional
        The verbosity level. Defaut is 0

    Returns
    -------
    par_scores : numpy.ndarray
        score for each voxel. dtype: float64.
    """
    scores = np.zeros((len(list_rows), X.shape[0]-1))
    t0 = time.time()
    for i, row in enumerate(list_rows):       
        
        X_ = X[:,row]
        scores[i] = np.array([dist(X_[i+1], X_[i]) for i in range(X_.shape[0]-1)])
        
        if verbose > 0:
            print_verbose(t0, i, list_rows, total, thread_id, verbose)

    
    return scores

def get_roi_adjacency(ds, rois):
    
    
    if rois == None:
        rois = [r for r in ds.fa.keys() if r!='voxel_indices']
    
    data = []
    roi_list = []
    
    for roi in rois:
        
        roi_values = np.unique(ds.fa[roi].value)
        if len(roi_values) > 1:
            roi_values = roi_values[1:]
            
        
        for c, r in enumerate(roi_values):
            
            nz = np.nonzero(ds.fa[roi].value == r)[0]
            
            rows = [[c, r] for r in nz]
            data.append(rows)
            roi_list.append("%s_%d" %(roi, r))
    
    
    data = np.array(data)
    
    ones = np.ones_like(data[:,0])
    B = coo_matrix((ones, (data[:,0], data[:,1]))).tolil()
    
    # TODO: Save proximity
    
    return B, roi_list
    
    


def print_verbose(t0, i, list_rows, total, thread_id, verbose):
    # One can't print less than each 10 iterations
    step = 11 - min(verbose, 10)
    if (i % step == 0):
        # If there is only one job, progress information is fixed
        if total == len(list_rows):
            crlf = "\r"
        else:
            crlf = "\n"
        percent = float(i) / len(list_rows)
        percent = round(percent * 100, 2)
        dt = time.time() - t0
        # We use a max to avoid a division by zero
        remaining = (100. - percent) / max(0.01, percent) * dt
        sys.stderr.write(
            "Job #%d, processed %d/%d voxels "
            "(%0.2f%%, %i seconds remaining)%s"
            % (thread_id, i, len(list_rows), percent, remaining, crlf))