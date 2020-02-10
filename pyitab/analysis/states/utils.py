import os
import _pickle as pickle
import numpy as np
from scipy.spatial.distance import euclidean

from scipy.io import loadmat, savemat
from sklearn.cluster import KMeans
from nitime.timeseries import TimeSeries
from nitime.analysis import SpectralAnalyzer
from mvpa_itab.similarity.analysis import SeedAnalyzer
from mvpa_itab.conn.states.base import cluster_state

import logging
logger = logging.getLogger(__name__)

def get_data(filename):
    """
    Returns upper triangular matrix data and number of ROIs.
    The data is specific for the task, is a non-stationary connectivity
    matrix. 
    
    Parameters
    ----------
    filename : string
        The filename of a .mat matlab file in which the matrix is
        a data variable in matlab.
        
    Returns
    -------
    data : n_session x n_timepoints x n_connections numpy array
        The upper-part of the non-stationary matrix
        
    n_roi : int
        Number of ROI of the original matrix.
    """
    
    #filename = '/media/robbis/DATA/fmri/movie_viviana/mat_corr_sub_REST.mat'
    logger.info("Loading %s" %(filename))
    data = loadmat(filename)
    data = np.array(data['data'], dtype=np.float16)
    n_roi = data.shape[-1]
    ix, iy = np.triu_indices(data.shape[-1], k=1)
    data = data[:,:,ix,iy]
    
    return data, n_roi



def save_data(X, key, filename):
    logger.info("Saving %s data: %s" %(key, filename))
    savemat(filename, {key: X})






def filter_data(data, method='demean'):
    
    mean = np.mean(data, axis=2)
    data -= mean
    
    return



def get_centroids(X, labels):
    """
    Returns the centroid of a clustering experiment
    
    Parameters
    ----------
    X : n_samples x n_features array
        The full dataset used for clustering
    
    labels : n_samples array
        The clustering labels for each sample.
        
        
    Returns
    -------
    centroids : n_cluster x n_features shaped array
        The centroids of the clusters.
    """
    
    return np.array([X[labels == l].mean(0) for l in np.unique(labels)])


def fit_centroids(X, centroids):
    """
    Function used to transform centroids to original data. 
    Given centroids it fits them to each subject data.
    
    Parameters
    ----------
    X : n_samples x n_timepoints x n_features array
        The full dataset used for clustering
    
    centroids : n_states x n_features array
        The centroids for each state.
        
        
    Returns
    -------
    centroids : n_samples x n_timepoints shaped array
        The list of most similar state for each subject and each timepoint. 
    """
    
    k = centroids.shape[0]
    
    results_ = []
    
    for subj in X:
        km = KMeans(n_clusters=k,
                    init=centroids).fit(subj)
                    
        results_.append(km.labels_)
        
    return np.array(results_)




def fit_states(X, centroids, distance=euclidean):
    """
    Returns the similarity of the dataset to each centroid,
    using a dissimilarity distance function.
    
    Parameters
    ----------
    X : n_samples x n_features array
        The full dataset used for clustering
    
    centroids : n_cluster x n_features array
        The cluster centroids.
        
    distance : a scipy.spatial.distance function | default: euclidean
        This is the dissimilarity measure, this should be a python
        function, see scipy.spatial.distance.
        
    
    Returns
    -------
    results : n_samples x n_centroids array
        The result of the analysis,
    
    """
    

    ts_seed = TimeSeries(centroids, sampling_interval=1.)
    
    results_ = []
    
    for subj in X:
        ts_target = TimeSeries(subj, sampling_interval=1.)
        S = SeedAnalyzer(ts_seed, ts_target, distance)
        results_.append(S.measure)
        
    
    return results_



def get_state_frequencies(state_dynamics, method='spectrum_fourier'):
    """
    Returns the spectrum of the state occurence for each subject.
    
    Parameters
    ----------
    state_dynamics :    n_states x n_subjects x n_timepoints array
                        The state dynamics output from fit_states
                        function.
                        
    method : a string, check nitime.spectral.SpectralAnalyzer for 
             allowed methods.
    
    
    Returns
    -------
    results : n_subjects list of tuple,
              first element is the array of frequencies,
              second element is the array n_states x frequencies
              of the spectrum.
    
    """
    
    results = []
    for s in state_dynamics:
        ts = TimeSeries(s, sampling_interval=1.)
        S = SpectralAnalyzer(ts)
        try:
            result = getattr(S, method)
        except AttributeError as  _:
            result = S.spectrum_fourier
        
        results.append(result)
        
    return results



def get_transition_matrix(group_fitted_timecourse):
    """
    Extract the probability transition matrix given the
    fitted timecourse for each subject.
    This is the output of fit_centroids function
    
    Parameters
    ----------
    group_fitted_timecourse :   n_subjects x n_timepoints array
                                The state dynamics output from fit_centroids
                                function.
                        
    
    
    Returns
    -------
    transition_p : n_states x n_states matrix,
                   The probabilities of transition from x state to y state, stored
                    in matrix[x,y]. The syntax is matrix[starting_state, ending_state].
    """
    
    n_states = group_fitted_timecourse.max()
    transitions = np.zeros((n_states+1, n_states+1))
    for fitted_timecourse in group_fitted_timecourse:
        for i, state in enumerate(fitted_timecourse):
            if (i+1) < len(fitted_timecourse):
                transitions[state, fitted_timecourse[i+1]] += 1
    
    
    return transitions   


def get_state_duration(group_fitted_timecourse):
    """
    Extract the average of state duration given the
    fitted timecourse for each subject.
    This is the output of fit_centroids function
    
    Parameters
    ----------
    group_fitted_timecourse :   n_subjects x n_timepoints array
                                The state dynamics output from fit_centroids
                                function.
                        
    
    
    Returns
    -------
    subj_state_duration : nsubjects x n_states matrix,
                   The duration of each state for each subject/session.
    mean_state_duration : nstates array
                   The average duration for each state.
    """        
    
    counter = 0
    subj_state_duration = []
    mean_subject_duration = []
    
    n_states = np.max(group_fitted_timecourse)+1
    
    for _, fitted_tc in enumerate(group_fitted_timecourse):
        state_duration = []
        
        for i, state in enumerate(fitted_tc):
            if (i+1) < len(fitted_tc):
                if state == fitted_tc[i+1]:
                    counter += 1
                else:
                    counter += 1
                    state_duration.append([state+1, np.float(counter)])
                    counter = 0
        
        subj_state_duration.append(state_duration)
        state_duration = np.array(state_duration)
        mean_subject_duration.append([[i+1, state_duration[state_duration[:,0]==(i+1)].mean(0)[1]]
                                       for i in range(n_states)])
    
    subj_state_duration = np.array(subj_state_duration)
    mean_subject_duration = np.array(mean_subject_duration)
    
    mean_subject_duration[:,:,1][np.isnan(mean_subject_duration[:,:,1])] = 0
    
    return subj_state_duration, mean_subject_duration




       
        
        