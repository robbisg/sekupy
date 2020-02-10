import numpy as np
from sklearn.cluster import KMeans
from scipy.signal import argrelextrema, argrelmin
from scipy.spatial.distance import pdist, squareform
import logging



logger = logging.getLogger(__name__)


def clustering(X, n_cluster):
    """
    Function used to cluster data using kmeans and
    a fixed number of cluster.
    
    Returns the labels
    """
    
    km = KMeans(n_clusters=n_cluster).fit(X)
    return km.labels_



def cluster_state(X, k_range=range(2, 15)):
    """
    This performs either preprocessing and clustering
    using a range of k-clusters.
    
    Returns the preprocessed dataset and a list of labels.
    """

    
    clustering_ = []
    
    if k_range[0] < 2:
        k_range = range(2, k_range[-1])
    
    for k in k_range:
        logger.info('Clustering with k: '+str(k))
        labels = clustering(X, k)
        clustering_.append(labels)
        
    return X, clustering_



def get_extrema_histogram(arg_extrema, n_timepoints):
    
    hist_arg = np.zeros(n_timepoints)
    n_subjects = len(np.unique(arg_extrema[0]))
    
    for i in range(n_subjects):
        sub_max_arg = arg_extrema[1][arg_extrema[0] == i]
        hist_arg[sub_max_arg] += 1
        
    return hist_arg



def subsample_data(data, method='speed', peak='min'):
    """
    Function used to select timepoints using 
    speed methods (low velocity states) or 
    variance methods (high variable states)
    
    Returns the preprocessed dataset
    """
    
    peak_mapper = {'max': np.greater_equal,
                   'min': np.less_equal}
    
    
    
    method = get_subsampling_measure(method)
    _, measure = method(data)
    
    peaks = argrelextrema(measure, peak_mapper[peak], axis=1, order=5)
    
    X = data[peaks]
    
    
    return X



def get_subsampling_measure(method):
    
    method_mapping = {
                      'speed': get_min_speed_arguments,
                      'variance': get_max_variance_arguments
                      }
    
    
    return method_mapping[method]



def get_max_variance_arguments(data):
    """
    From the data it extract the points with high local variance 
    and returns the arguments of these points and the 
    variance for each point.
    """
    
    stdev_data = data.std(axis=2)   
    arg_maxima = argrelextrema(np.array(stdev_data), np.greater, axis=1)
    
    
    return arg_maxima, stdev_data
    


def get_min_speed_arguments(data):
    """
    From the data it extract the points with low local velocity 
    and returns the arguments of these points and the 
    speed for each point.    
    """
    
    subj_speed = []
    for i in range(data.shape[0]):
        distance_ = squareform(pdist(data[i], 'euclidean'))
        
        speed_ = [distance_[i, i+1] for i in range(distance_.shape[0]-1)]
        subj_speed.append(np.array(speed_))
    
    subj_speed = np.vstack(subj_speed)
    subj_min_speed = argrelmin(np.array(subj_speed), axis=1)
    
    return subj_min_speed, subj_speed







    
       

