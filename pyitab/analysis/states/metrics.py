from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import scipy
from scipy.spatial.distance import pdist, squareform, euclidean, \
    correlation, cosine
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)



def get_k(labels):
    return len(np.unique(labels))



def get_centers(X, labels):
    return np.array([X[labels == l].mean(0) for l in np.unique(labels)])



def ch_criterion(X, labels, distance=euclidean):
        
    k = get_k(labels)
    n = X.shape[0]
    
    b = bgss(X, labels, distance)
    w = wgss(X, labels, distance)
    
    return (n - k)/(k - 1) * b / w


def bgss(X, labels, distance=euclidean):
    
    ds_mean = X.mean(0)
    ss = 0
    for i in np.unique(labels):
        cluster_data = X[labels == i]
        ss += (distance(cluster_data.mean(0), ds_mean) ** 2) * cluster_data.shape[0]
        
    return ss



def wgss(X, labels, distance=euclidean):
    
    ss = 0
    for i in np.unique(labels):
        cluster_data = X[labels == i]
        cluster_mean = cluster_data.mean(0)
        css = 0
        for x in cluster_data:
            css += distance(x, cluster_mean) ** 2
        
        ss += css
        
    return ss
            


def kl_criterion(X, labels, previous_labels=None, next_labels=None, precomputed=True):
    
    n_cluster = get_k(labels)
    
    """
    if n_cluster <= 1 or previous_labels==None or next_labels==None:
        return 0
    """
    
    n_prev_clusters = len(np.unique(previous_labels))
    n_next_clusters = len(np.unique(next_labels))
    
    if n_cluster != n_next_clusters-1 or n_prev_clusters+1 != n_cluster:
        return 0
    
    M_previous = m(X, previous_labels, precomputed=precomputed)
    M_next = m(X, next_labels, precomputed=precomputed)
    M_current = m(X, labels, precomputed=precomputed)

    return 1 - 2 * M_current/M_previous + M_next/M_previous



def W(X, labels, precomputed=True):

    #distance = squareform(pdist(X, 'euclidean'))
    import itertools
    w_ = 0
    for k in np.unique(labels):
        cluster_ = labels == k
        
        if precomputed == True:
            index_cluster = np.nonzero(cluster_)
            combinations_ = itertools.combinations(index_cluster[0], 2)
            nrow = cluster_.shape[0]
            array_indices = [get_triu_array_index(n[0], n[1], nrow) for n in combinations_]
            cluster_dispersion = X[array_indices].sum()
        else:
            
            X_k = X[cluster_]
            cluster_distance = squareform(pdist(X_k, 'euclidean'))
            #cluster_distance = distance[cluster_,:][:,cluster_]
            upper_index = np.triu_indices(cluster_distance.shape[0], k=1)
            cluster_dispersion = cluster_distance[upper_index].sum()
        
        w_ += (cluster_dispersion ** 2) * 0.5 * 1./cluster_.shape[0]
    return w_
    


def m(X, labels, precomputed=True):
    n_cluster = get_k(labels)
    return W(X, labels, precomputed=precomputed) * np.power(n_cluster, 2./X.shape[1])

    
    
def gap(X, labels, nrefs=20, refs=None):
    """
    Compute the Gap statistic for an nxm dataset in X.
    Either give a precomputed set of reference distributions in refs as an (n,m,k) scipy array,
    or state the number k of reference distributions in nrefs for automatic generation with a
    uniformed distribution within the bounding box of X.
    Give the list of k-values for which you want to compute the statistic in ks.
    """
    shape = X.shape
    if refs==None:
        tops = X.max(axis=0)
        bots = X.min(axis=0)
        dists = scipy.matrix(scipy.diag(tops-bots))
        
    
        rands = scipy.random.random_sample(size=(shape[0],shape[1],nrefs))
        for i in range(nrefs):
            rands[:,:,i] = rands[:,:,i]*dists+bots
    else:
        rands = refs
    
    k = get_k(labels)
    kml = labels
    kmc = np.array([X[labels == l].mean(0) for l in np.unique(labels)])

    disp = sum([euclidean(X[m,:], kmc[kml[m],:]) for m in range(shape[0])])

    refdisps = scipy.zeros((rands.shape[2],))
    for j in range(rands.shape[2]):
        km = KMeans(n_clusters=k).transform(rands[:,:,j])
        kml = km.labels_
        kmc = km.cluster_centers_
        refdisps[j] = sum([euclidean(rands[m,:,j], kmc[kml[m],:]) for m in range(shape[0])])
        
    gaps = scipy.log(scipy.mean(refdisps))-scipy.log(disp)
        
    return gaps



def explained_variance(X, labels):
    
    explained_variance_ = 0
    k_ = get_k(labels)
    great_avg = X.mean()
    
    for i in np.unique(labels):
        
        cluster_mask = labels == i
        group_avg = X[cluster_mask].mean()
        n_group = np.count_nonzero(cluster_mask)
        
        group_var = n_group * np.power((group_avg - great_avg), 2) / (k_ - 1)
        
        explained_variance_ += group_var
        
    return explained_variance_
        


def global_explained_variance(X, labels):
    
    # Get the centroids for each cluster
    centroids = np.array([X[labels == l].mean(0) for l in np.unique(labels)])
    
    # Compute the global power
    global_conn_power = X.std(axis=1)
    denominator_ = np.sum(global_conn_power**2)
    
    numerator_ = 0
    
    for i, conn_pwr in enumerate(global_conn_power):
        
        k_map = centroids[labels[i]]

        if k_map.shape[0] == 2:
            corr_ = 1 - cosine(X[i], k_map)
        else:
            corr_ = scipy.stats.pearsonr(X[i], k_map)[0]
        
        numerator_ += np.power((conn_pwr * corr_), 2)
        
    return numerator_/denominator_




def cross_validation_index(X, labels):
    
    n_maps = get_k(labels)
    n_points = X.shape[0]
    
    centroids = np.array([X[labels == l].mean(0) for l in np.unique(labels)])
    
    sum_ = 0
    for i, u in enumerate(X):
        sum_ += euclidean(u, u)**2 - np.dot(centroids[labels[i]], u) ** 2
        
    
    cv_criterion = sum_/(n_points**2 - 1) * ((n_points - 1)/(n_points - n_maps - 1))**2
    
    return cv_criterion



def get_triu_array_index(i, j, n_row):
    return (n_row*i+j)-np.sum([(s+1) for s in range(i+1)])




def index_i(X, labels):
    
    k = get_k(labels)
    if k == 2:
        return 0
    
    centroids = get_centers(X, labels)
    center = X.mean(0)
    
    ek = 0.
    e1 = 0.
    for i, x in enumerate(X):
        
        ek += euclidean(x, centroids[labels[i]])
        e1 += euclidean(x, center)
        
    pair_dist = pdist(np.vstack(centroids), 'euclidean')
    
    dk = pair_dist.max()
    mk = pair_dist.min()

    i_ = (1./k * np.float(e1)/ek * dk * mk)**2

    return i_        


metrics = {'silhouette': metrics.silhouette_score,
           'kl': kl_criterion,
           'wgss': wgss,
           'gev': global_explained_variance,
           'ev': explained_variance,
           'ch': ch_criterion
           }
           