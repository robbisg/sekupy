import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin


def gaussian_kernel(dist, dc):
    """[summary]
    
    Parameters
    ----------
    dist : [type]
        [description]
    dc : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    n_samples = dist.shape[0]
    
    rho = np.zeros(n_samples)
    
    m_indices = np.triu_indices(n_samples, k=1)
    
    # Gaussian kernel
    for i, j in np.vstack(m_indices).T:
        gaussian_k = np.exp(-(dist[i,j]/dc)*(dist[i,j]/dc))
        rho[i] = rho[i] + gaussian_k
        rho[j] = rho[j] + gaussian_k
    
    return rho



def cutoff(dist, dc):
    """[summary]
    
    Parameters
    ----------
    dist : [type]
        [description]
    dc : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    n_samples = dist.shape[0]
    
    rho = np.zeros(n_samples)
    
    m_indices = np.triu_indices(n_samples, k=1)
    
    # Gaussian kernel
    for i, j in np.vstack(m_indices).T:
        if dist[i, j] < dc:
            rho[i] += 1
            rho[j] += 1

    return rho    


class PeakDensityClustering(BaseEstimator, ClusterMixin, TransformerMixin):
    
    def __init__(self, dc='percentage', percentage=2., 
                 cluster_threshold=12., rhofx=gaussian_kernel):
        """[summary]
        
        Parameters
        ----------
        dc : str, optional
            [description], by default 'percentage'
        percentage : [type], optional
            [description], by default 2.
        cluster_threshold : [type], optional
            [description], by default 12.
        rhofx : [type], optional
            [description], by default gaussian_kernel
        
        Returns
        -------
        [type]
            [description]
        """
        
        if dc != 'percentage':
            self.dc = dc
        else:
            self.dc = 0
            
        self.perc = percentage
        self.cluster_threshold = cluster_threshold
        self.labels_ = None
        self.rhofx = rhofx


    def _get_threshold(self):

        if 1 <= self.cluster_threshold < 4.:
            metric = self.rho_ * self.delta_
            threshold = np.mean(metric) + \
                self.cluster_threshold * np.std(metric) 
        elif self.cluster_threshold < 1.:
            threshold = np.max(self.rho_) * np.max(self.delta_) * \
                 self.cluster_threshold
        else:
            threshold = self.cluster_threshold

        return threshold
             
    
    def _compute_distance(self, X):
        """[summary]
        
        Parameters
        ----------
        X : [type]
            [description]
        
        Returns
        -------
        [type]
            [description]
        """
        
        n_samples = X.shape[0]
        
        xdist = pdist(X) 
        dist = squareform(xdist)
        
        if self.dc == 0:
            position = int(round(n_samples * self.perc/100.))
            self.dc = np.sort(xdist)[position]
            
        return dist
    
    
    def _compute_rho(self, dist, dc):
        return self.rhofx(dist, dc)
    
    
    def _compute_delta(self, n_samples):

        rho = self.rho_
        dist = self.dist_
        
        ordrho = np.argsort(rho)[::-1]
        delta = np.zeros(n_samples)
        nneigh = np.zeros(n_samples)
        
        delta[ordrho[0]] = -1
        nneigh[ordrho[0]] = 0
    
        for i in range(n_samples):
            
            min_rho_mask = rho >= rho[i]
            
            min_dist = dist[i][min_rho_mask]
            nonzero = np.nonzero(min_dist)
            delta[i] = np.max(delta)
            if np.count_nonzero(min_rho_mask) != 1:
            
                delta[i] = np.min(min_dist[nonzero])
                ind = np.where(dist == delta[i])
                nneigh[i] = ind[0][0]
                if ind[0][0] == i:
                    nneigh[i] = ind[1][0]
        
        return delta, nneigh
    
    
    def _assign_cluster(self, n_samples, cluster_idx):
        
        rho = self.rho_
        dist = self.dist_
        
        clustering = np.zeros_like(rho)
        
        clustering = np.zeros_like(rho)
        clustering[cluster_idx] = cluster_idx
    
        for idx in range(n_samples):
            if clustering[idx] == 0:
                argmin = np.argmin(dist[idx, cluster_idx])
                clustering[idx] = cluster_idx[argmin]
    
        clustering = np.int_(clustering)
        
        return clustering
        
        
    def _compute_halo(self, n_samples, cluster_idx):
        
        rho = self.rho_
        dist = self.dist_
        dc = self.dc
        
        clustering = self.labels_
        
        halo = self.labels_.copy()
        n_cluster = len(cluster_idx) 
        
        if n_cluster > 1:
            bord_rho = np.zeros(n_cluster)
        
        m_indices = np.vstack(np.tril_indices(n_samples, k=1)).T
        
        # Gaussian kernel
        for i, j in m_indices:
            
            if clustering[i] != clustering[j] and dist[i, j] <= dc:
                rho_aver = 0.5*(rho[i]+rho[j])
    
                idc = np.argwhere(cluster_idx == clustering[i])
                if rho_aver > bord_rho[idc]:
                    bord_rho[idc] = rho_aver
                
                jdc = np.argwhere(cluster_idx == clustering[j])
                if rho_aver > bord_rho[jdc]:
                    bord_rho[jdc] = rho_aver
                
        for i in range(n_samples):
            idc = np.argwhere(cluster_idx == clustering[i])
            if rho[i] < bord_rho[idc]:
                halo[i] = 0
        
        return halo
    
    
    def fit(self, X, y=None):
        """[summary]
        
        Parameters
        ----------
        X : [type]
            [description]
        y : [type], optional
            [description], by default None
        
        Returns
        -------
        [type]
            [description]
        """
        
        n_samples = X.shape[0]
    
        self.dist_ = self._compute_distance(X)
        self.rho_ = self._compute_rho(self.dist_, self.dc)
        self.delta_, self.nn_ = self._compute_delta(n_samples)
        
        # Get centers
        self.threshold = self._get_threshold()
        cluster_idx = np.nonzero(self.delta_ * self.rho_ > self.threshold)[0]
        self.cluster_centers_ = X[cluster_idx]
        
        self.labels_ = self._assign_cluster(n_samples, cluster_idx)
        self.halo_ = self._compute_halo(n_samples, cluster_idx)

        return self