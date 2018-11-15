import numpy as np

# TODO : Documentation
def z_fisher(r):
    
    F = 0.5*np.log((1+r)/(1-r))
    
    return F


def partial_correlation(X, Z):
    """
    Returns the partial correlation coefficients between 
    elements of X controlling for the elements in Z.
    """
 
     
    X = np.asarray(X).transpose()
    Z = np.asarray(Z).transpose()
    n = X.shape[1]
 
    partial_corr = np.zeros((n,n), dtype=np.float)
    
    for i in range(n):
        partial_corr[i,i] = 0
        for j in range(i+1,n):
            beta_i = np.linalg.lstsq(Z, X[:,j])[0]
            beta_j = np.linalg.lstsq(Z, X[:,i])[0]
 
            res_j = X[:,j] - Z.dot(beta_i)
            res_i = X[:,i] - Z.dot(beta_j)
 
            corr = np.corrcoef(res_i, res_j)
 
            partial_corr[i,j] = corr.item(0,1)
            partial_corr[j,i] = corr.item(0,1)
 
    return partial_corr