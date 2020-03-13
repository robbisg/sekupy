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


def similiarity(seed, target, measure, **kwargs):

    # If there is more than one channel in the seed time-series:
    if len(seed.shape) > 1:

        # Preallocate results
        Cxy = np.empty((seed.shape[0],
                        target.shape[0]), dtype=np.float)

        for seed_idx, this_seed in enumerate(seed):
            res = []
            for single_ts in target:
                
                try:
                    measure_sim = measure(this_seed, single_ts, **kwargs)
                    res.append(measure_sim)
                except TypeError as  _:
                    raise TypeError('Class measure must take 2 arguments!')
            
            Cxy[seed_idx] = np.array(res)
    # In the case where there is only one channel in the seed time-series:
    else:
        #To correct!!!
        len_target = target.shape[0]
        rr = [measure(seed, target[i]) for i in range(len_target)]
        Cxy = np.array(rr)
        
        
    return Cxy.squeeze()