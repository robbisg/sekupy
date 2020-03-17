import pandas as pd
import json
import os
import numpy as np
from scipy.io import loadmat
from joblib import Parallel, delayed
from pyitab.analysis.results.bids import get_configuration_fields, get_dictionary, \
    find_directory
from pyitab.analysis.results.base import filter_dataframe
from sklearn import metrics
import logging

logger = logging.getLogger(__name__)


def purge_dataframe(data, keys=['ds.a.snr', 
                                'ds.a.time', 
                                'ds.a.states',
                                'n_clusters', 
                                'n_components']):
    """
    keys = ['ds.a.snr', 'ds.a.time', 'ds.a.states', 
            'n_clusters', 'n_components']
    """
    
    n_clusters = np.zeros_like(data['n_clusters'], dtype=np.int8)

    for k in keys:

        if k == 'n_components' or k == 'n_clusters':
            data[k][data[k].values == 'None'] = np.nan
            values = pd.to_numeric(data[k]).values
            mask = np.logical_not(np.isnan(values))
            n_clusters[mask] = values[mask]
            continue
                    
        values = []
        for i, v in enumerate(data[k].values):
            if k == 'ds.a.states':
                for s in ['array', '\n', ' ']:
                    v = v.replace(s, '')
                n_state = np.int(data['subject'].values[i]) - 1
                v = np.array(np.safe_eval(v))[n_state]
            else:
                v = np.safe_eval(v)[0]
                if k == 'ds.a.time':
                    v = v[0]

            values.append(v)
        
        data[k[5:]] = values
    
    data['n_states'] = n_clusters

    return data.drop(columns=keys, axis=1)


def get_values_states(path, directory, field_list, result_keys):


    dir_path = os.path.join(path, directory)

    conf_fname = os.path.join(dir_path, "configuration.json")
    
    with open(conf_fname) as f:
        conf = json.load(f)
    
    fields, _ = get_configuration_fields(conf, *field_list)
    
    files = os.listdir(dir_path)
    files = [f for f in files if f.find(".mat") != -1]
    
    results = []
    #logger.debug(files)

    for fname in files:
        fname_fields = get_dictionary(fname)
        fields.update(fname_fields)
        #logger.debug(fields)
        
        data = loadmat(os.path.join(dir_path, fname), squeeze_me=True)
        data_dict = dict(zip(data['data'].dtype.names, data['data'].tolist()))

        fields.update(data_dict)
        results.append(fields.copy())

    return results


def calculate_metrics(dataframe, metrics_kwargs=None, fixed_variables={}):
    """This function calculates the metrics that will be used
    to identify the number of clusters. 
    
    Parameters
    ----------
    dataframe : a pandas dataframe
        The dataframe must contain the field ```n_states```, it is supposed
        to have other fields with unique values (e.g. those that identifies the analysis)        
    metrics_kwargs : dictionary, optional
        dictionary with other metrics, , by default None
    fixed_variables : dict, optional
        Variables that will be added to output dataframe, usually they are fields
        that identify a single analysis, and are used to filter the dataframe, by default {}
    
    Returns
    -------
    dataframe:
        A dataframe with a number of rows equal to the number of k specified in 
        n_states column of the original dataframe. 
        The single entry is composed by the metric name, the metric value and 
        the associated k, plus fixed_variables given as input parameter.
    """
    
    # Filtered dataframe
    from pyitab.analysis.states.metrics import kl_criterion, global_explained_variance,\
        wgss, explained_variance, index_i, cross_validation_index
    default_metrics = {'Silhouette': metrics.silhouette_score,
                        'Krzanowski-Lai': kl_criterion,
                        'Global Explained Variance': global_explained_variance,
                        'Within Group Sum of Squares': wgss,
                        'Explained Variance': explained_variance,
                        'Index I': index_i,
                        "Cross-validation":cross_validation_index
                        }

    X = dataframe['X'].values[0]
    clustering_labels = dataframe['labels'].values

    assert X.shape[0] == len(clustering_labels[0])
    

    if metrics_kwargs is not None:
        default_metrics.update(metrics_kwargs)
       
    df_metrics = []
    for i, label in enumerate(clustering_labels):

        metrics_ = dict()
        
        k = dataframe['n_states'].values[i]
        
        logger.info('Calculating metrics for k: %s' %(str(k)))
        
        for metric_name, metric_function in default_metrics.items():
            
            logger.info(" - Calculating %s" %(metric_name))
            
            if metric_name == 'Krzanowski-Lai':
                if i == len(clustering_labels) - 1:
                    prev_labels = clustering_labels[i-1]
                    next_labels = np.arange(0, label.shape[0])
                elif k == 2:
                    prev_labels = np.zeros_like(label)
                    next_labels = clustering_labels[i+1]
                else:   
                    prev_labels = clustering_labels[i-1]
                    next_labels = clustering_labels[i+1]
                    
                m_ = metric_function(X,
                                     label,
                                     previous_labels=prev_labels,
                                     next_labels=next_labels,
                                     precomputed=False)
                
            else:
                m_ = metric_function(X, label)
            
            metrics_ = {
                'name': metric_name,
                'value': m_,
                'k': k
            }
            metrics_.update(fixed_variables)

            df_metrics.append(metrics_.copy())

    return pd.DataFrame(df_metrics)


def calculate_centroids(dataframe):
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

    centroids = []
    for idx, row in dataframe.iterrows():
        X = row['X']
        labels = row['labels']
        centroid = np.array([X[labels == l].mean(0) for l in np.unique(labels)])
        centroids.append(centroid)
    
    dataframe['centroids'] = np.array(centroids)
        
    return dataframe


def find_best_k(dataframe):
    """[summary]
    
    Parameters
    ----------
    dataframe : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    """

    from sklearn.preprocessing import MinMaxScaler

    metrics_name = np.unique(dataframe['name'])

    drop_keys = ['name', 'value', 'k']
    keys = [k for k in dataframe.keys() if k not in drop_keys] 
    
    df_list = []
    for name in metrics_name:
        df = filter_dataframe(dataframe, name=[name])
        df = df.sort_values('k')
        values = df['value'].values
        k_step = df['k'].values

        if name in ['Silhouette', 'Krzanowski-Lai', 'Index I']:
            guessed_cluster = np.nonzero(np.max(values) == values)[0][0] + k_step[0]
        else:
            data = np.vstack((k_step, values)).T

            tvalues = MinMaxScaler().fit_transform(data)
            theta = np.arctan2(tvalues.T[1][-1] - tvalues.T[1][0],
                               k_step[-1] - k_step[0])

            co = np.cos(theta)
            si = np.sin(theta)
            rotation_matrix = np.array(((co, -si), (si, co)))
            # rotate data vector
            data = np.vstack((k_step, tvalues.T[1])).T
            data = data.dot(rotation_matrix)

            fx = np.max
            if name != 'Global Explained Variance':
                fx = np.min
            guessed_cluster = np.nonzero(data[:, 1] == fx(data[:, 1]))[0][0] + k_step[0]

        result = {
            'name': name,
            'guess': guessed_cluster,
        }

        result.update(dict(zip(keys, df[keys].values[0])))

        df_list.append(result)
    
    return pd.DataFrame(df_list)


def dynamics_errors(dataframe):
    import itertools

    def _parallel(row, i):
        true_dynamics = row['targets']
        clustering_dynamics = row['dynamics']
        cluster_idx = np.unique(clustering_dynamics)
    
        cluster_binary = np.zeros((len(cluster_idx), len(clustering_dynamics)), dtype=np.bool)
        for i in cluster_idx: 
            cluster_binary[i] = clustering_dynamics == i

        permuted_dynamics = np.zeros_like(clustering_dynamics)
        min_error = 1e6
        perm = itertools.permutations(cluster_idx)
        for p in perm:
            x, y = np.nonzero(cluster_binary[p, :])
            permuted_dynamics[y] = x
            error = np.sum(np.abs(permuted_dynamics == true_dynamics))

            if error < min_error:
                min_error = error
                min_p = p

        return min_error/len(clustering_dynamics)

    errors = Parallel(n_jobs=-1, verbose=1)(delayed(_parallel)(row, i) \
        for i, (idx, row) in enumerate(dataframe.iterrows()))

    dataframe['dynamics_errors'] = np.array(errors)

    return dataframe


def state_errors(dataframe):

    if 'centroids' not in dataframe.keys():
        dataframe = calculate_centroids(dataframe)
    
    similarity_state = []
    for i, (idx, row) in enumerate(dataframe.iterrows()):
        
        true_centers = row['states']
        triu = np.triu_indices(true_centers.shape[1], k=1)
        true_vector = np.array([m[triu] for m in true_centers])
        
        est_centers = row['centroids']
        
        similarity = np.dot(true_vector, est_centers.T)
        
        t_similarity = np.diag(np.sqrt(np.dot(true_vector, true_vector.T)))
        c_similarity = np.diag(np.sqrt(np.dot(est_centers, 
                                              est_centers.T)))

        idx_true, idx_clustering = np.nonzero(similarity == similarity.max(0))
        
        norm_similarity = []
        for x, y in zip(idx_true, idx_clustering):
            n = similarity[x, y]/(t_similarity[x] * c_similarity[y])
            norm_similarity.append(n)

        norm_similarity = np.array(norm_similarity).mean()

        similarity_state.append(norm_similarity)

    dataframe['centroid_similarity'] = np.array(similarity_state)

    return dataframe
