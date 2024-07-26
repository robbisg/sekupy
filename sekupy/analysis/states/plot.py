import matplotlib.pyplot as pl
import numpy as np
import os
from sekupy.utils.matrix import copy_matrix, array_to_matrix
from sklearn.manifold.mds import MDS
from mvpa_itab.conn.states.utils import get_centroids

def plot_states_matrices(X, 
                         labels,
                         node_number=[6,5,8,10,4,5,7], 
                         node_networks=['DAN','VAN','SMN','VIS','AUD','LAN','DMN'],
                         use_centroid=False,
                         n_cols=3,
                         save_fig=False,
                         save_path="/media/robbis/DATA/fmri/movie_viviana",
                         save_name_condition=None,
                         **kwargs
                         ):
    """
    Plots the centroids in square matrix form.
    It could be used with original data and labels but also 
    with the original centroids if you set use_centroids as True. 
    
    """

    position = [sum(node_number[:i+1]) for i in range(len(node_number))]
    
    if not use_centroid:
        centroids = get_centroids(X, labels)(X, labels)
        n_states = len(np.unique(labels))
    else:
        centroids = X.copy()
        n_states = X.shape[0]
    
    
    position_label = [-0.5+position[i]-node_number[i]/2. for i in range(len(node_number))]
    n_rows = np.ceil(n_states / float(n_cols))

    fig = pl.figure()
        
    for i in np.arange(n_states):
        
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        matrix_ = copy_matrix(array_to_matrix(centroids[i]), diagonal_filler=0)
        n_nodes = matrix_.shape[0]
        ax.imshow(matrix_, interpolation='nearest', vmin=0, cmap=pl.cm.bwr)
        for _, end_network in zip(node_networks, position):
            ax.vlines(end_network-0.5, -0.5, n_nodes-0.5)
            ax.hlines(end_network-0.5, -0.5, n_nodes-0.5)
        
        ax.set_title('State '+str(i+1))
        ax.set_xticks(position_label)
        ax.set_xticklabels(node_networks)
        ax.set_yticks(position_label)
        ax.set_yticklabels(node_networks)
        
        #pl.colorbar()
        
    if save_fig:
        fname = "%s_state_%s.png" % (str(save_name_condition), str(i+1))
        fig.savefig(os.path.join(save_path, fname))
    
    pl.close('all')
    return fig


def plot_center_matrix(X, clustering, n_cluster=5, **kwargs):
    
    
    configuration = {
                     'node_number':[6,5,8,10,4,5,7],
                     'node_networks':['DAN','VAN','SMN','VIS','AUD','LAN','DMN'],
                     'save_fig':True,
                     'save_path':"/media/robbis/DATA/fmri/movie_viviana",
                     'save_name_condition':None
                     
                     }
    
    
    configuration.update(**kwargs)
    
    node_number = configuration['node_number']
    node_networks = configuration['node_networks']
    
    position = [sum(node_number[:i+1]) for i in range(len(node_number))]
    position_label = [-0.5+position[i]-node_number[i]/2. for i in range(len(node_number))]
    
    matrix_indices = np.arange(n_cluster**2).reshape(n_cluster, n_cluster) + 1
    
    fig = pl.figure(figsize=(25,20))
    for i in range(n_cluster-1):
        centers = get_centroids(X, clustering[i])
        for j, matrix in enumerate(centers):
            pos = matrix_indices[j,i+1]
            ax = fig.add_subplot(n_cluster, n_cluster, pos)
            matrix = copy_matrix(array_to_matrix(matrix), diagonal_filler=0)
            total_nodes = matrix.shape[0]
            ax.imshow(matrix, interpolation='nearest', vmin=0)
            for _, n_nodes in zip(node_networks, position):
                ax.vlines(n_nodes-0.5, -0.5, total_nodes-0.5)
                ax.hlines(n_nodes-0.5, -0.5, total_nodes-0.5)
            ax.set_xticks(position_label)
            ax.set_xticklabels(node_networks)
            ax.set_yticks(position_label)
            ax.set_yticklabels(node_networks)
            
            
    return fig


def plot_condition_centers(X, labels, **kwargs):
    
    
    configuration = {
                     'node_number':[6,5,8,10,4,5,7],
                     'node_networks':['DAN','VAN','SMN','VIS','AUD','LAN','DMN'],
                     'save_fig':True,
                     'save_path':"/media/robbis/DATA/fmri/movie_viviana",
                     'save_name_condition':None,
                     'vmax':1                     
                     }
    
    
    configuration.update(**kwargs)
    vmax = configuration['vmax']
    node_number = configuration['node_number']
    node_networks = configuration['node_networks']
    centroids = get_centroids(X, labels)
    position = [sum(node_number[:i+1]) for i in range(len(node_number))]
    position_label = [-0.5+position[i]-node_number[i]/2. for i in range(len(node_number))]
    
    n_rows = np.floor(np.sqrt(len(np.unique(labels))))
    n_cols = np.ceil(len(np.unique(labels))/n_rows)
    
    fig = pl.figure(figsize=(16,13))
    for j, matrix in enumerate(centroids):
        ax = fig.add_subplot(n_rows, n_cols, j+1)
        matrix = copy_matrix(array_to_matrix(matrix), diagonal_filler=0)
        total_nodes = matrix.shape[0]
        ax.imshow(matrix, interpolation='nearest', vmin=0, vmax=vmax)
        for _, n_nodes in zip(node_networks, position):
            ax.vlines(n_nodes-0.5, -0.5, total_nodes-0.5)
            ax.hlines(n_nodes-0.5, -0.5, total_nodes-0.5)
        ax.set_xticks(position_label)
        ax.set_xticklabels(node_networks, rotation=45)
        ax.set_yticks(position_label)
        ax.set_yticklabels(node_networks)
        
    return fig


    
def plot_metrics(metrics_, metric_names, k_step):
    """
    Plots the clustering metrics.
    """
    
    fig = pl.figure(figsize=(12,10))
    n_rows = np.ceil(len(metric_names)/2.)
    for i, m in enumerate(metrics_.T):
        ax = fig.add_subplot(int(n_rows), 2, i+1)
        ax.plot(k_step, m, '-o')
        ax.set_title(list(metric_names)[i])
        ax.set_xticks(k_step)
        ax.set_xticklabels(k_step)
        
    #pl.close('all')
    return fig
    


def plot_dynamics(state_dynamics, condition, path, **kwargs):
    """
    Plot the dynamics of the states for each session.
    """
    
    fname_cfg = {'prefix':'',
                 'suffix':''}
    
    fname_cfg.update(kwargs)
    
    
    for i, ts in enumerate(state_dynamics):
        _ = pl.figure(figsize=(18,10))
        for j, sts in enumerate(ts):
            pl.plot(sts, label=str(j+1))
        pl.legend()
        
        pl.xlabel("Time")
        pl.ylabel("Dissimilarity")
        
        fname = "%scondition_%s_session_%02d_dynamics%s.png" % (fname_cfg['prefix'],
                                                               condition,
                                                               i+1,
                                                               fname_cfg['suffix'],                                                               
                                                               )
        fname = os.path.join(path, fname)
        pl.savefig(fname)
        
    pl.close('all') 



def plot_frequencies(state_frequency, condition, path):
    """
    Plots the frequency of the state
    """
    
    for i, ts in enumerate(state_frequency):
        _ = pl.figure(figsize=(12,10))
        freq = ts[0]
        values = ts[1]
        for j, f in enumerate(values):
            pl.plot(freq[1:150], f[1:150], label=str(j+1))
        pl.legend()
        
        pl.xlabel("Frequency")
        pl.ylabel("Power")
        
        fname = os.path.join(path, "condition_%s_session_%02d_freq.png" % (condition, i+1))
        pl.savefig(fname)
        
    pl.close('all')
       
    

def plot_positions(dict_centroids, **kwargs):
    
    configuration = {
                     "conditions": ['movie', 'scramble', 'rest'],
                     "colors": ['red', 'blue', 'green'],
                     "save_fig":False,
                     "path":None,
                     }
    
    configuration.update(kwargs)
        
    X_c = [v for _, v in dict_centroids.iteritems()]
    
    X_c = np.vstack(X_c)
    
    pos = MDS(n_components=2).fit_transform(X_c)

    color = dict(zip(configuration['conditions'], configuration['colors']))
    
    fig = pl.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    for i, c in enumerate(configuration["conditions"]):
        pos1 = i*5
        pos2 = (i+1)*5
        ax.scatter(pos[pos1:pos2, 0], pos[pos1:pos2, 1], c=color[c], s=150)
        for j, (x, y) in enumerate(pos[pos1:pos2]):
            ax.annotate(str(j+1), (x,y), fontsize=15)
    
    if configuration["save_fig"]:
        fname = os.path.join(configuration['path'], "mds.png")
        pl.savefig(fname)
        
    return fig


def plot_predicted_matrices(df):
    rows = len(df) + 1
    df = df.sort_values('n_states')
    cols = np.max(np.unique(df['n_states']))

    fig, axes = pl.subplots(rows, cols)

    for i, (r, row) in enumerate(df.iterrows()):
        for j, c in enumerate(row['centroids']):
            matrix = array_to_matrix(c, diagonal_filler=1, copy=False)
            axes[i+1, j].imshow(matrix, cmap=pl.cm.magma)


    states = df['states'].values[0]

    for j, s in enumerate(states):
        axes[0, j].imshow(s, cmap=pl.cm.magma)
    
